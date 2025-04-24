import argparse
import random
from typing import Set, Dict, List, Tuple
import collections
from torch.distributions import Normal
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------  GAME  ----------------
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import (
    Tier1NDaysNCampaignsAgent,
)
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle

# ════════════════════════════════════════════════════════════════════
# 1.  UTILITIES ──────────  replay buffer & normalisation helpers
# ════════════════════════════════════════════════════════════════════
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # type: ignore
        self.buffer[self.pos] = Transition(*args)  # type: ignore
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        trans = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*trans))
        # stack into tensors
        return Transition(
            torch.stack(batch.state).to(self.device),
            torch.stack(batch.action).to(self.device),
            torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.stack(batch.next_state).to(self.device),
            torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════
# 2.  NETWORKS  ──────────  actor & Q-functions (twin critics)
# ════════════════════════════════════════════════════════════════════
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# The Actor Network -> Gaussian Policy
class GaussianPolicy(nn.Module):
    """
    Actor outputs mean & log_std;   action = tanh(μ + σ⊙ε)
    Two-dimensional action:   [bid_multiplier , limit_multiplier] ∈ (0,1)
    The tanh is mapped to (0,1) then later to (0.5,1.5) inside the agent.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(state)
        std = log_std.exp()
        eps = torch.randn_like(std)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)
        # log π(a|s)  (tanh-squash adjustment)
        log_prob = (
            Normal(mu, std).log_prob(pre_tanh)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob

# The Critic Network -> Q-function
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return self.net(torch.cat([state, action], dim=-1))


# ════════════════════════════════════════════════════════════════════
# 3.  AGENT  ──────────  SAC implementation for AdX bidding
# ════════════════════════════════════════════════════════════════════
class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    STATE_DIM = 8
    ACTION_DIM = 2

    def __init__(self, ckpt_path: str = None, inference: bool = False):
        super().__init__()
        self.name = "SAC_Ad_Bidding_Agent"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- networks
        hd = 128 # hidden dimension
        
        # one actor and two critics (twin Q-functions)
        # stabilize training by using two critics
        self.actor = GaussianPolicy(self.STATE_DIM, hd, self.ACTION_DIM).to(self.device)
        self.q1 = QNetwork(self.STATE_DIM, self.ACTION_DIM, hd).to(self.device)
        self.q2 = QNetwork(self.STATE_DIM, self.ACTION_DIM, hd).to(self.device)
        self.q1_target = QNetwork(self.STATE_DIM, self.ACTION_DIM, hd).to(self.device)
        self.q2_target = QNetwork(self.STATE_DIM, self.ACTION_DIM, hd).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # --- optimisers
        lr = 3e-4
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        # --- entropy temperature
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -float(self.ACTION_DIM)

        # --- other SAC hyper-parameters
        self.gamma = 0.99
        self.tau = 0.005 # soft target update
        self.batch_size = 256  # replay buffer batch size
        self.update_after = 1000          # steps before first gradient update
        self.updates_per_step = 1          
        self.buffer = ReplayBuffer(50_000)
        self.total_steps = 0

        # --- trajectory trackers (previous step → reward)
        self.prev_state: torch.Tensor | None = None
        self.prev_action: torch.Tensor | None = None

        # profit / quality snapshot to compute daily reward
        self.prev_profit = 0.0
        self.prev_quality = 1.0
        self.prev_reach: Dict[int, int] = {}
        
        self.inference = inference
        if ckpt_path is not None and inference:
            # load checkpoint if provided
            self._load_checkpoint(ckpt_path)
            
        self.MAX_CAMPAIGNS = 10          # expected upper-bound in a 10-agent game
        self.GAME_LENGTH   = 10.0        # days


    # ────────────────────────────────────────────────────────────────
    # auxiliary: build state vector
    def _state_vector(self) -> torch.Tensor:
        """Return an 8-dim continuous state for the actor/critics.

        [0]  day_norm                    ∈ [0,1]
        [1]  n_campaigns_norm            ∈ [0,1]
        [2]  avg_progress                ∈ [0,1]
        [3]  min_progress                ∈ [0,1]
        [4]  max_progress                ∈ [0,1]
        [5]  urgency_weighted_progress   ∈ [0,1]   (more weight if few days left)
        [6]  avg_remaining_budget_ratio  ∈ [0,1]
        [7]  quality_score               ∈ [0,1.38]  (upper bound ≈ effective-reach limit)
        """
        day_norm = self.get_current_day() / self.GAME_LENGTH
        active   = list(self.get_active_campaigns())
        n_act    = len(active)

        # ── Empty default ───────────────────────────────────────────────
        if n_act == 0:
            return torch.tensor(
                [day_norm, 0, 0, 0, 0, 0, 0, self.get_quality_score()],
                dtype=torch.float32,
                device=self.device,
            )

        # ── Per-campaign features ───────────────────────────────────────
        progresses, budget_ratios, urg_weights = [], [], []
        for c in active:
            prog        = self.get_cumulative_reach(c) / max(1, c.reach)
            rem_budget  = (c.budget - self.get_cumulative_cost(c)) / max(1.0, c.budget)
            days_left   = max(1, c.end_day - self.get_current_day() + 1)
            weight      = 1.0 / days_left                         # urgency ↑ as days_left ↓

            progresses.append(prog)
            budget_ratios.append(rem_budget)
            urg_weights.append(weight)

        # ── Aggregations ────────────────────────────────────────────────
        avg_prog  = sum(progresses) / n_act
        min_prog  = min(progresses)
        max_prog  = max(progresses)

        # weighted progress (normalised by sum of weights)
        w_prog    = sum(p * w for p, w in zip(progresses, urg_weights)) / sum(urg_weights)

        avg_rem_budget = sum(budget_ratios) / n_act
        n_cmp_norm     = n_act / self.MAX_CAMPAIGNS

        state_vec = torch.tensor(
            [
                day_norm,
                n_cmp_norm,
                avg_prog,
                min_prog,
                max_prog,
                w_prog,
                avg_rem_budget,
                self.get_quality_score(),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return state_vec

    # ────────────────────────────────────────────────────────────────
    # RL hooks
    def on_new_game(self) -> None:
        # flush episode trackers
        self.prev_state, self.prev_action = None, None
        self.prev_profit = 0.0
        self.prev_quality = 1.0
        self.prev_reach.clear()

    # ────────────────────────────────────────────────────────────────
    # AD BIDS  (one call per day  →  SAC step)
    def get_ad_bids(self) -> Set[BidBundle]:
        # ---------- 1. build today’s state  ----------
        state = self._state_vector()

        # ---------- 2. compute reward for PREVIOUS action ----------
        if self.prev_state is not None and self.prev_action is not None:
            # daily reward components
            r_profit = self.get_cumulative_profit() - self.prev_profit
            # campaign progress
            active = self.get_active_campaigns()
            prog_improvements = []
            for c in active:
                cur = self.get_cumulative_reach(c)
                before = self.prev_reach.get(c.uid, 0)
                prog_improvements.append((cur - before) / max(1, c.reach))
                self.prev_reach[c.uid] = cur
            r_progress = sum(prog_improvements) / len(prog_improvements) if prog_improvements else 0.0
            # quality change
            r_quality = self.get_quality_score() - self.prev_quality

            reward = (
                1.0 * r_profit + 50.0 * r_progress + 20.0 * r_quality
            )  # weights chosen heuristically

            done = self.get_current_day() - 1 >= 10  # prev day terminal?
            self.buffer.push(
                self.prev_state,
                self.prev_action,
                reward,
                state,
                float(done),
            )

        # ---------- 3. select new action with exploration ----------
        with torch.no_grad():
            if self.inference:
                mu, _ = self.actor(state.unsqueeze(0))
                action_tanh = torch.tanh(mu)        # deterministic
            else:
                action_tanh, _ = self.actor.sample(state.unsqueeze(0))
        
        action = action_tanh.squeeze(0)

        # rescale to (0,1) → (0.5,1.5) multipliers
        bid_mul = 0.5 + 0.5 * action[0].item()
        limit_mul = 0.5 + 0.5 * action[1].item()
        
        # print(
        #     f"Day {self.get_current_day()}: "
        #     f"Action: {action.tolist()}, "
        #     f"Bid multiplier: {bid_mul:.2f}, "
        #     f"Limit multiplier: {limit_mul:.2f}"
        # )

        # ---------- 4. translate into BidBundles ----------
        bundles: Set[BidBundle] = set()
        for campaign in self.get_active_campaigns():
            cum_reach = self.get_cumulative_reach(campaign)
            cum_cost = self.get_cumulative_cost(campaign)
            
            remaining_reach = campaign.reach - cum_reach
            remaining_budget = campaign.budget - cum_cost
            if remaining_reach <= 0 or remaining_budget <= 0:
                continue
            
            rem_reach = campaign.reach - cum_reach
            rem_budget = max(0.0, campaign.budget - cum_cost)

            baseline_bid = max(0.1, rem_budget / rem_reach)
            bid_per_imp = baseline_bid * bid_mul
            spend_limit = rem_budget * limit_mul

            bid_entry = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=bid_per_imp,
                bid_limit=spend_limit,
            )
            bundle = BidBundle(campaign.uid, spend_limit, {bid_entry})
            # print(
            #     f"Campaign {campaign.uid}: "
            #     f"Bid per impression: {bid_per_imp:.2f}, "
            #     f"Spending limit: {spend_limit:.2f}, "
            #     f"Reach: {cum_reach:.2f} / {campaign.reach:.2f}, "
            #     f"Budget: {cum_cost:.2f} / {campaign.budget:.2f}"
            # )            
            
            bundles.add(bundle)

            # initial reach snapshot for reward calc
            self.prev_reach.setdefault(campaign.uid, cum_reach)

        # ---------- 5. SAC gradient updates ----------
        self._learn()

        # ---------- 6. store snapshots for next step ----------
        self.prev_state = state
        self.prev_action = action
        self.prev_profit = self.get_cumulative_profit()
        self.prev_quality = self.get_quality_score()
        
        # print(
        #     f"Profit: {self.prev_profit:.2f}, "
        #     f"Quality: {self.prev_quality:.2f}"
        # )

        return bundles

    # ────────────────────────────────────────────────────────────────
    # CAMPAIGN BIDS  (rule based, unchanged)
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        day = self.get_current_day()
        Q = self.get_quality_score()
        for c in campaigns_for_auction:
            min_bid = 0.1 * c.reach
            max_bid = c.reach
            base = min_bid if Q >= 1 else min_bid + (max_bid - min_bid) * (1 - Q)
            if day >= 7:
                base *= 1.2
            bids[c] = self.clip_campaign_bid(c, base)
        return bids

    # ────────────────────────────────────────────────────────────────
    #  SAC optimization step(s)
    def _learn(self):
        if self.inference:
            return
        
        self.total_steps += 1
        if len(self.buffer) < self.update_after or self.total_steps % 1 != 0:
            return

        for _ in range(self.updates_per_step):
            batch = self.buffer.sample(self.batch_size)
            s, a, r, s2, d = (
                batch.state,
                batch.action,
                batch.reward,
                batch.next_state,
                batch.done,
            )

            # Q targets
            with torch.no_grad():
                a2, logp2 = self.actor.sample(s2)
                q1_t = self.q1_target(s2, a2)
                q2_t = self.q2_target(s2, a2)
                q_min = torch.min(q1_t, q2_t) - self.alpha * logp2
                y = r + self.gamma * (1 - d) * q_min

            # critic losses
            q1_pred = self.q1(s, a)
            q2_pred = self.q2(s, a)
            q1_loss = F.mse_loss(q1_pred, y)
            q2_loss = F.mse_loss(q2_pred, y)

            self.q1_opt.zero_grad()
            q1_loss.backward()
            self.q1_opt.step()

            self.q2_opt.zero_grad()
            q2_loss.backward()
            self.q2_opt.step()

            # actor loss
            a_new, logp = self.actor.sample(s)
            q1_new = self.q1(s, a_new)
            q2_new = self.q2(s, a_new)
            actor_loss = (self.alpha * logp - torch.min(q1_new, q2_new)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # temperature loss
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # soft-update targets
            with torch.no_grad():
                for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                    p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                    p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # property helper
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def _save_checkpoint(self, fname: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "log_alpha": self.log_alpha.detach(),
            },
            fname,
        )

    def _load_checkpoint(self, fname: str):
        ckpt = torch.load(fname, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        # hard-sync targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())


# ════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP (unchanged – now drives SAC)
# ════════════════════════════════════════════════════════════════════
def train(eps: int, ckpt: str, ma_window: int = 100):
    ag = MyNDaysNCampaignsAgent()
    sim = AdXGameSimulator()
    foes = [Tier1NDaysNCampaignsAgent(name=f"T1-{i}") for i in range(9)]
    
    ma_queue = deque(maxlen=ma_window)
    
    for ep in range(1, eps + 1):
        sim.run_simulation([ag] + foes, num_simulations=1)
        
        # 2. get final profit and update moving average
        profit = ag.get_cumulative_profit()
        ma_queue.append(profit)
        ma = sum(ma_queue) / len(ma_queue)
        
        # 3. print stats
        print(
            f"[train] Ep {ep}/{eps}  "
            f"Buf={len(ag.buffer):5d}  "
            f"Profit={profit:7.2f}  "
            f"MA{ma_window}={ma:7.2f}"
        )
        
        ag.on_new_game()
        
    ag._save_checkpoint(ckpt)
    print("✔ saved", ckpt)


def evaluate_sac(ckpt_file: str, num_runs: int = 500):
    eval_agent = MyNDaysNCampaignsAgent(ckpt_path=ckpt_file, inference=True)
    sim        = AdXGameSimulator()
    foes       = [Tier1NDaysNCampaignsAgent(name=f"T1-{i}") for i in range(9)]

    sim.run_simulation([eval_agent] + foes, num_simulations=num_runs)


# -------------  CLI -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_eps", type=int, default=1000)
    parser.add_argument("--ckpt",      type=str, default="sac_adx.pth")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_runs", type=int, default=500)
    args = parser.parse_args()

    sac_agent = MyNDaysNCampaignsAgent()

    if args.eval_only:
        evaluate_sac(args.ckpt, args.eval_runs)
    else:
        train(args.train_eps, args.ckpt)
        evaluate_sac(args.ckpt, args.eval_runs)
