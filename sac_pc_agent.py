from __future__ import annotations
import math, random, argparse, collections
from typing import Dict, List, Set, Tuple

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Normal

# ─────────────  GAME API  ─────────────
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

# ══════════════════════════════════════
# 1. replay buffer
# ══════════════════════════════════════
Transition = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    def __init__(self, cap: int = 50_000):
        self.cap, self.data, self.pos, self.device = cap, [], 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(self, *args):
        if len(self.data) < self.cap:
            self.data.append(None)  # type: ignore
        self.data[self.pos] = Transition(*args)  # type: ignore
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch: int) -> Transition:
        s = random.sample(self.data, batch)
        b = Transition(*zip(*s))
        return Transition(
            torch.stack(b.state).to(self.device),
            torch.stack(b.action).to(self.device),
            torch.tensor(b.reward, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.stack(b.next_state).to(self.device),
            torch.tensor(b.done, dtype=torch.float32).unsqueeze(1).to(self.device),
        )

    def __len__(self): 
        return len(self.data)


# ══════════════════════════════════════
# 2. networks
# ══════════════════════════════════════
LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, hidden: int = 128, a_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)

    def forward(self, s):
        z = self.net(s)
        mu = self.mu(z)
        log_std = torch.clamp(self.log_std(z), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        eps = torch.randn_like(std)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)
        logp = (
            Normal(mu, std).log_prob(pre_tanh) - torch.log1p(-a.pow(2) + 1e-6)
        ).sum(-1, keepdim=True)
        return a, logp


class QNet(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 128):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a): return self.f(torch.cat([s, a], -1))


# ══════════════════════════════════════
# 3. SAC agent with per-campaign rewards
# ══════════════════════════════════════
MAX_CAMPAIGNS = 10
GAME_LENGTH = 10.0

class SACPerCampaign(NDaysNCampaignsAgent):
    S_DIM, A_DIM = 8, 2

    # memory entry for campaign c chosen yesterday
    MemEntry = collections.namedtuple(
        "MemEntry", ("state", "action", "prev_reach", "prev_cost", "end_day", "uid")
    )

    def __init__(self, ckpt: str | None = None, inference=False):
        super().__init__()
        self.name, self.device = "SAC_PC", torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hid = 128
        self.actor = GaussianPolicy(self.S_DIM, hid, self.A_DIM).to(self.device)
        self.q1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.q2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1.load_state_dict(self.q1.state_dict())
        self.t2.load_state_dict(self.q2.state_dict())

        lr = 3e-4
        self.o_actor = optim.Adam(self.actor.parameters(), lr)
        self.o_q1 = optim.Adam(self.q1.parameters(), lr)
        self.o_q2 = optim.Adam(self.q2.parameters(), lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.o_alpha = optim.Adam([self.log_alpha], lr)

        self.gamma, self.tau = 0.99, 0.005
        self.batch, self.update_after, self.updates_ps = 128, 100, 4
        self.target_entropy = -float(self.A_DIM)
        self.buffer = ReplayBuffer(50_000)
        self.total_steps = 0
        self.inference = inference

        # per-campaign log of previous-day choices
        self.mem: Dict[int, SACPerCampaign.MemEntry] = {}

        if ckpt:
            self._load(ckpt)
            self.inference = True

    # ── α helper ──
    @property
    def alpha(self): 
        return self.log_alpha.exp()

    # ── checkpoint ──
    def _save(self, f): 
        torch.save(
        {"actor": self.actor.state_dict(), 
         "q1": self.q1.state_dict(), 
         "q2": self.q2.state_dict(),
         "log_alpha": self.log_alpha.detach()
        }, f)

    def _load(self, f):
        ck = torch.load(f, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
        self.t1.load_state_dict(ck["q1"])
        self.t2.load_state_dict(ck["q2"])
        self.log_alpha.data.copy_(ck["log_alpha"])

    # ── campaign-level state ───────────────────────────────────────────
    def _state_c(self, c: Campaign) -> torch.Tensor:
        """
        Return an 8-dim vector describing the current campaign *c* in context:
            [0] day_norm                    ∈ [0,1]
            [1] n_campaigns_norm            ∈ [0,1]
            [2] avg_progress_all            ∈ [0,1]
            [3] min_progress_all            ∈ [0,1]
            [4] max_progress_all            ∈ [0,1]
            [5] this_progress * urgency     ∈ [0,1]
            [6] remaining_budget_ratio_c    ∈ [0,1]
            [7] quality_score               ∈ [0,1.38]
        """
        # ----- global scalars -------------------------------------------------
        day_norm = self.get_current_day() / GAME_LENGTH
        actives  = list(self.get_active_campaigns())
        n_norm   = len(actives) / MAX_CAMPAIGNS if MAX_CAMPAIGNS else 0.0

        # ----- progress stats across campaigns --------------------------------
        progresses: List[float] = []
        for camp in actives:
            p = self.get_cumulative_reach(camp) / max(1, camp.reach)
            progresses.append(p)

        if progresses:
            avg_prog = sum(progresses) / len(progresses)
            min_prog = min(progresses)
            max_prog = max(progresses)
        else:           # degenerate case: no active campaigns
            avg_prog = min_prog = max_prog = 0.0

        # ----- per-campaign features ------------------------------------------
        cum_reach  = self.get_cumulative_reach(c)
        cum_cost   = self.get_cumulative_cost(c)
        prog_c     = cum_reach / max(1, c.reach)
        rem_budget = (c.budget - cum_cost) / max(1.0, c.budget)

        days_left  = max(1, c.end_day - self.get_current_day() + 1)
        urgency    = 1.0 / days_left                     # ↑ when few days remain

        # ----- build tensor ----------------------------------------------------
        return torch.tensor(
            [
                day_norm,          # [0]
                n_norm,            # [1]
                avg_prog,          # [2]
                min_prog,          # [3]
                max_prog,          # [4]
                prog_c * urgency,  # [5]
                rem_budget,        # [6]
                self.get_quality_score(),  # [7]
            ],
            dtype=torch.float32,
            device=self.device,
        )


    # ── reset episode mem ──
    def on_new_game(self):
        self.mem.clear()

    # ── main bidding function ──
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles: Set[BidBundle] = set()

        # 1.  compute reward for campaigns we bid on yesterday
        for uid, m in list(self.mem.items()):
            # It might have ended and vanished from active list; retrieve via stored uid
            camp = next((c for c in self.get_active_campaigns() if c.uid == uid), None)
            
            if camp is None:
                # use last known state; reward is zero because reach & cost no longer change
                self.buffer.push(m.state, m.action, 0.0,
                                torch.zeros_like(m.state), 1.0)
                self.mem.pop(uid, None)
                continue

            new_reach = self.get_cumulative_reach(camp)
            new_cost = self.get_cumulative_cost(camp)
            
            delta_cost = new_cost - m.prev_cost
            eff_prev = self.effective_reach(m.prev_reach, camp.reach)
            eff_new = self.effective_reach(new_reach, camp.reach)
            reward = (eff_new - eff_prev) * camp.budget - delta_cost
            
            # NaN / inf guard
            if not math.isfinite(reward):
                reward = 0.0

            done = float(self.get_current_day() > camp.end_day)
            next_state = self._state_c(camp) if not done else torch.zeros(m.state, dtype=torch.float32, device=self.device)

            self.buffer.push(m.state, 
                             m.action, 
                             reward, 
                             next_state, 
                             done)

            # remove entry if campaign ended
            if done: 
                self.mem.pop(uid, None)
            else:
                # update prev snapshot for tomorrow
                self.mem[uid] = m._replace(prev_reach=new_reach, prev_cost=new_cost)

        # 2. choose action for each currently active campaign
        for c in self.get_active_campaigns():
            cum, cost = self.get_cumulative_reach(c), self.get_cumulative_cost(c)
            
            if cum >= c.reach or cost >= c.budget:  # no need to bid
                self.mem.pop(c.uid, None)
                continue

            s_c = self._state_c(c).unsqueeze(0)
            
            with torch.no_grad():
                if self.inference:
                    mu, _ = self.actor(s_c)
                    action = torch.tanh(mu)
                else:
                    action, _ = self.actor.sample(s_c)
                    
            action = action.squeeze(0)
            bid_mul, lim_mul = 0.5 + 0.5 * action[0].item(), 0.5 + 0.5 * action[1].item()

            rem_reach = max(1, c.reach - cum)
            rem_budget = max(1e-6, c.budget - cost)
            
            base = rem_budget / rem_reach
            base = float(min(max(base, 0.1), rem_budget)) # clip to [0.1, rem_budget]
            
            bid = float(min(max(base * bid_mul, 0.1), rem_budget)) # clip to [0.1, rem_budget]
            limit = float(min(max(rem_budget * lim_mul, bid), rem_budget)) # clip to [bid, rem_budget]
            
            # final sanity check
            if not (math.isfinite(bid) and math.isfinite(limit)):
                continue      # skip insane numbers gracefully

            bid_entry = Bid(self, c.target_segment, bid, limit)
            bundles.add(BidBundle(c.uid, limit, {bid_entry}))

            # store snapshot for tomorrow’s reward
            self.mem[c.uid] = self.MemEntry(
                state=s_c.squeeze(0),
                action=action,
                prev_reach=cum,
                prev_cost=cost,
                end_day=c.end_day,
                uid=c.uid,
            )
            
        # If no bids were produced we still need to learn
        if not bundles:
            # to keep replay buffer fresh, call _learn() anyway
            self._learn()
            return set()

        # learn from replay buffer
        self._learn()
        return bundles
    
    # ── campaign-auction bids (simple) ──
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

    # ── SAC update ──
    def _learn(self):
        if self.inference or len(self.buffer) < self.update_after: 
            return
        
        for _ in range(self.updates_ps):
            s, a, r, s2, d = self.buffer.sample(self.batch)

            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                y = r + self.gamma * (1 - d) * (torch.min(self.t1(s2, a2), self.t2(s2, a2)) - self.alpha * lp2)

            # critics
            for qnet, opt, target in ((self.q1, self.o_q1, self.t1), (self.q2, self.o_q2, self.t2)):
                loss = F.mse_loss(qnet(s, a), y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(qnet.parameters(), 5); opt.step()
                with torch.no_grad():
                    for p, tp in zip(qnet.parameters(), target.parameters()):
                        tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

            # actor
            an, lpn = self.actor.sample(s)
            actor_loss = (self.alpha * lpn - torch.min(self.q1(s, an), self.q2(s, an))).mean()
            self.o_actor.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), 5); self.o_actor.step()

            # α
            alpha_loss = -(self.log_alpha * (lpn + self.target_entropy).detach()).mean()
            self.o_alpha.zero_grad(); alpha_loss.backward(); self.o_alpha.step()


# ══════════════════════════════════════
# 4. training / evaluation helpers
# ══════════════════════════════════════
def train(eps: int, ckpt: str):
    ag = SACPerCampaign()
    sim = AdXGameSimulator()
    foes = [Tier1NDaysNCampaignsAgent(name=f"T1-{i}") for i in range(9)]
    for ep in range(1, eps + 1):
        sim.run_simulation([ag] + foes, num_simulations=1)
        ag.on_new_game()
        if ep % 50 == 0: 
            print(f"[train] {ep}/{eps}, buf={len(ag.buffer)}")
    ag._save(ckpt)
    print("✔ saved", ckpt)


def evaluate(ckpt_file: str, num_runs: int = 500):
    eval_agent = SACPerCampaign(ckpt=ckpt_file, inference=True)
    sim        = AdXGameSimulator()
    foes       = [Tier1NDaysNCampaignsAgent(name=f"T1-{i}") for i in range(9)]

    sim.run_simulation([eval_agent] + foes, num_simulations=num_runs)

# ══════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_eps", type=int, default=1000)
    p.add_argument("--ckpt", type=str, default="sac_pc.pth")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_runs", type=int, default=500)
    args = p.parse_args()

    if args.eval_only:
        evaluate(args.ckpt, args.eval_runs)
    else:
        train(args.train_eps, args.ckpt)
        evaluate(args.ckpt, args.eval_runs)
