from __future__ import annotations

import math, random, argparse, collections
from typing import Dict, Set, Tuple
from collections import deque

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Normal

# ─────────────  GAME API  ─────────────
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign, MarketSegment
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

import numpy as np
from .path_utils import path_from_local_root # For submission
# from path_utils import path_from_local_root  # For regular use

# ══════════════════════════════════════
# 1. replay buffer
# ══════════════════════════════════════

Transition  = collections.namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, device=None):
        self.capacity   = capacity
        self.alpha      = alpha
        self.pos        = 0
        self.size       = 0
        self.data       = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_prio   = 1.0
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _add(self, transition):
        """Add a single (possibly n-step) transition to the ring buffer."""
        self.data[self.pos] = transition
        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a single-step transition."""
        self._add(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions with importance-sampling weights."""
        assert self.size > 0, "Buffer is empty!"
        
        prios = self.priorities if self.size == self.capacity else self.priorities[:self.pos]
        probs = (prios + 1e-5) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(probs), batch_size, p=probs)
        samples = [self.data[idx] for idx in indices]
        batch = Transition(*zip(*samples))

        states      = torch.stack(batch.state).to(self.device)
        actions     = torch.stack(batch.action).to(self.device)
        rewards     = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones       = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        total = len(probs)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, new_prios):
        """After learning, update the priorities of sampled transitions."""
        for idx, prio in zip(indices, new_prios):
            self.priorities[idx] = prio
            self.max_prio = max(self.max_prio, prio)
            
    def __len__(self):
        return self.size

# ══════════════════════════════════════
# 2. networks
# ══════════════════════════════════════
LOG_STD_MIN, LOG_STD_MAX = -5, 1
class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)

    def forward(self, s):
        x = self.net(s)
        return self.mu(x), self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
    
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
    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+a_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, s, a):
        return self.net(torch.cat([s,a], -1))


# ══════════════════════════════════════
# 3. SAC agent with per-campaign rewards
# ══════════════════════════════════════
GAME_LENGTH = 10.0

class SACPIAgent(NDaysNCampaignsAgent):
    S_DIM, A_DIM = 12, 2

    # memory entry for campaign c chosen yesterday
    MemEntry = collections.namedtuple(
        "MemEntry", ("state", "action", "prev_reach", "prev_cost", "end_day", "uid")
    )

    def __init__(self, ckpt: str | None = None, inference=False):
        super().__init__()
        self.name = "LeAgent"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hid = 256
        self.actor = GaussianPolicy(self.S_DIM, self.A_DIM).to(self.device)
        self.q1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.q2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1.load_state_dict(self.q1.state_dict())
        self.t2.load_state_dict(self.q2.state_dict())

        a_lr = 1e-4
        q_lr = 3e-4
        self.o_actor = optim.Adam(self.actor.parameters(), a_lr)
        self.o_q1 = optim.Adam(self.q1.parameters(), q_lr)
        self.o_q2 = optim.Adam(self.q2.parameters(), q_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.o_alpha = optim.Adam([self.log_alpha], 1e-4)

        self.gamma, self.tau = 0.95, 0.005
        self.batch, self.update_after, self.updates_ps = 128, 256, 2

        self.target_entropy = -float(self.A_DIM)
        
        self.buffer = PrioritizedReplayBuffer(
            capacity=100_000, alpha=0.6, device=self.device
        )
        self.mem: Dict[int, SACPIAgent.MemEntry] = {}

        self.total_steps = 0
        self.inference = inference
        
        self._last_Q = self.get_quality_score()
        
        if ckpt and inference:
            self._load(ckpt)
            self.inference = True

        # For Campaign Segmentation
        self.SEGMENT_FREQ = {
            # --------- triple-attribute segments (8) ---------
            MarketSegment({'Male','Young','LowIncome'}):    1836,
            MarketSegment({'Male','Young','HighIncome'}):    517,
            MarketSegment({'Male','Old','LowIncome'}):      1795,
            MarketSegment({'Male','Old','HighIncome'}):      808,
            MarketSegment({'Female','Young','LowIncome'}):  1980,
            MarketSegment({'Female','Young','HighIncome'}):  256,
            MarketSegment({'Female','Old','LowIncome'}):    2401,
            MarketSegment({'Female','Old','HighIncome'}):    407,

            # --------- double-attribute segments (12) --------
            MarketSegment({'Male','Young'}):               2353,
            MarketSegment({'Male','Old'}):                 2603,
            MarketSegment({'Female','Young'}):             2236,
            MarketSegment({'Female','Old'}):               2808,
            MarketSegment({'Young','LowIncome'}):          3816,
            MarketSegment({'Old','LowIncome'}):            4196,
            MarketSegment({'Young','HighIncome'}):          773,
            MarketSegment({'Old','HighIncome'}):           1215,
            MarketSegment({'Male','LowIncome'}):           3631,
            MarketSegment({'Female','LowIncome'}):         4381,
            MarketSegment({'Male','HighIncome'}):          1325,
            MarketSegment({'Female','HighIncome'}):         663,
        }

        self.TOTAL_DAILY_USERS = 10_000        # given in the hand-out
        
        # max simultaneous active campaigns
        self.max_running = 10
        self.baseline_cpi = 0.2

        # EMA smoothing factor for CPI
        self.ema_alpha = 0.2
        # Per-segment current CPI estimate (initialized to baseline_cpi)
        self.segment_cpi: Dict[MarketSegment, float] = {
            seg: self.baseline_cpi for seg in self.SEGMENT_FREQ
        }
        # Previous-day totals for reach and cost per segment
        self._seg_prev_totals: Dict[MarketSegment, Tuple[float, float]] = {
            seg: (0.0, 0.0) for seg in self.SEGMENT_FREQ
        }
        
        self._all_campaigns: Dict[int, Campaign] = {}
        
        self.global_reward_ema   = 0.0
        self.reward_ema_beta     = 0.8    # tune between 0.8 and 0.95
        
    # ── α helper ──
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    # ── checkpoint ──
    def _save(self, f):
        torch.save(dict(actor=self.actor.state_dict()), f)

    def _load(self, f):
        # For submission
        ckpt_path = path_from_local_root(f)
        print("Loading checkpoint from", ckpt_path)
        ck = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        
    def _update_segment_cpi_from_deltas(self):
        new_totals = {}

        for seg in self.SEGMENT_FREQ:
            # total impressions & cost delivered so far in this segment
            tot_reach = 0.0
            tot_cost = 0.0

            for c in self._all_campaigns.values():
                if c.target_segment != seg:
                    continue
                # skip assigned deals if you want only auctioned
                if c.reach == c.budget:
                    continue
                
                tot_reach += self.get_cumulative_reach(c)
                tot_cost  += self.get_cumulative_cost(c)
                    
            prev_reach, prev_cost = self._seg_prev_totals[seg]
            dr = tot_reach - prev_reach
            dc = tot_cost  - prev_cost

            if dr > 1e-6 and dc > 0.0:
                observed_price = dc / dr
                # clamp to avoid outliers
                observed_price = float(np.clip(observed_price, 0.001, 10.0))
                old_cpi = self.segment_cpi[seg]
                self.segment_cpi[seg] = (
                    (1 - self.ema_alpha) * old_cpi
                    + self.ema_alpha * observed_price
                )
                
            new_totals[seg] = (tot_reach, tot_cost)

        self._seg_prev_totals = new_totals

    def get_baseline_cpi(self, segment: MarketSegment) -> float:
        """Returns the current EMA-based CPI for the given segment."""
        return self.segment_cpi.get(segment, 0.0)
    
        
    def _state_c(self,c : Campaign, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio) -> torch.Tensor:
        """
        Per-campaign + global state for SAC:
        — Global time & market context
        — Campaign urgency & capacity
        — Bid anchors (baseline vs. realized CPI)
        — Creative quality
        """
        D       = GAME_LENGTH
        today   = self.get_current_day()

        # 1) Global context (shared across campaigns)
        day_frac    = today / D                 # how far into the 10-day game
        market_imp  = n_norm                    # normalized impressions so far
        market_cpi  = avg_p                     # avg clearing price today
        market_bid  = avg_b                     # avg bid price today

        # 2) Campaign delivery & time
        delivered   = self.get_cumulative_reach(c)
        spent       = self.get_cumulative_cost(c)
        prog_frac   = delivered / c.reach       # % of reach already delivered
        rem_reach   = c.reach - delivered
        rem_budget  = c.budget - spent
        rem_days    = max(1, c.end_day - today + 1)
        days_frac   = rem_days / D              # % of game time left

        # 3) Market supply & pacing
        supply      = self.SEGMENT_FREQ[c.target_segment]
        pace_ratio  = rem_reach / (supply * rem_days)
        pace_mul    = 0.6 + 0.8 * np.tanh(2 * pace_ratio)

        # 4) Bid anchors
        true_vpi     = rem_budget / max(1.0, rem_reach)      # actual $ value/impression left => [0, 1]
        realized_cpi = spent / max(1.0, delivered)           # what we’ve actually paid so far

        # 5) Creative quality
        Q            = self.get_quality_score()              # [0,1] higher = better ad

        features = [
            # ── global / market ─────────────────────────
            day_frac,
            market_imp,
            market_cpi,
            market_bid,
            ends_ratio,

            # ── campaign timing & pacing ────────────────
            prog_frac,
            rem_budget  / c.budget,    # remaining budget ratio
            days_frac,
            pace_mul,

            # ── bid anchors ─────────────────────────────
            true_vpi,
            realized_cpi,

            # ── quality ────────────────────────────────
            Q,
        ]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    # ── reset episode mem ──
    def on_new_game(self):
        self.mem.clear()
        for seg in self.SEGMENT_FREQ:
            self._seg_prev_totals[seg] = (0.0, 0.0)
        self._all_campaigns.clear()

    # ── main bidding function ──
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles: Set[BidBundle] = set()
        
        active  = list(self.get_active_campaigns())
        day     = self.get_current_day()
        n_norm  = len(active) / self.max_running
        Q = self.get_quality_score()

        # 1) progress & budget ratios
        reaches = [self.get_cumulative_reach(c)/c.reach for c in active] or [0.0]
        budgets = [(c.budget-self.get_cumulative_cost(c))/c.budget for c in active] or [0.0]
        avg_p, min_p, max_p = map(float, (np.mean(reaches), np.min(reaches), np.max(reaches)))
        avg_b, min_b, max_b = map(float, (np.mean(budgets), np.min(budgets), np.max(budgets)))

        # 2) endings today
        ends_ratio = sum(1 for c in active if c.end_day == day) / self.max_running

        if not self.inference:
            active_map = {c.uid: c for c in active}
            # 1.  compute reward for campaigns we bid on yesterday (this loop is not needed for inference)
            
            for uid, m in list(self.mem.items()):
                # It might have ended and vanished from active list; retrieve via stored uid
                camp = active_map.get(uid)
                
                if camp is None or day > camp.end_day:
                    # use last known state; reward is zero because reach & cost no longer change
                    self.buffer.push(m.state.detach(), m.action.detach(), 0.0, torch.zeros_like(m.state), 1.0)
                    self.mem.pop(uid, None)
                    continue

                new_reach = self.get_cumulative_reach(camp)
                new_cost = self.get_cumulative_cost(camp)

                delta_reach = (new_reach - m.prev_reach) / max(1.0, camp.reach)
                delta_cost = ((new_cost - m.prev_cost) / max(1.0, camp.budget))

                raw_reward = 4 * delta_reach - 2 * delta_cost + 1 * (Q - 0.5)

                # Update the global EMA
                β = self.reward_ema_beta
                self.global_reward_ema = β*self.global_reward_ema + (1-β)*raw_reward

                # Compute a centered (advantage‐like) reward
                reward = raw_reward - self.global_reward_ema

                done = float(self.get_current_day() >= camp.end_day)
                next_state = (torch.zeros_like(m.state) if done else self._state_c(camp, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio))
                
                self.buffer.push(m.state.detach(), m.action.detach(), reward, next_state.detach(), done)

                # remove entry if campaign ended
                if done: 
                    self.mem.pop(uid, None)
                else:
                    # update prev snapshot for tomorrow
                    self.mem[uid] = m._replace(prev_reach=new_reach, prev_cost=new_cost)

        # 2. choose action for each currently active campaign
        for c in active:
            self._all_campaigns[c.uid] = c
            
            cum, cost = self.get_cumulative_reach(c), self.get_cumulative_cost(c)
            
            if cum >= c.reach or cost >= c.budget:  # no need to bid
                self.mem.pop(c.uid, None)
                continue

            s_c = self._state_c(c, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio).unsqueeze(0)
            
            with torch.no_grad():
                if self.inference:
                    mu, _ = self.actor(s_c)
                    action = torch.tanh(mu)
                else:
                    action, _ = self.actor.sample(s_c)
            
            rem_reach = max(1, c.reach - cum)
            rem_budget = max(1e-6, c.budget - cost)
            
            baseline = rem_budget / max(1, rem_reach)
            baseline_spend_limit = rem_budget
            
            action = action.squeeze(0)

            bid_mul = 0.6 + 0.5 * (action[0].item())  # action[0] ∈ [-1,1] → bid_mul ∈ [0.1, 1.1]
            lim_mul = 0.6 + 0.4 * (action[1].item())  # action[1] ∈ [-1,1] → lim_mul ∈ [0.2, 1.0]
            
            bid = float(np.clip(baseline * bid_mul, 0.1, c.reach))
            limit = max(bid, min(rem_budget, baseline_spend_limit * lim_mul))

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

        # learn from replay buffer
        self._learn()
        return bundles

    # ---------- rule-based campaign auction ----------
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        self._update_segment_cpi_from_deltas()
        
        # for seg in self.SEGMENT_FREQ:
        #     print(f"Segment {seg}: Current CPI: {self.segment_cpi[seg]:.2f}")
        
        Q = self.get_quality_score()
        
        slots = self.max_running - len(self.get_active_campaigns())
        if slots <= 0:
            return {}

        ranked = []

        for c in campaigns_for_auction:
            # ---------- basic constants -----------------------------------
            D         = c.end_day - c.start_day + 1
            seg       = c.target_segment
            
            # ---------- difficulty: share of daily inventory needed -------
            supply    = self.SEGMENT_FREQ[seg]            # avg imp / day
            frac      = c.reach / max(1, supply * D)            # 0 … >1
            
            if frac > 1.0:
                continue                                 # infeasible campaign
            
            base_cpi = self.get_baseline_cpi(seg)
            V = c.reach * base_cpi         # Ties “willingness‑to‑pay” to a constant CPI (self.baseline_cpi)
            
            diff_mul  = 0.6 + 0.6 * frac     # pay up to 20 % more for hard jobs

            if np.random.rand() < 0.10:
                diff_mul *= np.random.uniform(0.7, 0.9)  # learn cheap wins
            
            # ---------- one‑shot bid (bounded by budget) ------------------
            raw_bid = V * diff_mul / max(Q, 1e-6)
            bid     = self.clip_campaign_bid(c, raw_bid)
            
            # print(f"Campaign {c.uid}: "
            #         f"Segment CPI: {base_cpi:.2f}, "
            #         f"Difficulty: {diff_mul:.2f}, "
            #         f"Reach: {c.reach:.2f}, "
            #         f"Raw Bid: {raw_bid:.2f}, "
            #         f"Bid: {bid:.2f}, "
            #         f"factor: {base_cpi * diff_mul / max(Q, 1e-6):.2f}, "
            #         f"Effective CPI: {bid / c.reach:.2f}")
            
            expected_reach = c.reach / (base_cpi + 1e-6)
            roi = expected_reach / bid
            ranked.append((roi, c, bid))
            
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = ranked[:slots]
        bids = {c: bid for _, c, bid in selected}
        
        return bids
    
    # ── SAC update ──
    def _learn(self):
        if self.inference or len(self.buffer) < max(self.batch, self.update_after):
            return

        for _ in range(self.updates_ps):
            beta       = 0.4 + min(1.0, self.total_steps / 100_000) * (1.0 - 0.4)
            
            s, a, r, s2, d, idxs, is_weights = self.buffer.sample(self.batch, beta=beta)

            # target policy smoothing
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                noise = torch.randn_like(a2) * 0.1
                noise = torch.clamp(noise, -0.2, 0.2)
                a2 = (a2 + noise).clamp(-1, 1)
                
                # target policy entropy
                target_q = torch.min(self.t1(s2, a2), self.t2(s2, a2))
                y = r + self.gamma * (1 - d) * (target_q - self.alpha * lp2)

            # update critics
            td_errors = []
            for q, opt, targ in ((self.q1,self.o_q1,self.t1),(self.q2,self.o_q2,self.t2)):
                q_pred = q(s,a)
                # q_pred = self.popart(q_pred)
                se     = (q_pred - y).pow(2)
                loss   = (is_weights * se).mean()
                
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(),1.)
                opt.step()
                
                with torch.no_grad():
                    td_errors.append((q_pred - y).abs().squeeze().cpu())
                    for p, tp in zip(q.parameters(), targ.parameters()):
                        tp.data.mul_(1-self.tau).add_(self.tau*p.data)
                # print(f"Critic loss: {loss.item():.4f}")

            new_prios = torch.max(torch.stack(td_errors, dim=1), dim=1)[0]
            self.buffer.update_priorities(idxs, new_prios.numpy())

            an, lp = self.actor.sample(s)
            q_min  = torch.min(self.q1(s,an), self.q2(s,an))
            a_loss = (self.alpha*lp - q_min).mean()
            self.o_actor.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),5.)
            self.o_actor.step()

            alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
            self.o_alpha.zero_grad()
            alpha_loss.backward()
            self.o_alpha.step()
            
            if self.total_steps % 200 == 0:
                print(f"Step {self.total_steps}:")
                print(f"Avg Reward: {r.mean().item():.4f}")
                print(f"Actor loss: {a_loss.item():.4f}  Alpha loss: {alpha_loss.item():.4f}")
                print(f"Critic loss: {loss.item():.4f}  Target Q: {target_q.mean().item():.4f}")
                print(f"Alpha: {self.alpha.item():.4f}  Target entropy: {self.target_entropy:.4f}")
                print(f"Log alpha: {self.log_alpha.item():.4f}  Log std: {self.actor.log_std.weight.data.mean().item():.4f}")
                for seg in self.SEGMENT_FREQ:
                    print(f"Segment {seg}: Current CPI: {self.segment_cpi[seg]:.2f}")
                
            self.total_steps += 1

class BaselineAgent(NDaysNCampaignsAgent):

    def __init__(self, name = "BaselineAgent"):
        super().__init__()
        self.name = name
        # For Campaign Segmentation
        self.SEGMENT_FREQ = {
            # --------- triple-attribute segments (8) ---------
            MarketSegment({'Young','LowIncome', 'Male'}):    1836,
            MarketSegment({'Young','HighIncome', 'Male'}):    517,
            MarketSegment({'Old','LowIncome', 'Male'}):      1795,
            MarketSegment({'Old','HighIncome', 'Male'}):      808,
            MarketSegment({'Young','LowIncome', 'Female'}):  1980,
            MarketSegment({'Young','HighIncome', 'Female'}):  256,
            MarketSegment({'Old','LowIncome', 'Female'}):    2401,
            MarketSegment({'Old','HighIncome', 'Female'}):    407,

            # --------- double-attribute segments (12) --------
            MarketSegment({'Young', 'Male'}):               2353,
            MarketSegment({'Old','Male'}):                 2603,
            MarketSegment({'Young', 'Female'}):             2236,
            MarketSegment({'Old', 'Female'}):               2808,
            MarketSegment({'Young','LowIncome'}):          3816,
            MarketSegment({'Old','LowIncome'}):            4196,
            MarketSegment({'Young','HighIncome'}):          773,
            MarketSegment({'Old','HighIncome'}):           1215,
            MarketSegment({'Male','LowIncome'}):           3631,
            MarketSegment({'Female','LowIncome'}):         4381,
            MarketSegment({'Male','HighIncome'}):          1325,
            MarketSegment({'Female','HighIncome'}):         663,
        }

        # max simultaneous active campaigns
        self.max_running = 10
        self.baseline_cpi = 0.2

        # EMA smoothing factor for CPI
        self.ema_alpha = 0.2
        # Per-segment current CPI estimate (initialized to baseline_cpi)
        self.segment_cpi: Dict[MarketSegment, float] = {
            seg: self.baseline_cpi for seg in self.SEGMENT_FREQ
        }
        # Previous-day totals for reach and cost per segment
        self._seg_prev_totals: Dict[MarketSegment, Tuple[float, float]] = {
            seg: (0.0, 0.0) for seg in self.SEGMENT_FREQ
        }
        
        self._all_campaigns: Dict[int, Campaign] = {}

    def on_new_game(self) -> None:
        for seg in self.SEGMENT_FREQ:
            self._seg_prev_totals[seg] = (0.0, 0.0)
        self._all_campaigns.clear()
        
    def _update_segment_cpi_from_deltas(self):
        new_totals = {}

        for seg in self.SEGMENT_FREQ:
            # total impressions & cost delivered so far in this segment
            tot_reach = 0.0
            tot_cost = 0.0

            for c in self._all_campaigns.values():
                if c.target_segment != seg:
                    continue
                # skip assigned deals if you want only auctioned
                if c.reach == c.budget:
                    continue
                
                tot_reach += self.get_cumulative_reach(c)
                tot_cost  += self.get_cumulative_cost(c)
                    
            prev_reach, prev_cost = self._seg_prev_totals[seg]
            dr = tot_reach - prev_reach
            dc = tot_cost  - prev_cost

            if dr > 1e-6 and dc > 0.0:
                observed_price = dc / dr
                # clamp to avoid outliers
                observed_price = float(np.clip(observed_price, 0.001, 10.0))
                old_cpi = self.segment_cpi[seg]
                self.segment_cpi[seg] = (
                    (1 - self.ema_alpha) * old_cpi
                    + self.ema_alpha * observed_price
                )
                
            new_totals[seg] = (tot_reach, tot_cost)

        self._seg_prev_totals = new_totals

    def get_baseline_cpi(self, segment: MarketSegment) -> float:
        """Returns the current EMA-based CPI for the given segment."""
        return self.segment_cpi.get(segment, 0.0)

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        # Retrieve the active campaigns for which this agent is eligible to bid.
        # print(f"Day {self.get_current_day()}: {len(self.get_active_campaigns())} active campaigns")
        
        active_campaigns = self.get_active_campaigns()
        
        if not active_campaigns:
            return bundles

        # For each active campaign, compute a randomized bid.
        for camp in active_campaigns:
            self._all_campaigns[camp.uid] = camp
            
            cumulative_reach = self.get_cumulative_reach(camp)
            cumulative_cost = self.get_cumulative_cost(camp)
            
            rem_reach = camp.reach - cumulative_reach
            rem_budget = camp.budget - cumulative_cost
            seg = camp.target_segment
            
            baseline = rem_budget / max(1, rem_reach)
            random_bid_factor = np.random.uniform(0.1, 1.1)
            bid_cpi = np.clip(baseline * random_bid_factor, 0.1, camp.reach)
            
            baseline_spend_limit = rem_budget
            random_spend_factor = np.random.uniform(0.2, 1.0)
            spend_limit = max(bid_cpi, min(rem_budget, baseline_spend_limit * random_spend_factor))

            bid = Bid(
                bidder       = self,
                auction_item = seg,
                bid_per_item = bid_cpi,
                bid_limit    = spend_limit
            )
            
            bundles.add(BidBundle(camp.uid, spend_limit, {bid}))         
        
        return bundles

    # ---------- rule-based campaign auction ----------
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        self._update_segment_cpi_from_deltas()
        
        # for seg in self.SEGMENT_FREQ:
        #     print(f"Segment {seg}: Current CPI: {self.segment_cpi[seg]:.2f}")
        
        Q = self.get_quality_score()
        
        slots = self.max_running - len(self.get_active_campaigns())
        if slots <= 0:
            return {}

        ranked = []

        for c in campaigns_for_auction:
            # ---------- basic constants -----------------------------------
            D         = c.end_day - c.start_day + 1
            seg       = c.target_segment
            
            # ---------- difficulty: share of daily inventory needed -------
            supply    = self.SEGMENT_FREQ[seg]            # avg imp / day
            frac      = c.reach / max(1, supply * D)            # 0 … >1
            
            if frac > 1.0:
                continue                                 # infeasible campaign
            
            base_cpi = self.get_baseline_cpi(seg)
            V = c.reach * base_cpi
            
            diff_mul  = 0.6 + 0.6 * frac

            if np.random.rand() < 0.10:
                diff_mul *= np.random.uniform(0.7, 0.9)  # learn cheap wins
            
            # ── 2. quality multiplier 0.20 ↔ 1.00  (good ↓, bad ↑) ─────────
            qual_mul = 0.20 + 0.80 * (1 - Q)         # Q∈[0,1] → qual_mul∈[0.20,1.00]
            
            # ---------- one‑shot bid (bounded by budget) ------------------
            raw_bid = V * diff_mul * qual_mul
            bid     = self.clip_campaign_bid(c, raw_bid)
            
            # print(f"Campaign {c.uid}: "
            #         f"Segment CPI: {base_cpi:.2f}, "
            #         f"Difficulty: {diff_mul:.2f}, "
            #         f"Quality: {qual_mul:.2f}, "
            #         f"Reach: {c.reach:.2f}, "
            #         f"Raw Bid: {raw_bid:.2f}, "
            #         f"Bid: {bid:.2f}, "
            #         f"factor: {base_cpi * diff_mul * qual_mul:.2f}, "
            #         f"Effective CPI: {bid / c.reach:.2f}")
            
            expected_reach = c.reach / (base_cpi + 1e-6)
            roi = expected_reach / bid
            ranked.append((roi, c, bid))
            
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = ranked[:slots]
        bids = {c: bid for _, c, bid in selected}
        
        return bids

my_agent_submission = SACPIAgent(ckpt="./model_checkpoints/sac_pc_v2_pre.pth", inference=True)