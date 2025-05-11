from __future__ import annotations

import math, random, argparse, collections
from typing import Dict, Set
from collections import deque

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Normal

# ─────────────  GAME API  ─────────────
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign, MarketSegment
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

import numpy as np
# from .path_utils import path_from_local_root # For submission
from path_utils import path_from_local_root  # For regular use

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
        probs = prios ** self.alpha
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
LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, d_model=64, nhead=4, depth=2, mlp=256, dropout=0.1):
        """
        Args
        ----
        s_dim : number of scalar input features  (= sequence length)
        a_dim : action dimension
        d_model : embedding/channel width
        """
        super().__init__()
        self.s_dim = s_dim

        self.val_embed = nn.Linear(1, d_model)               # value token
        self.pos_embed = nn.Embedding(s_dim + 1, d_model)  # +1 for CLS token

        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        
        self.encoder = nn.TransformerEncoder(enc_layer, depth)

        # "CLS" style pooling: prepend learnable cls token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp, mlp),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(mlp, a_dim)
        self.log_std = nn.Linear(mlp, a_dim)
        
        self._init_weights()


    def _init_weights(self):
        # Initialization following Transformer defaults
        nn.init.normal_(self.cls, std=0.02)
        nn.init.xavier_uniform_(self.val_embed.weight)
        nn.init.zeros_(self.val_embed.bias)
        nn.init.xavier_uniform_(self.pos_embed.weight)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(self, s: torch.Tensor):
        """
        Compute mu and log_std for a batch of states.

        Args:
            s (Tensor): [B, s_dim] input features.

        Returns:
            mu (Tensor): [B, a_dim] action means.
            log_std (Tensor): [B, a_dim] log standard deviations.
        """
        B, S = s.size()
        # Value token embeddings
        x = s.unsqueeze(-1)  # [B, S, 1]
        val_tokens = self.val_embed(x)  # [B, S, d_model]

        # Positional embeddings (first idx 0 reserved for CLS)
        positions = torch.arange(1, S + 1, device=s.device).unsqueeze(0)
        pos_tokens = self.pos_embed(positions)  # [1, S, d_model]

        tok = val_tokens + pos_tokens

        # Prepend CLS token
        cls = self.cls.expand(B, -1, -1)  # [B,1,d_model]
        seq = torch.cat([cls, tok], dim=1)       # [B, S+1, d_model]
        seq = self.dropout(seq)

        # Transformer encoding and CLS pooling
        encoded = self.encoder(seq)             # [B, S+1, d_model]
        h = encoded[:, 0]                       # [B, d_model]

        # MLP head
        z = self.head(h)                        # [B, mlp_dim]
        mu = self.mu(z)
        log_std = self.log_std(z).clamp(LOG_STD_MIN, LOG_STD_MAX)
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
    
class TimeSeriesEncoder(nn.Module):
    def __init__(self, k: int, hidden: int):
        super().__init__()
        # GRU over last k days of [reach, spend]
        self.gru = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
    def forward(self, hist: torch.Tensor):
        # hist: [B, k, 2]
        _, h = self.gru(hist)     # h: [1, B, hidden]
        return h.squeeze(0)       # [B, hidden]

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


# ══════════════ Pop-Art normaliser ══════════════
class PopArt(nn.Module):
    def __init__(self, beta=0.99999):
        super().__init__()
        self.mu  = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.sig = nn.Parameter(torch.ones(1),  requires_grad=False)
        self.beta= beta

    def forward(self, x):
        # x is a tensor shape [1]
        with torch.no_grad():
            d = 1 - self.beta
            new_mu  = self.beta * self.mu + d * x.mean()
            new_var = (
                self.beta * (self.sig**2 + self.mu**2)
              + d * (x**2).mean()
              - new_mu**2
            )
            self.mu.data.copy_(new_mu)
            self.sig.data.copy_(torch.sqrt(new_var + 1e-6))
        return (x - self.mu) / self.sig

# ══════════════════════════════════════
# 3. SAC agent with per-campaign rewards
# ══════════════════════════════════════
MAX_CAMPAIGNS = 10
GAME_LENGTH = 10.0

class SACPerCampaign(NDaysNCampaignsAgent):
    S_DIM, A_DIM = 12, 2

    # memory entry for campaign c chosen yesterday
    MemEntry = collections.namedtuple(
        "MemEntry", ("state", "action", "prev_reach", "prev_cost", "end_day", "uid")
    )

    def __init__(self, ckpt: str | None = None, inference=False, n_critics=3):
        super().__init__()
        self.name = "SAC_PC"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hid = 256
        self.actor = GaussianPolicy(self.S_DIM, self.A_DIM).to(self.device)
        self.qs = nn.ModuleList([QNet(self.S_DIM, self.A_DIM, hid) for _ in range(n_critics)]).to(self.device)
        self.targets = nn.ModuleList([QNet(self.S_DIM, self.A_DIM, hid) for _ in range(n_critics)]).to(self.device)
        for q, t in zip(self.qs, self.targets):
                    t.load_state_dict(q.state_dict())

        a_lr = 1e-4
        q_lr = 4e-4
        self.o_actor = optim.Adam(self.actor.parameters(), a_lr)
        self.q_optimizers = [optim.Adam(q.parameters(), lr=q_lr) for q in self.qs]
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.o_alpha = optim.Adam([self.log_alpha], 5e-5) # We tune α to control exploration. A smaller LR here prevents wild swings in temperature, stabilizing stochasticity in bids.

        # a high γ still gives nonzero weight to “tomorrow,” but slightly lower than 0.995 avoids over-valuing distant outcomes that rarely exist.
        # A smaller τ makes targets update more responsively—important when episodes are short, so we want up-to-date bootstrapping.
        self.gamma, self.tau = 0.98, 0.003
        
        # Performing 8 gradient steps per environment interaction amplifies learning signal from each batch—helpful when data is costly to collect (campaign days only tick forward once per step)
        self.batch, self.update_after, self.updates_ps = 256, 512, 6
        
        # Updating the actor every critic step (delay=1) accelerates policy improvement. In a short-horizon task, you want to quickly translate better Q-estimates into improved bidding behavior.
        self.policy_delay = 1 
        self.target_entropy = -float(self.A_DIM)
        self.popart = PopArt().to(self.device)
        
        self.per_alpha = 0.6

        self.buffer = PrioritizedReplayBuffer(
            capacity=100_000,
            alpha=self.per_alpha,
            device=self.device
        )
        
        self.mem: Dict[int, SACPerCampaign.MemEntry] = {}

        self.total_steps = 0
        self.inference = inference
        
        self._last_Q = self.get_quality_score()
        
        if ckpt and inference:
            self._load(ckpt)
            self.inference = True
        
        # For EMA Normalization
        self.r_beta = 0.999
        self.r_mean = 0.0
        self.r_var  = 1.0
        
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

            # --------- single-attribute segments (6) ---------
            MarketSegment({'Male'}):                       4956,
            MarketSegment({'Female'}):                     5044,
            MarketSegment({'Young'}):                      4589,
            MarketSegment({'Old'}):                        5411,
            MarketSegment({'LowIncome'}):                  8012,
            MarketSegment({'HighIncome'}):                 1988,
        }

        self.TOTAL_DAILY_USERS = 10_000        # given in the hand-out
        
        
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
    

    def _state_c(self, c: Campaign, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio) -> torch.Tensor:
        day_n = self.get_current_day() / GAME_LENGTH
        r_now = self.get_cumulative_reach(c)
        c_now = self.get_cumulative_cost(c)

        # per-campaign
        prog   = r_now / max(1, c.reach)
        rem_b  = (c.budget - c_now) / max(1.0, c.budget)
        days_l = (c.end_day - self.get_current_day() + 1) / GAME_LENGTH
        length = (c.end_day - c.start_day + 1) / 3.0

        baseline = (c.budget - c_now) / max(1, c.reach - r_now)
        cpi      = c_now / max(1, r_now)

        x,a,b = r_now / max(1, c.reach), 4.08577, 3.08577
        slope = (a / (1 + (a*x - b)**2)) / c.reach
        
        urg   = 1.0 / max(1, c.end_day - self.get_current_day() + 1)
        qs = self.get_quality_score()

        dq = self.get_quality_score() - getattr(self, "_last_Q", self.get_quality_score())
        self._last_Q = self.get_quality_score()

        # [day_n, n_norm, avg_p, min_p, max_p, prog * urg, rem_b, qs] => 8
        # [day_n, n_norm, avg_p, avg_b, ends_ratio, prog, rem_b, days_l, baseline, cpi, slope, qs] => 12
        # [day_n, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio, prog, rem_b, days_l, length, baseline, cpi, slope, dq, qs] => 18
        
        scalars = [day_n, n_norm, avg_p, avg_b, ends_ratio, prog, rem_b, days_l, baseline, cpi, slope, qs]

        s = torch.tensor(scalars, dtype=torch.float32, device=self.device)
        return s
        
        # seg_id = torch.tensor(self._seg_id(c.target_segment), device=self.device)
        # seg_vec = self.seg_emb(seg_id)

        # return torch.cat([s, seg_vec], dim=-1)  # → (29,)

    # ── reset episode mem ──
    def on_new_game(self):
        self.mem.clear()

    # ── main bidding function ──
    def get_segment_competitiveness(self, segment):
        """
        Returns a competitiveness score for a market segment.
        Higher values indicate more competitive segments that should receive bonus rewards.
        """
        # Example implementation - you might want to customize this based on your strategy
        segment_size = self.SEGMENT_FREQ.get(segment, 0)
        total_users = self.TOTAL_DAILY_USERS
        
        # Smaller segments might be more valuable/competitive
        if segment_size > 0:
            return 1.0 + (1.0 - segment_size/total_users)
        return 1.0  # Default value for unknown segments



    def get_ad_bids(self) -> Set[BidBundle]:
        bundles: Set[BidBundle] = set()
        
        active  = list(self.get_active_campaigns())
        day     = self.get_current_day()
        n_norm  = len(active) / MAX_CAMPAIGNS

        # 1) progress & budget ratios
        reaches = [self.get_cumulative_reach(c)/c.reach for c in active] or [0.0]
        budgets = [(c.budget-self.get_cumulative_cost(c))/c.budget for c in active] or [0.0]
        avg_p, min_p, max_p = map(float, (np.mean(reaches), np.min(reaches), np.max(reaches)))
        avg_b, min_b, max_b = map(float, (np.mean(budgets), np.min(budgets), np.max(budgets)))

        # 2) endings today
        ends_ratio = sum(1 for c in active if c.end_day == day) / MAX_CAMPAIGNS

        if not self.inference:
            active_map = {c.uid: c for c in active}
            # 1.  compute reward for campaigns we bid on yesterday (this loop is not needed for inference)
            
            for uid, m in list(self.mem.items()):
                # It might have ended and vanished from active list; retrieve via stored uid
                camp = active_map.get(uid)
                
                if camp is None or day > camp.end_day:
                    # use last known state; reward is zero because reach & cost no longer change
                    self.buffer.push(m.state, m.action, 0.0,
                                    torch.zeros_like(m.state), 1.0)
                    self.mem.pop(uid, None)
                    continue

                new_reach = self.get_cumulative_reach(camp)
                new_cost = self.get_cumulative_cost(camp)
                

                #--------------method 1-------------------
                # delta_cost = new_cost - m.prev_cost # change in cost
                # eff_prev = self.effective_reach(m.prev_reach, camp.reach)
                # eff_new = self.effective_reach(new_reach, camp.reach)
                
                # raw_reward = (eff_new - eff_prev) * camp.budget - delta_cost # reward = budget * reach - cost
                
                #--------------method 5 (Optimized Method 1)-------------------
                                # Base reward from method 1 (proven to work well)
                delta_cost = new_cost - m.prev_cost
                eff_prev = self.effective_reach(m.prev_reach, camp.reach)
                eff_new = self.effective_reach(new_reach, camp.reach)
                raw_reward = (eff_new - eff_prev) * camp.budget - delta_cost # reward = budget * reach - cost

                base_reward = (eff_new - eff_prev) * camp.budget - delta_cost
                
                # Add this to the reward calculation if you can identify regional preferences 
                segment_competitiveness = self.get_segment_competitiveness(camp.target_segment) 
                segment_bonus = (eff_new - eff_prev) * camp.budget * segment_competitiveness * 0.1 
                raw_reward += segment_bonus
                
                # Add a smaller quality component - only when it matters
                current_quality = self.get_quality_score() or 0.5
                quality_bonus = 0.0
                
                # Only add quality bonus when quality is below 1.0 and improving
                if hasattr(self, '_prev_quality') and current_quality < 1.0:
                    quality_delta = current_quality - self._prev_quality
                    if quality_delta > 0:
                        # Smaller weight (0.1 instead of 0.2)
                        quality_bonus = quality_delta * camp.budget * 0.1
                self._prev_quality = current_quality
                
                # More targeted urgency factor - only for campaigns at risk of failing
                days_remaining = max(1, camp.end_day - self.get_current_day())
                progress_ratio = eff_new / 1.0  # Normalized progress (effective reach is already normalized)
                urgency_bonus = 0.0
                
                # Only apply urgency to campaigns that are behind schedule
                if days_remaining <= 2 and progress_ratio < 0.8:
                    # Smaller weight (0.15 instead of 0.3)
                    urgency_bonus = (eff_new - eff_prev) * camp.budget * 0.15
                
                # Combine rewards with base reward having highest weight
                raw_reward = base_reward + quality_bonus + urgency_bonus


                
                # Clip extreme rewards to prevent instability
                raw_reward = max(min(raw_reward, 1000), -1000)

            

                # Pop-Art normalization
                reward = self.popart(torch.tensor([raw_reward], dtype=torch.float32, device=self.device))[0]
                reward = reward.item()  # convert to scalar
                
                # NaN / inf guard
                if not math.isfinite(reward):
                    reward = torch.tensor(0.0, device=self.device)

                done = float(self.get_current_day() >= camp.end_day)
                next_state = (torch.zeros_like(m.state) if done else self._state_c(camp, n_norm, avg_p, min_p, max_p, avg_b, min_b, max_b, ends_ratio))
                
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
        for c in active:
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
                    
            action = action.squeeze(0)
            bid_mul, lim_mul = 0.5 + 0.5 * action[0].item(), 0.5 + 0.5 * action[1].item()

            ########## compute bid & limit
            if hasattr(self, 'get_auction_round') and self.get_auction_round() > 1: 
                # More aggressive in later rounds 
                bid_mul *= 1.1
                print("hasattr")
            #########

            rem_reach = max(1, c.reach - cum)
            rem_budget = max(1e-6, c.budget - cost)
            
            base = rem_budget / rem_reach
            base = float(np.clip(base, 0.10, 10.0))       # 10 USD CPM hard-cap
            bid = float(np.clip(base * bid_mul, 0.10, min(10.0, rem_budget)))
            limit = float(np.clip(rem_budget * lim_mul, bid, rem_budget))
            
            # base = rem_budget / rem_reach
            # base = float(min(max(base, 0.1), rem_budget)) # clip to [0.1, rem_budget] ?
            # bid = float(min(max(base * bid_mul, 0.1), rem_budget)) # clip to [0.1, rem_budget]
            # limit = float(min(max(rem_budget * lim_mul, bid), rem_budget)) # clip to [bid, rem_budget] ?

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
    
    # ── SAC update ──
    def _learn(self):
        if self.inference or len(self.buffer) < max(self.batch, self.update_after):
            return
        
        beta = min(1.0, 0.4 + self.total_steps * (1.0 - 0.4) / 100000)  # linear β-annealing

        for _ in range(self.updates_ps):
            s, a, r, s2, d, idxs, weights = self.buffer.sample(self.batch, beta)
        
            # ── target Q with n-step γⁿ ──
            with torch.no_grad():
                a2, lp2 = self.actor.sample(s2)
                noise   = (torch.randn_like(a2) * 0.2).clamp(-0.5, 0.5)
                a2      = (a2 + noise).clamp(-1, 1)
                q_tgts = [t(s2, a2) for t in self.targets]
                # take the element-wise minimum over the ensemble
                min_q_tgt = torch.min(torch.cat(q_tgts, dim=1), dim=1, keepdim=True)[0]
                y       = r + (1 - d) * self.gamma * (min_q_tgt - self.alpha * lp2)

            # ── critic updates (weighted MSE) ──
            td_errors = []
            for i, (q, opt) in enumerate(zip(self.qs, self.q_optimizers)):
                q_pred = q(s, a)
                td    = y - q_pred
                loss  = (weights * td.pow(2)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                td_errors.append(td.abs().mean(dim=1).detach())

                # update its target
                with torch.no_grad():
                    for p, tp in zip(q.parameters(), self.targets[i].parameters()):
                        tp.data.mul_(1-self.tau).add_(self.tau*p.data)
        
            # ---- update PER priorities (use average TD-error) ----
            new_prios = torch.stack(td_errors, dim=1).mean(dim=1).cpu().numpy()
            self.buffer.update_priorities(idxs, new_prios + 1e-6)

            # ── actor + α ──
            if self.total_steps % self.policy_delay == 0:
                an, lp = self.actor.sample(s)
                q_values = [q(s, an) for q in self.qs]
                q_min = torch.min(torch.cat(q_values, dim=1), dim=1, keepdim=True)[0]
                a_loss = (self.alpha * lp - q_min).mean()
                
                self.o_actor.zero_grad(); a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
                self.o_actor.step()

                alpha_loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
                self.o_alpha.zero_grad(); alpha_loss.backward(); self.o_alpha.step()

            self.total_steps += 1
            if self.total_steps % 1000 == 0:
                print(
                    f"[learn] step {self.total_steps}  "
                    f"α={self.alpha.item():.2f}  "
                    f"loss={loss.item():.2f}  "
                    f"actor_loss={a_loss.item():.2f}  "
                    f"alpha_loss={alpha_loss.item():.2f}"
                )
        
    # ---------- rule-based campaign auction ----------
    # def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
    #     bids = {}
    #     day = self.get_current_day()
    #     Q = self.get_quality_score()
    #     for c in campaigns_for_auction:
    #         min_bid = 0.1 * c.reach
    #         max_bid = c.reach
    #         base = min_bid if Q >= 1 else min_bid + (max_bid - min_bid) * (1 - Q)
    #         if day >= 7:
    #             base *= 1.15
    #         bids[c] = self.clip_campaign_bid(c, base)
    #     return bids
    # def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
    #     bids = {}
    #     day = self.get_current_day()
    #     Q = self.get_quality_score()
        
    #     # Safety check for quality score
    #     if Q is None:
    #         Q = 0.5  # Default quality score if None
        
    #     # Track segment popularity to adjust bids
    #     segment_counts = {}
    #     for c in campaigns_for_auction:
    #         if c.target_segment is not None:  # Safety check
    #             segment = c.target_segment
    #             segment_counts[segment] = segment_counts.get(segment, 0) + 1
        
    #     # Find most popular segment (with safety check)
    #     max_count = max(segment_counts.values()) if segment_counts else 1
        
    #     for c in campaigns_for_auction:
    #         # Safety checks for campaign attributes
    #         if c.reach is None or c.budget is None or c.start_day is None or c.end_day is None:
    #             continue  # Skip campaigns with missing attributes
                
    #         # Base bid calculation
    #         min_bid = 0.1 * c.reach
    #         max_bid = c.reach
            
    #         # Quality score adjustment
    #         if Q >= 1:
    #             base = min_bid
    #         else:
    #             base = min_bid + (max_bid - min_bid) * (1 - Q)
            
    #         # Day-based adjustment (bid more aggressively in later days)
    #         if day >= 7:
    #             base *= 1.2
    #         elif day >= 5:
    #             base *= 1.1
            
    #         # Segment popularity adjustment with safety check
    #         segment_popularity = 0
    #         if c.target_segment is not None and max_count > 0:
    #             segment_popularity = segment_counts.get(c.target_segment, 0) / max_count
            
    #         # Bid more for less contested segments (better ROI opportunity)
    #         # and slightly less for highly contested segments
    #         popularity_factor = 1.0 - (segment_popularity * 0.2)  # 0.8-1.0 range
    #         base *= popularity_factor
            
    #         # Campaign value assessment with safety checks
    #         if c.reach > 0:  # Prevent division by zero
    #             value_ratio = c.budget / c.reach  # Higher ratio means more valuable campaign
                
    #             # Calculate average value with safety checks
    #             valid_campaigns = [camp for camp in campaigns_for_auction 
    #                               if camp.reach is not None and camp.budget is not None and camp.reach > 0]
                
    #             if valid_campaigns:
    #                 avg_value = sum(camp.budget / camp.reach for camp in valid_campaigns) / len(valid_campaigns)
                    
    #                 # Adjust bid based on campaign value (bid more for high-value campaigns)
    #                 if value_ratio > avg_value * 1.2:
    #                     base *= 1.15  # Premium for high-value campaigns
    #                 elif value_ratio < avg_value * 0.8:
    #                     base *= 0.9   # Discount for low-value campaigns
            
    #         # Final adjustments based on campaign duration
    #         campaign_duration = c.end_day - c.start_day + 1
    #         if campaign_duration <= 3:  # Short campaigns need quick action
    #             base *= 1.1
            
    #         # Ensure base is a valid number
    #         if base is None or not math.isfinite(base):
    #             base = min_bid  # Fallback to minimum bid
                
    #         # Clip the bid to ensure it's valid
    #         bids[c] = self.clip_campaign_bid(c, base)
            
    #     return bids
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        day = self.get_current_day()
        Q = self.get_quality_score() or 0.5  # Default if None
        
        # Track segment popularity and value metrics
        segment_counts = {}
        segment_values = {}
        total_campaigns = len(campaigns_for_auction)
        
        # First pass: gather statistics
        for c in campaigns_for_auction:
            if c.reach is None or c.budget is None or c.start_day is None or c.end_day is None:
                continue
                
            # Track segment popularity
            if c.target_segment is not None:
                segment = c.target_segment
                segment_counts[segment] = segment_counts.get(segment, 0) + 1
                
                # Calculate value density (budget/reach ratio)
                if c.reach > 0:
                    value = c.budget / c.reach
                    if segment not in segment_values:
                        segment_values[segment] = []
                    segment_values[segment].append(value)
        
        # Calculate segment value metrics
        segment_avg_values = {}
        for segment, values in segment_values.items():
            segment_avg_values[segment] = sum(values) / len(values) if values else 0
            
        # Find most valuable segments (top 30%)
        sorted_segments = sorted(segment_avg_values.items(), key=lambda x: x[1], reverse=True)
        high_value_segments = set()
        if sorted_segments:
            cutoff = max(1, int(len(sorted_segments) * 0.3))
            high_value_segments = {s[0] for s in sorted_segments[:cutoff]}
        
        # Second pass: calculate bids
        for c in campaigns_for_auction:
            if c.reach is None or c.budget is None or c.start_day is None or c.end_day is None:
                continue
                
            # Base bid calculation with progressive scaling
            min_bid = 0.1 * c.reach
            max_bid = c.reach
            
            # Quality score adjustment - more aggressive when quality is lower
            if Q >= 1:
                base = min_bid
            else:
                # More aggressive bidding when quality is lower
                base = min_bid + (max_bid - min_bid) * (1 - Q)**0.8
            
            # Day-based adjustment - more aggressive in later days
            if day >= 8:
                base *= 1.3  # Very aggressive in final days
            elif day >= 6:
                base *= 1.2  # More aggressive in later days
            elif day >= 4:
                base *= 1.1  # Slightly more aggressive in mid-game
            
            # Campaign duration adjustment - bid more for shorter campaigns
            campaign_duration = c.end_day - c.start_day + 1
            if campaign_duration <= 2:  # Very short campaigns
                base *= 1.25
            elif campaign_duration <= 4:  # Short campaigns
                base *= 1.15
            
            # Value-based adjustment
            if c.reach > 0:
                value_ratio = c.budget / c.reach
                
                # Calculate average value across all campaigns
                valid_campaigns = [camp for camp in campaigns_for_auction 
                                  if camp.reach is not None and camp.budget is not None and camp.reach > 0]
                
                if valid_campaigns:
                    avg_value = sum(camp.budget / camp.reach for camp in valid_campaigns) / len(valid_campaigns)
                    
                    # Bid more aggressively for high-value campaigns
                    if value_ratio > avg_value * 1.5:
                        base *= 1.25  # Significantly higher for very valuable campaigns
                    elif value_ratio > avg_value * 1.2:
                        base *= 1.15  # Higher for valuable campaigns
                    elif value_ratio < avg_value * 0.7:
                        base *= 0.85  # Lower for less valuable campaigns
            
            # Segment popularity and value-based adjustments
            if c.target_segment is not None:
                # Adjust for segment popularity
                segment_popularity = segment_counts.get(c.target_segment, 0) / total_campaigns
                
                # Bid more for less contested segments
                popularity_factor = 1.0 - (segment_popularity * 0.3)  # 0.7-1.0 range
                base *= popularity_factor
                
                # Bid more for high-value segments
                if c.target_segment in high_value_segments:
                    base *= 1.15
            
            # Ensure base is a valid number
            if base is None or not math.isfinite(base):
                base = min_bid
                
            # Clip the bid to ensure it's valid
            bids[c] = self.clip_campaign_bid(c, base)
            
        return bids

# ══════════════════════════════════════
# 4. training / evaluation helpers
# ══════════════════════════════════════
def train(eps: int, ckpt: str, ma_window: int = 100):
    ag = SACPerCampaign()
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
        
    ag._save(ckpt)
    print("✔ saved", ckpt)


def evaluate(ckpt_file: str, num_runs: int = 500):
    eval_agent = SACPerCampaign(ckpt=ckpt_file, inference=True)
    sim        = AdXGameSimulator()
    foes       = [Tier1NDaysNCampaignsAgent(name=f"T1-{i}") for i in range(9)]

    sim.run_simulation([eval_agent] + foes, num_simulations=num_runs)
    
def evaluate_v2(ckpt_file: str, num_runs: int = 500):
    eval_agent = SACPerCampaign(ckpt=ckpt_file, inference=True)
    sim        = AdXGameSimulator()
    foes       = [Tier1NDaysNCampaignsAgent(name=f"T1-0")]

    sim.run_simulation([eval_agent] + foes, num_simulations=num_runs)

# my_agent_submission = SACPerCampaign(ckpt='./model_checkpoints/sac_pc_v2_sv_12_sd_popart_sa.pth', inference=True)

# ══════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_eps", type=int, default=1000)
    p.add_argument("--ckpt", type=str, default="sac_pc_v2_exp.pth")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_runs", type=int, default=500)
    args = p.parse_args()

    if args.eval_only:
        evaluate(args.ckpt, args.eval_runs)
        # evaluate_v2(args.ckpt, args.eval_runs)
    else:
        train(args.train_eps, args.ckpt)
        evaluate(args.ckpt, args.eval_runs)
