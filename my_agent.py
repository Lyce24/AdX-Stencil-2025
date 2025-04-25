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
from .path_utils import path_from_local_root


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
    def __init__(self,s_dim, hidden, a_dim, embed_dim = 32, num_heads=4):
        super().__init__()
        # 1) project from s_dim → embed_dim
        self.to_embed = nn.Linear(s_dim, embed_dim)
        # a little self‐attn over the scalar vector (treated as sequence length=1)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.net  = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, a_dim)
        self.log_std = nn.Linear(hidden, a_dim)

    def forward(self, s):
        # s: [B, s_dim]
        x = self.to_embed(s)            # [B, embed_dim]
        x = x.unsqueeze(1)              # [B,1,embed_dim]
        x, _ = self.attn(x, x, x)       # [B,1,embed_dim]
        x = x.squeeze(1)                # [B,embed_dim]
        x = self.net(x)
        mu      = self.mu(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, s):
        mu, log_std = self(s)
        std = log_std.exp()
        eps = torch.randn_like(std)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)
        logp = Normal(mu, std).log_prob(pre_tanh) \
               - torch.log1p(-a.pow(2)+1e-6)
        return a, logp.sum(-1, keepdim=True)

# class GaussianPolicy(nn.Module):
#     def __init__(self, s_dim: int, hidden: int = 128, a_dim: int = 2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(s_dim, hidden),
#             nn.ReLU(),
#             nn.LayerNorm(hidden),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#         )
#         self.mu = nn.Linear(hidden, a_dim)
#         self.log_std = nn.Linear(hidden, a_dim)

#     def forward(self, s):
#         z = self.net(s)
#         mu = self.mu(z)
#         log_std = torch.clamp(self.log_std(z), LOG_STD_MIN, LOG_STD_MAX)
#         return mu, log_std

#     def sample(self, s):
#         mu, log_std = self(s)
#         std = log_std.exp()
#         eps = torch.randn_like(std)
#         pre_tanh = mu + eps * std
#         a = torch.tanh(pre_tanh)
#         logp = (
#             Normal(mu, std).log_prob(pre_tanh) - torch.log1p(-a.pow(2) + 1e-6)
#         ).sum(-1, keepdim=True)
#         return a, logp


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

    def __init__(self, ckpt: str | None = None, inference=False):
        super().__init__()
        self.name = "SAC_PC"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hid = 256
        self.actor = GaussianPolicy(self.S_DIM, hid, self.A_DIM).to(self.device)
        self.q1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.q2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t2 = QNet(self.S_DIM, self.A_DIM, hid).to(self.device)
        self.t1.load_state_dict(self.q1.state_dict())
        self.t2.load_state_dict(self.q2.state_dict())

        a_lr = 2e-4
        q_lr = 3e-4
        self.o_actor = optim.Adam(self.actor.parameters(), a_lr)
        self.o_q1 = optim.Adam(self.q1.parameters(), q_lr)
        self.o_q2 = optim.Adam(self.q2.parameters(), q_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.o_alpha = optim.Adam([self.log_alpha], 1e-4)

        self.gamma, self.tau = 0.995, 0.003
        self.batch, self.update_after, self.updates_ps = 256, 512, 6
        self.policy_delay = 2
        self.target_entropy = -float(self.A_DIM)
        self.popart = PopArt().to(self.device)
        
        self.buffer = ReplayBuffer(100_000)
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
        torch.save(dict(actor=self.actor.state_dict(),
                        q1=self.q1.state_dict(),
                        q2=self.q2.state_dict(),
                        log_alpha=self.log_alpha.detach()), f)

    def _load(self, f):
        # For submission
        ckpt_path = path_from_local_root(f)
        ck = torch.load(ckpt_path, map_location=self.device)

        # ck = torch.load(f, map_location=self.device)
            
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
        self.t1.load_state_dict(ck["q1"])
        self.t2.load_state_dict(ck["q2"])
        self.log_alpha.data.copy_(ck["log_alpha"])
        
        
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
                
                delta_cost = new_cost - m.prev_cost # change in cost
                eff_prev = self.effective_reach(m.prev_reach, camp.reach)
                eff_new = self.effective_reach(new_reach, camp.reach)
                
                raw_reward = (eff_new - eff_prev) * camp.budget - delta_cost # reward = budget * reach - cost
                
                # Pop-Art normalization
                reward = self.popart(torch.tensor([raw_reward], dtype=torch.float32, device=self.device))[0]
                reward = reward.item()  # convert to scalar
                
                # NaN / inf guard
                if not math.isfinite(reward):
                    reward = torch.tensor(0.0, device=self.device)
                    
                # # EMA normalization
                # reward = (raw_reward - self.r_mean) / math.sqrt(self.r_var + 1e-6)
                # delta      = raw_reward - self.r_mean
                # self.r_mean += (1 - self.r_beta) * delta
                # self.r_var   = self.r_beta * self.r_var + (1 - self.r_beta) * (delta * delta)
                
                # # # Raw reward
                # reward = raw_reward

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

            rem_reach = max(1, c.reach - cum)
            rem_budget = max(1e-6, c.budget - cost)
            
            base = rem_budget / rem_reach
            base = float(min(max(base, 0.1), rem_budget)) # clip to [0.1, rem_budget]
            
            bid = float(min(max(base * bid_mul, 0.1), rem_budget)) # clip to [0.1, rem_budget]
            limit = float(min(max(rem_budget * lim_mul, bid), rem_budget)) # clip to [bid, rem_budget]

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
    def _learn(self, training_method = "smooth"):
        if self.inference or len(self.buffer) < max(self.batch, self.update_after):
            return
        
        if training_method == "smooth":
            for _ in range(self.updates_ps):
                s, a, r, s2, d = self.buffer.sample(self.batch)

                # target policy smoothing
                with torch.no_grad():
                    a2, lp2 = self.actor.sample(s2)
                    noise   = torch.randn_like(a2)*0.2
                    a2      = (a2 + noise.clamp(-0.5,0.5)).clamp(-1,1)
                    target_q= torch.min(self.t1(s2,a2), self.t2(s2,a2))
                    y = r + self.gamma*(1-d)*(target_q - self.alpha*lp2)

                # update critics
                for q, opt, targ in ((self.q1,self.o_q1,self.t1),(self.q2,self.o_q2,self.t2)):
                    loss = F.mse_loss(q(s,a), y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(),5)
                    opt.step()
                    with torch.no_grad():
                        for p, tp in zip(q.parameters(), targ.parameters()):
                            tp.data.mul_(1-self.tau).add_(self.tau*p.data)
                    # print(f"Critic loss: {loss.item():.4f}")

                # delayed actor & alpha
                if self.total_steps % self.policy_delay == 0:
                    an, lp = self.actor.sample(s)
                    q_min  = torch.min(self.q1(s,an), self.q2(s,an))
                    a_loss = (self.alpha*lp - q_min).mean()
                    self.o_actor.zero_grad()
                    a_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),5)
                    self.o_actor.step()

                    alpha_loss = -(self.log_alpha*(lp + self.target_entropy).detach()).mean()
                    self.o_alpha.zero_grad()
                    alpha_loss.backward()
                    self.o_alpha.step()
                    
                    # print(f"Actor loss: {a_loss.item():.4f}")
                    # print(f"Alpha loss: {alpha_loss.item():.4f}")
                    
                
        else:
            for _ in range(self.updates_ps):
                s, a, r, s2, d = self.buffer.sample(self.batch)

                with torch.no_grad():
                    a2, lp2 = self.actor.sample(s2)
                    y = r + self.gamma * (1 - d) * (torch.min(self.t1(s2, a2), self.t2(s2, a2)) - self.alpha * lp2)

                # critics
                for qnet, opt, target in ((self.q1, self.o_q1, self.t1), (self.q2, self.o_q2, self.t2)):
                    loss = F.mse_loss(qnet(s, a), y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(qnet.parameters(), 5)
                    opt.step()
                    with torch.no_grad():
                        for p, tp in zip(qnet.parameters(), target.parameters()):
                            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

                # actor
                an, lpn = self.actor.sample(s)
                actor_loss = (self.alpha * lpn - torch.min(self.q1(s, an), self.q2(s, an))).mean()
                self.o_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
                self.o_actor.step()

                # α
                alpha_loss = -(self.log_alpha * (lpn + self.target_entropy).detach()).mean()
                self.o_alpha.zero_grad()
                alpha_loss.backward()
                self.o_alpha.step()
                
        self.total_steps += 1
        
        
    # ---------- rule-based campaign auction ----------
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        day = self.get_current_day()
        Q = self.get_quality_score()
        for c in campaigns_for_auction:
            min_bid = 0.1 * c.reach
            max_bid = c.reach
            base = min_bid if Q >= 1 else min_bid + (max_bid - min_bid) * (1 - Q)
            if day >= 7:
                base *= 1.15
            bids[c] = self.clip_campaign_bid(c, base)
        return bids
    
    # def get_campaign_bids(self,campaigns_for_auction: Set[Campaign]):
    #     """
    #     Optimised rule-based campaign bidding.

    #     Bid = min_bid + ω · (max_bid - min_bid)
    #     with ω blended from scarcity, urgency, and (1-quality).

    #     Returns a dict {campaign : bid}.
    #     """
    #     bids = {}
    #     today    = self.get_current_day()
    #     Q      = max(0.2, self.get_quality_score())  # clamp Q to stabilize
        
    #     for c in campaigns_for_auction:

    #         # ----- auction bounds ------------------------------------------------
    #         min_bid = 0.1 * c.reach
    #         max_bid = float(c.reach)

    #         # ----- 1) Scarcity  ---------------------------------------------------
    #         seg_key = c.target_segment        
    #         daily_supply  = self.SEGMENT_FREQ.get(seg_key, self.TOTAL_DAILY_USERS / 26)
    #         window_days   = c.end_day - c.start_day + 1
    #         supply_window = daily_supply * window_days
    #         scarcity      = min(c.reach / (supply_window + 1e-6), 1.0)
            
    #         # ----- 2) Urgency -----------------------------------------------------
    #         duration  = c.end_day - c.start_day + 1          # 1–3 days
    #         urgency   = 1.0 / duration                       # 1.0, 0.5, 0.33

    #         # ----- 3) Quality deficit -------------------------------------------
    #         q_deficit = max(0.0, 1.0-Q)
                    
    #         # ----- Blend into ω ---------------------------------------------------
    #         # weights sum to 1
    #         w_s, w_u, w_q = 0.3, 0.1, 0.6
    #         omega = (w_s*scarcity + w_u*urgency + w_q*q_deficit)
            
    #         omega = max(0.0, min(omega, 1.0))                # clamp

    #         bid  = min_bid + omega * (max_bid - min_bid)

    #         if (bid / c.reach) < 0.30 and scarcity < 0.6:
    #             bid = 0.0

    #         # 7)   Late-game 15 % boost  ----------------------------------------
    #         if today >= 7 and bid > 0.0:
    #             bid *= 1.15
                
    #     return bids

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

my_agent_submission = SACPerCampaign(ckpt='./model_checkpoints/sac_pc_v2_sv_12_sd_popart_sa.pth', inference=True)

# ══════════════════════════════════════
# 5. CLI
# ══════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_eps", type=int, default=1000)
    p.add_argument("--ckpt", type=str, default="sac_pc_v2.pth")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_runs", type=int, default=500)
    args = p.parse_args()

    if args.eval_only:
        # evaluate(args.ckpt, args.eval_runs)
        evaluate_v2(args.ckpt, args.eval_runs)
    else:
        train(args.train_eps, args.ckpt)
        evaluate(args.ckpt, args.eval_runs)
