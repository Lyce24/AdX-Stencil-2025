import argparse
import random
from typing import Set, Dict, Tuple

# Import the game API classes and functions.
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
import numpy as np
import math
from collections import deque

class EmptyAgent(NDaysNCampaignsAgent):
    def __init__(self, name: str = "EmptyAgent"):
        super().__init__()
        self.name = name
        
    def on_new_game(self) -> None:
        # This agent does not store per-game state.
        pass
    
    def get_ad_bids(self) -> Set[BidBundle]:
        # This agent does not place any ad bids.
        return set()
    
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        # This agent does not place any campaign bids.
        return set()

class RandomCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name: str = "RandomCampaignsAgent"):
        super().__init__()
        self.name = name

    def on_new_game(self) -> None:
        # This baseline agent does not store per-game state.
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        # Retrieve the active campaigns for which this agent is eligible to bid.
        active_campaigns = self.get_active_campaigns()
        
        # For each active campaign, compute a randomized bid.
        for campaign in active_campaigns:
            cumulative_reach = self.get_cumulative_reach(campaign)
            cumulative_cost = self.get_cumulative_cost(campaign)
            
            remaining_reach = campaign.reach - cumulative_reach
            remaining_budget = campaign.budget - cumulative_cost
            
            # Skip bidding if the campaign is already fulfilled or there is no budget left.
            if remaining_reach <= 0 or remaining_budget <= 0:
                continue
            
            # Compute a baseline bid per impression.
            baseline_bid = max(0.1, remaining_budget / max(1, remaining_reach))
            
            # Randomize the bid by multiplying with a random factor in [0.5, 1.5].
            random_factor = random.uniform(0.5, 1.5)
            bid_per_item = baseline_bid * random_factor
            bid_per_item = max(0.1, bid_per_item)  # ensure the bid is not below the minimum
            
            # Randomize spending limit: choose a value between a minimum (e.g., half of remaining budget or at least 1)
            # and the remaining budget.
            spending_limit = random.uniform(max(1.0, remaining_budget * 0.5), remaining_budget)
            
            # Create a Bid object. The auction_item is the campaign's target market segment.
            bid_entry = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=bid_per_item,
                bid_limit=spending_limit
            )
            
            # Create a BidBundle that wraps the Bid for the campaign.
            bid_bundle = BidBundle(campaign.uid, spending_limit, {bid_entry})

            bundles.add(bid_bundle)
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        # For each campaign up for auction, choose a random bid within the allowed range.
        for campaign in campaigns_for_auction:
            min_bid = 0.1 * campaign.reach
            max_bid = campaign.reach
            # Random bid from a uniform distribution.
            bid = random.uniform(min_bid, max_bid)
            # Clip the bid to make sure it's in the valid range.
            bid = self.clip_campaign_bid(campaign, bid)
            bids[campaign] = bid
        return bids   

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
    
class CPMAgent(NDaysNCampaignsAgent):

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

            # --------- single-attribute segments (6) ---------
            MarketSegment({'Male'}):                       4956,
            MarketSegment({'Female'}):                     5044,
            MarketSegment({'Young'}):                      4589,
            MarketSegment({'Old'}):                        5411,
            MarketSegment({'LowIncome'}):                  8012,
            MarketSegment({'HighIncome'}):                 1988,
        }
        
        # per-segment EMA parameters
        self.segment_cpm:   Dict[MarketSegment, float] = {s: 500 for s in self.SEGMENT_FREQ}
        self.base_alpha:    Dict[MarketSegment, float] = {s: 0.4 for s in self.SEGMENT_FREQ}
        self.prev_stats:    Dict[MarketSegment, (float, float)] = {
            s: (0.0, 0.0) for s in self.SEGMENT_FREQ
        }

        # max simultaneous active campaigns
        self.max_running = 7
        
        # Track previous cumulative reach & cost to compute daily deltas
        # Keyed by campaign.uid

    def on_new_game(self) -> None:
        for seg in self.prev_stats:
            self.prev_stats[seg] = (0.0, 0.0)
            # self.segment_cpm[seg] = 500
        # clear the previous stats for the new game
        # self.segment_cpi.clear()
        # self.segment_alpha.clear()
        # # reinitialize the segment EMA parameters
        # self.segment_cpi = {seg: 0.5 for seg in self.SEGMENT_FREQ}
        # self.segment_alpha = {seg: 0.3 for seg in self.SEGMENT_FREQ}
        

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        # Retrieve the active campaigns for which this agent is eligible to bid.
        active_campaigns = self.get_active_campaigns()
        
        if not active_campaigns:
            return bundles
        
        today   = self.get_current_day()
        Q       = self.get_quality_score()
        
        # Update the Segment CPI and EMA values
        # Aggregate deltas per segment
        agg_r = {seg: 0.0 for seg in self.SEGMENT_FREQ}
        agg_c = {seg: 0.0 for seg in self.SEGMENT_FREQ}
        for camp in self.get_active_campaigns():
            seg = camp.target_segment
            agg_r[seg] += self.get_cumulative_reach(camp)
            agg_c[seg] += self.get_cumulative_cost(camp)
        
        for seg in self.SEGMENT_FREQ:
            cur_r, cur_c = agg_r[seg], agg_c[seg]
            prev_r, prev_c = self.prev_stats.get(seg, (0.0, 0.0))
            delta_r, delta_c = cur_r - prev_r, cur_c - prev_c
            if delta_r > 0:
                obs_cpi = delta_c / delta_r * 1000.0
                # print(f"Segment {seg}: "
                #       f"Current Reach: {cur_r:.2f}, "
                #         f"Current Cost: {cur_c:.2f}, "
                #         f"Previous Reach: {prev_r:.2f}, "
                #         f"Previous Cost: {prev_c:.2f}, "
                #         f"Delta Reach: {delta_r:.2f}, "
                #         f"Delta Cost: {delta_c:.2f}, "
                #         f"Observed CPI: {obs_cpi:.2f}, "
                #         f"Segment CPI: {self.segment_cpm[seg]:.2f}, "
                #         f"Base Alpha: {self.base_alpha[seg]:.2f}, "
                #         f"Current Alpha: {self.base_alpha[seg] * (1 + np.log1p(delta_r/self.SEGMENT_FREQ[seg])):.2f}")
                
                # Dynamic alpha based on observation significance
                alpha = min(self.base_alpha[seg] * (1 + np.log1p(delta_r/self.SEGMENT_FREQ[seg])), 0.9)
                self.segment_cpm[seg] = (1-alpha)*self.segment_cpm[seg] + alpha*obs_cpi
            self.prev_stats[seg] = (cur_r, cur_c)

        
        # For each active campaign, compute a randomized bid.
        for camp in active_campaigns:
            cumulative_reach = self.get_cumulative_reach(camp)
            cumulative_cost = self.get_cumulative_cost(camp)
            
            rem_reach = camp.reach - cumulative_reach
            rem_budget = camp.budget - cumulative_cost
            seg = camp.target_segment
            
            # Skip bidding if the campaign is already fulfilled or there is no budget left.
            if rem_reach <= 0 or rem_budget <= 0:
                continue
            
            # ───────── Pacing / urgency ─────────
            V      = rem_budget / rem_reach        # advertiser's value CPI
            slack  = 1.0 - cumulative_reach / camp.reach        # 1 ⇒ just started, 0 ⇒ done
            days_left = max(1, camp.end_day - today + 1)

            # ───────── Base price & dynamic premium ─────────
            base_cpi    = 0.8 * (rem_budget / rem_reach)
            
            urgency  = 1.0 + 0.3 * slack / days_left             # ≤ 1+URGENCY_SLOPE
            bid_cpi = min(base_cpi * urgency, V)                # cap at value
            
            jitter   = np.random.uniform(1 - 0.15, 1 + 0.15)
            bid_cpi  = np.clip(bid_cpi * jitter, 0.05, V)  # cap at value

            cap_mul = np.random.uniform(0.8, 1.0)
            spend_limit = max(bid_cpi,                              # ≥ cost of 1 impression
                   min(rem_budget, bid_cpi * rem_reach * cap_mul))

            # print(f"Campaign {camp.uid}: "
            #         f"Remaining Reach: {rem_reach:.2f}, "
            #         f"Remaining Budget: {rem_budget:.2f}, "
            #         f"Target Reach: {camp.reach:.2f}, "
            #         f"Target Budget: {camp.budget:.2f}, "
            #         f"Base CPI: {base_cpi:.2f}, "
            #         f"Bid per Impression: {bid_cpi:.2f}, "
            #         f"Spending Limit: {spend_limit:.2f}")

            # ───────── Package the bid ─────────
            bid = Bid(
                bidder       = self,
                auction_item = seg,
                bid_per_item = bid_cpi,
                bid_limit    = spend_limit
            )
            bundles.add(BidBundle(camp.uid, spend_limit, {bid}))         
        
        return bundles

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        active_now = len(self.get_active_campaigns())
        capacity   = max(0, self.max_running - active_now)
        if capacity == 0:
            return {}

        Q = self.get_quality_score()
        bids = []

        for c in campaigns_for_auction:
            duration = (c.end_day - c.start_day + 1)
            urgency  = 1.0 / duration

            fr = c.reach / (self.SEGMENT_FREQ[c.target_segment] * duration)
            
            if fr > 1.0:
                continue

            p_s = self.segment_cpm.get(c.target_segment, 500)
            expected_cost = 0.7 * c.reach * p_s / 1000

            margin        = max(0.05, 0.20 + 0.30 * fr)
            raw_bid       = expected_cost * (1 + margin) * Q
            bid           = self.clip_campaign_bid(c, raw_bid)
            
            print(f"Campaign {c.uid}: "
                  f"Expected Cost: {expected_cost:.2f}, "
                  f"Bid: {bid:.2f}, "
                  f"Urgency: {urgency:.2f}, "
                  f"Quality Score: {Q:.2f}, "
                  f"Segment CPI: {p_s:.2f}, "
                  f"Reach Factor: {fr:.2f}")

            roi   = (bid - expected_cost) / max(1e-6, expected_cost)
            score = roi + 0.1 * urgency
            bids.append((score, c, bid))

        bids.sort(key=lambda x: x[0], reverse=True)
        selected = bids[:capacity]
        return {c: bid for _, c, bid in selected}

class TestAgent(NDaysNCampaignsAgent):
    def __init__(self, name):
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
        self.baseline_cpi = 0.4
        
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

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        self._update_segment_cpi_from_deltas()
        
        for seg in self.SEGMENT_FREQ:
            print(f"Segment {seg}: Current CPI: {self.segment_cpi[seg]:.2f}")
        
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
            
            diff_mul  = 0.7 + 0.6 * frac     # pay up to 25 % more for hard jobs

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
            #         f"Effective CPI: {bid / c.reach:.2f}")
            
            expected_reach = c.reach / (base_cpi + 1e-6)
            roi = expected_reach / bid
            ranked.append((roi, c, bid))
            
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = ranked[:slots]
        bids = {c: bid for _, c, bid in selected}
        
        return bids
    
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        for campaign in self.get_active_campaigns():
            self._all_campaigns[campaign.uid] = campaign

            bids = set()
            bid_per_item = min(1, max(0.1, (campaign.budget - self.get_cumulative_cost(campaign)) /
                               (campaign.reach - self.get_cumulative_reach(campaign) + 0.0001)))
            total_limit = max(1.0, campaign.budget - self.get_cumulative_cost(campaign))
            auction_item = campaign.target_segment
            bid = Bid(self, auction_item, bid_per_item, total_limit)
            bids.add(bid)
            bundle = BidBundle(campaign_id=campaign.uid, limit=total_limit, bid_entries=bids)
            bundles.add(bundle)
        return bundles

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [BaselineAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
