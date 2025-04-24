import argparse
import random
from typing import Set, Dict

# Import the game API classes and functions.
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle


# class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

#     def __init__(self):
#         # TODO: fill this in (if necessary)
#         super().__init__()
#         self.name = ""  # TODO: enter a name.

#     def on_new_game(self) -> None:
#         # TODO: fill this in (if necessary)
#         pass

#     def get_ad_bids(self) -> Set[BidBundle]:
#         # TODO: fill this in
#         bundles = set()

#         return bundles

#     def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
#         # TODO: fill this in 
#         bids = {}

#         return bids
    

class RandomCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = "Random_Baseline"  # A unique name for your agent.

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

    def __init__(self):
        super().__init__()
        self.name = "Baseline"  # A unique name for your agent.

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


class RuleBasedCampaignsAgent(NDaysNCampaignsAgent):
    def __init__(self):
        super().__init__()
        self.name = "Optimal_RuleBased_AdBidding_Agent"

    def on_new_game(self) -> None:
        # Reset or initialize any per-game state if needed
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        """
        For each active campaign, compute an ad bid bundle.
        The final bid is computed as:
        
            final_bid = baseline_bid * bid_multiplier,
            
        where:
          - baseline_bid = remaining_budget / remaining_impressions (min = 0.1), and
          - bid_multiplier is determined by time pressure and progress.
          
        The spending_limit is similarly set as:
        
            spending_limit = remaining_budget * spending_multiplier.
        """
        bundles = set()
        active_campaigns = self.get_active_campaigns()
        current_day = self.get_current_day()
        
        for campaign in active_campaigns:
            # Compute cumulative stats.
            cum_reach = self.get_cumulative_reach(campaign)
            cum_cost = self.get_cumulative_cost(campaign)
            rem_reach = campaign.reach - cum_reach
            rem_budget = campaign.budget - cum_cost
            
            if rem_reach <= 0 or rem_budget <= 0:
                continue
            
            # 1. Baseline Bid: remaining budget per remaining impression.
            baseline_bid = max(0.1, rem_budget / max(1, rem_reach))
            
            # 2. Progress and Time:
            progress = cum_reach / campaign.reach if campaign.reach > 0 else 0.0
            days_left = campaign.end_day - current_day + 1  # include current day
            
            # 3. Determine Bid Multiplier:
            #    - If early in the campaign (progress < 0.3), impressions add little value.
            #    - When progress is in the "steep" part of the effective reach curve (say 0.3-0.7), bid aggressively.
            #    - If progress is very high (>=0.9), bid conservatively.
            if days_left <= 3 and progress < 0.3:
                # Very few days and low progress: aggressive to catch up.
                bid_multiplier = 1.8  
            elif progress < 0.3:
                bid_multiplier = 0.9  # conserve budget at start.
            elif progress < 0.7:
                bid_multiplier = 1.5  # aggressive to get into the steep area.
            elif progress < 0.9:
                bid_multiplier = 1.2  # moderate aggressive bidding.
            else:
                bid_multiplier = 0.8  # nearly complete: reduce spend.
            
            # Optionally further adjust based on the urgency (days left) even if progress is moderate.
            if days_left <= 2 and progress < 0.7:
                bid_multiplier *= 1.2  # extra boost if time is critically short.
            
            # 4. Compute Spending Multiplier:
            # When time is short, use as much budget as possible.
            if days_left <= 3:
                spending_multiplier = 1.0
            else:
                spending_multiplier = 0.8
            
            # 5. Final Bid and Spending Limit.
            final_bid = baseline_bid * bid_multiplier
            spending_limit = rem_budget * spending_multiplier
            
            # Create the bid object.
            bid_entry = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=final_bid,
                bid_limit=spending_limit
            )
            bundle = BidBundle(campaign.uid, spending_limit, {bid_entry})
            bundles.add(bundle)
        
        return bundles
    
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        """
        Use a rule-based method for campaign bidding.
        (For simplicity we are not using RL for campaign bidding in this example.)
        """
        bids = {}
        current_day = self.get_current_day()
        quality = self.get_quality_score()
        for campaign in campaigns_for_auction:
            min_bid = 0.1 * campaign.reach
            max_bid = campaign.reach
            if quality >= 1.0:
                base_bid = min_bid
            else:
                base_bid = min_bid + (max_bid - min_bid) * (1.0 - quality)
            if current_day >= 7:
                base_bid *= 1.2
            final_bid = self.clip_campaign_bid(campaign, base_bid)
            bids[campaign] = final_bid
        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [BaselineAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)
