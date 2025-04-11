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
    



if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [RandomCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=100)
