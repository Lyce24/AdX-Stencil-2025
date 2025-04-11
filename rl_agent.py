import argparse
import random
from typing import Set, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import the game API classes and functions.
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle

#########################################
# Define the MLP models for bidding.
#########################################

class AdBiddingPolicy(nn.Module):
    """
    This MLP policy maps a 4-dimensional state to two outputs.
    The state consists of:
      1. Normalized current day (day / 10)
      2. Campaign progress (cumulative reach / target reach)
      3. Remaining budget ratio (remaining budget / campaign budget)
      4. Current quality score
    The outputs (after the sigmoid activation) are in (0, 1). By adding an offset of 0.5,
    they become multipliers roughly in the [0.5, 1.5] range.
    """
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=2):
        super(AdBiddingPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    
#########################################
# RL Agent definition.
#########################################

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):
    def __init__(self):
        super().__init__()
        self.name = "RL_Ad_Bidding_Agent"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = AdBiddingPolicy(input_dim=4, hidden_dim=32, output_dim=2).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 0.99  # Discount factor for future rewards
        
        # Store per-day log probabilities and rewards
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        
        # For computing daily rewards, we track previous profit, quality,
        # and for each campaign, the previous cumulative reach.
        self.prev_profit = 0.0
        self.prev_quality = self.get_quality_score()  # typically initialized to 1.0
        self.prev_reach: Dict[int, int] = {}
        
        # Reward weight hyperparameters (tunable)
        self.w_profit = 1.0
        self.w_progress = 50.0
        self.w_quality = 20.0
        self.w_penalty = 1.0  # In this example, no penalty is applied.
    
    def on_new_game(self) -> None:
        # Reset per-episode memory.
        self.log_probs.clear()
        self.rewards.clear()
        self.prev_profit = 0.0
        self.prev_quality = self.get_quality_score()
        self.prev_reach.clear()
        
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        active_campaigns = self.get_active_campaigns()
        day_log_prob = torch.tensor(0.0, device=self.device)
        
        # Loop through each active campaign.
        for campaign in active_campaigns:
            # Gather state information.
            cumulative_reach = self.get_cumulative_reach(campaign)
            cumulative_cost = self.get_cumulative_cost(campaign)
            remaining_reach = campaign.reach - cumulative_reach
            remaining_budget = campaign.budget - cumulative_cost
            
            if remaining_reach <= 0 or remaining_budget <= 0:
                continue
            
            # Baseline bid per impression (similar to Tier 1 agent heuristic).
            baseline_bid = max(0.1, remaining_budget / max(1, remaining_reach))
            
            # Build a feature vector:
            # 1. current day normalized (game length assumed 10 days)
            # 2. progress (fraction of campaign fulfilled)
            # 3. remaining budget ratio
            # 4. current quality score.
            current_day_norm = self.get_current_day() / 10.0
            progress = cumulative_reach / campaign.reach if campaign.reach > 0 else 0.0
            rem_budget_ratio = remaining_budget / campaign.budget if campaign.budget > 0 else 0.0
            quality = self.get_quality_score()
            
            state = torch.tensor([current_day_norm, progress, rem_budget_ratio, quality],
                        dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Forward pass through the policy network.
            policy_output = self.policy(state)  # shape: (1,2)
            # Create a Gaussian distribution with fixed standard deviation.
            sigma = torch.full_like(policy_output, 0.1)
            dist = torch.distributions.Normal(policy_output, sigma)
            # Sample action using the reparameterization trick.
            action = dist.rsample()
            # Record the log probability.
            log_prob = dist.log_prob(action).sum()
            day_log_prob += log_prob
            
            bid_factor = 0.5 + action[0, 0].item()   # multiplier for the baseline bid.
            limit_factor = 0.5 + action[0, 1].item()   # multiplier for the spending limit.
            
            # Compute final bid and spending limit.
            final_bid = baseline_bid * bid_factor
            spending_limit = remaining_budget * limit_factor
            
            # Create the Bid and BidBundle objects.
            bid_entry = Bid(
                bidder=self,
                auction_item=campaign.target_segment,
                bid_per_item=final_bid,
                bid_limit=spending_limit
            )
            bundle = BidBundle(campaign.uid, spending_limit, {bid_entry})

            bundles.add(bundle)
            
            # Track the campaign’s previous cumulative reach.
            if campaign.uid not in self.prev_reach:
                self.prev_reach[campaign.uid] = 0
        
        # Store the aggregated log probability for this day.
        self.log_probs.append(day_log_prob)
        
        # Compute the composite daily reward.
        # 1. Profit Difference: current cumulative profit minus previous.
        current_profit = self.get_cumulative_profit()
        r_profit = current_profit - self.prev_profit
        print(f"Current Profit: {current_profit:.2f}, Previous Profit: {self.prev_profit:.2f}, "
              f"Profit Reward: {r_profit:.2f}")
        
        # 2. Campaign Progress: average improvement in the fraction of reach achieved.
        progress_improvements = []
        for campaign in active_campaigns:
            current_reach = self.get_cumulative_reach(campaign)
            prev = self.prev_reach.get(campaign.uid, 0)
            prog_improve = (current_reach - prev) / campaign.reach if campaign.reach > 0 else 0.0
            progress_improvements.append(prog_improve)
            self.prev_reach[campaign.uid] = current_reach
        r_progress = sum(progress_improvements) / len(progress_improvements) if progress_improvements else 0.0
        print(f"Progress Improvements: {progress_improvements}, "
                f"Average Progress Reward: {r_progress:.2f}")
        
        # 3. Quality Score Improvement.
        current_quality = self.get_quality_score()
        r_quality = current_quality - self.prev_quality
        print(f"Current Quality: {current_quality:.2f}, Previous Quality: {self.prev_quality:.2f}, "
              f"Quality Reward: {r_quality:.2f}")
        
        # 4. (Optional) A penalty component can be added if needed.
        r_penalty = 0.0
        
        daily_reward = (self.w_profit * r_profit +
                        self.w_progress * r_progress +
                        self.w_quality * r_quality -
                        self.w_penalty * r_penalty)
        
        self.rewards.append(daily_reward)
        
        # Update trackers.
        self.prev_profit = current_profit
        self.prev_quality = current_quality

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


    def update_policy(self):
        """
        After the 10-day episode, compute the discounted returns from the collected daily rewards.
        Normalize the returns, then update the policy using the REINFORCE loss function:
            Loss = - Sum_day [ log_prob(day) * Return(day) ]
        """
        R = 0.0
        raw_returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            raw_returns.insert(0, R)
        raw_returns = torch.tensor(raw_returns, dtype=torch.float32, device=self.device)
        avg_raw_return = raw_returns.mean().item()
        
        # Normalize the returns.
        normalized_returns = (raw_returns - raw_returns.mean()) / (raw_returns.std() + 1e-5)
        
        losses = []
        for log_prob, R_val in zip(self.log_probs, normalized_returns):
            losses.append(-log_prob * R_val)
        loss = torch.stack(losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(f"Policy Update: Loss = {loss.item():.4f}, "
            f"Avg Raw Return = {avg_raw_return:.4f}, "
            f"Avg Normalized Return = {normalized_returns.mean().item():.4f}")

#########################################
# Training loop for the RL agent.
#########################################
    
def train_rl_ad_bidding(agent: MyNDaysNCampaignsAgent, num_episodes: int = 500, eval_every: int = 50):
    simulator = AdXGameSimulator()
    # Dummy opponents: using a baseline Tier1 agent.
    dummy_agents = [Tier1NDaysNCampaignsAgent(name=f"Dummy Agent {i+1}") for i in range(9)]
    
    for episode in range(1, num_episodes + 1):
        # Run one episode (a full 10-day game).
        simulator.run_simulation(agents=[agent] + dummy_agents, num_simulations=1)
        
        # After the episode, update the policy using the accumulated rewards and log probabilities.
        agent.update_policy()
        
        # Optionally save the model every few episodes.
        # if episode % eval_every == 0:
            # torch.save({'policy': agent.policy.state_dict()}, f"rl_ad_bid_agent_episode_{episode}.pth")
            # print(f"Episode {episode}: Model saved.")

        print(f"Episode {episode} complete.")

        # Clear episode memory.
        agent.on_new_game()

if __name__ == "__main__":
    agent = MyNDaysNCampaignsAgent()
    
    mode = 'train'
    episodes = 1000
    
    if mode == 'train':
        print("Training with environment interaction started ...")
        train_rl_ad_bidding(agent, num_episodes=episodes)
        torch.save({'policy': agent.policy.state_dict()}, "2mlp_bid_model_final.pth")
        print("Training completed.")
    elif mode == 'eval':
        try:
            agent.policy.load_state_dict(torch.load("2mlp_bid_model_final.pth", map_location=agent.device)['policy'])
        except Exception as e:
            print("Error loading model:", e)

        print("Running simulation evaluation with the trained agent ...")
        # Build a set of agents: our agent plus 9 dummy agents.
        test_agents = [agent] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i+1}") for i in range(9)]
        simulator = AdXGameSimulator()
        simulator.run_simulation(agents=test_agents, num_simulations=500)