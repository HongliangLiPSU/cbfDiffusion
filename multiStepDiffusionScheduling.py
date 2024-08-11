import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy

class InventoryEnvironment:
    def __init__(self, num_products, capacity, max_order, holding_cost, stockout_cost, order_cost, lead_time, demand_variability):
        self.num_products = num_products
        self.capacity = capacity
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.demand_variability = demand_variability
        self.pending_orders = [[] for _ in range(num_products)]
        self.state = np.zeros(num_products)
        self.target_service_level = 0.95  # Target service level
        
    def demand_func(self, seasonality: float = 1.0) -> np.ndarray:
        base_demand = np.random.lognormal(mean=np.log(self.capacity/4), sigma=self.demand_variability, size=self.num_products)
        return base_demand * seasonality
    
    def step(self, action, seasonality):
        demand = self.demand_func(seasonality)
        
        # Process pending orders
        for i in range(self.num_products):
            if self.pending_orders[i]:
                self.state[i] = min(self.state[i] + self.pending_orders[i].pop(0), self.capacity)
        
        # Add new orders to pending
        for i in range(self.num_products):
            self.pending_orders[i].append(action[i])
            if len(self.pending_orders[i]) > self.lead_time:
                self.pending_orders[i].pop(0)
        
        new_state = np.clip(self.state - demand, 0, self.capacity)
        
        # Calculate costs
        holding_cost = self.holding_cost * np.sum(new_state)
        stockout = np.maximum(demand - self.state, 0)
        stockout_cost = self.stockout_cost * np.sum(stockout)
        order_cost = self.order_cost * np.sum(action)
        
        total_cost = holding_cost + stockout_cost + order_cost
        
        # Calculate service level
        service_level = np.mean(stockout == 0)
        
        # Adjust reward based on service level
        service_level_penalty = 1000 * max(0, self.target_service_level - service_level)
        
        reward = -total_cost - service_level_penalty
        
        self.state = new_state
        return self.get_state(), reward, demand

    def get_state(self):
        return np.concatenate([self.state, np.array([sum(orders) for orders in self.pending_orders])])

    def reset(self):
        self.state = np.random.uniform(0, self.capacity, self.num_products)
        self.pending_orders = [[] for _ in range(self.num_products)]
        return self.get_state()

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_steps=10, prediction_horizon=5):
        super().__init__()
        self.num_steps = num_steps
        self.prediction_horizon = prediction_horizon
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim * prediction_horizon + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * prediction_horizon)
        )
        
    def forward(self, x, t):
        t = t.float() / self.num_steps
        t = t.view(-1, 1)
        x = torch.cat([x, t], dim=-1)
        return self.net(x)

    def sample(self, state, num_samples=1, return_trajectory=False):
        batch_size = state.shape[0]
        x = torch.randn(batch_size, num_samples, self.action_dim * self.prediction_horizon, device=state.device)
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        for t in range(self.num_steps, 0, -1):
            t_tensor = torch.full((batch_size, num_samples), t, device=state.device)
            x_input = torch.cat([state.unsqueeze(1).expand(-1, num_samples, -1), x], dim=-1)
            
            pred_noise = self.forward(x_input.view(-1, x_input.shape[-1]), t_tensor.view(-1))
            x = x - pred_noise.view(batch_size, num_samples, -1)
            
            if return_trajectory:
                trajectory.append(x.cpu().numpy())
            
            if t > 1:
                noise = torch.randn_like(x)
                sigma = 0.1
                x = x + sigma * noise
        
        final_actions = torch.sigmoid(x).view(batch_size, num_samples, self.prediction_horizon, self.action_dim)
        
        if return_trajectory:
            return final_actions, trajectory
        else:
            return final_actions

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.stack, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

def train_diffusion_policy(policy, env, num_epochs=2000, batch_size=128, lr=1e-4, gamma=0.99, target_update=10, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.StepLR(optimizer, step_size=200, gamma=0.5)
    replay_buffer = ReplayBuffer(100000)  # Increased buffer size
    target_policy = copy.deepcopy(policy)
    
    epsilon = epsilon_start
    
    for epoch in tqdm(range(num_epochs)):
        state = env.reset()
        total_reward = 0
        
        for t in range(100):  # episode length
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = np.random.uniform(0, env.max_order, env.num_products)
            else:
                action = policy.sample(state_tensor).squeeze().detach().numpy()
                action = action[0, :env.num_products]  # Take only the first action for each product
            
            next_state, reward, _ = env.step(action, 1.0)  # Assuming no seasonality for simplicity
            replay_buffer.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states = replay_buffer.sample(batch_size)
                states = torch.tensor(states).float()
                actions = torch.tensor(actions).float()
                rewards = torch.tensor(rewards).float().unsqueeze(1)
                next_states = torch.tensor(next_states).float()

                # Compute current Q-values
                current_actions = actions.unsqueeze(1).repeat(1, policy.prediction_horizon, 1)
                current_q = policy(torch.cat([states, current_actions.view(batch_size, -1)], dim=-1), torch.zeros(batch_size))
                current_q = current_q.view(batch_size, policy.prediction_horizon, -1)[:, 0, :]  # Take only the first step prediction

                # Compute next Q-values
                next_actions = target_policy.sample(next_states).squeeze(1)
                next_q = target_policy(torch.cat([next_states, next_actions.view(batch_size, -1)], dim=-1), torch.zeros(batch_size))
                next_q = next_q.view(batch_size, policy.prediction_horizon, -1)[:, 0, :]  # Take only the first step prediction

                # Compute expected Q-values
                expected_q = rewards + gamma * next_q

                # Compute loss
                loss = nn.MSELoss()(current_q, expected_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if epoch % target_update == 0:
            target_policy.load_state_dict(policy.state_dict())

        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return policy

def simulate_inventory(policy, env, initial_state, num_steps):
    state = initial_state
    states, actions, demands, rewards = [], [], [], []
    
    for t in range(num_steps):
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        
        action = policy.sample(state_tensor).squeeze().detach().numpy()
        action = action[0, :env.num_products]  # Take only the first action for each product
        
        next_state, reward, demand = env.step(action, seasonality)
        
        states.append(state)
        actions.append(action)
        demands.append(demand)
        rewards.append(reward)
        state = next_state
    
    return np.array(states), np.array(actions), np.array(demands), np.array(rewards)

def visualize_results(states, actions, demands, rewards):
    fig, axs = plt.subplots(5, 1, figsize=(12, 20))
    
    # Inventory levels
    for i in range(states.shape[1] // 2):  # Only plot actual inventory, not pending orders
        axs[0].plot(states[:, i], label=f'Inventory {i+1}')
    axs[0].set_ylabel('Inventory Level')
    axs[0].legend()
    
    # Order quantities
    for i in range(actions.shape[1]):
        axs[1].plot(actions[:, i], label=f'Order {i+1}')
    axs[1].set_ylabel('Order Quantity')
    axs[1].legend()
    
    # Demands
    for i in range(demands.shape[1]):
        axs[2].plot(demands[:, i], label=f'Demand {i+1}')
    axs[2].set_ylabel('Demand')
    axs[2].legend()
    
    # Rewards
    axs[3].plot(rewards, label='Reward')
    axs[3].set_ylabel('Reward')
    axs[3].legend()
    
    # Cumulative metrics
    cumulative_rewards = np.cumsum(rewards)
    service_level = np.mean(states[:, :states.shape[1]//2] > 0, axis=1)
    axs[4].plot(cumulative_rewards, label='Cumulative Reward')
    axs[4].set_ylabel('Cumulative Reward')
    axs[4].legend()
    ax2 = axs[4].twinx()
    ax2.plot(service_level, label='Service Level', color='r')
    ax2.set_ylabel('Service Level')
    ax2.legend(loc='lower right')
    
    for ax in axs:
        ax.set_xlabel('Time Step')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Total reward: {np.sum(rewards):.2f}")
    print(f"Average service level: {np.mean(service_level):.2%}")
    print(f"Average inventory level: {np.mean(states[:, :states.shape[1]//2]):.2f}")
    print(f"Average order quantity: {np.mean(actions):.2f}")

if __name__ == "__main__":
    env = InventoryEnvironment(num_products=2, capacity=250, max_order=80,
                               holding_cost=0.1, stockout_cost=2.0, order_cost=1.0,
                               lead_time=2, demand_variability=0.7)
    
    # Initialize the policy
    policy = DiffusionPolicy(state_dim=4, action_dim=2, prediction_horizon=5)  # 4 = 2 products + 2 pending orders
    
    # Train the policy
    trained_policy = train_diffusion_policy(policy, env, num_epochs=2000, batch_size=128)
    
    # Simulate inventory management
    initial_state = env.reset()
    states, actions, demands, rewards = simulate_inventory(trained_policy, env, initial_state, num_steps=200)
    
    # Visualize results
    visualize_results(states, actions, demands, rewards)