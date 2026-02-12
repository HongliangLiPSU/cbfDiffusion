import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class InventoryEnvironment:
    def __init__(self, num_products: int, capacity: float, max_order: float,
                 holding_cost: float, stockout_cost: float, order_cost: float):
        self.num_products = num_products
        self.capacity = capacity
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        
    def demand_func(self, seasonality: float = 1.0) -> np.ndarray:
        base_demand = np.random.lognormal(mean=np.log(self.capacity/4), sigma=0.5, size=self.num_products)
        return base_demand * seasonality
    
    def step(self, state: np.ndarray, action: np.ndarray, seasonality: float):
        demand = self.demand_func(seasonality)
        new_state = np.clip(state + action - demand, 0, self.capacity)
        
        # Calculate costs
        holding_cost = self.holding_cost * np.sum(new_state)
        stockout_cost = self.stockout_cost * np.sum(np.maximum(demand - (state + action), 0))
        order_cost = self.order_cost * np.sum(action)
        
        total_cost = holding_cost + stockout_cost + order_cost
        
        return new_state, total_cost, demand

class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x, t):
        t = t.float() / self.num_steps
        t = t.view(-1, 1)
        x = torch.cat([x, t], dim=-1)
        return self.net(x)

    def sample(self, state, num_samples=1, return_trajectory=False):
        batch_size = state.shape[0]
        x = torch.randn(batch_size, num_samples, self.net[-1].out_features, device=state.device)
        
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
        
        final_actions = torch.sigmoid(x)
        
        if return_trajectory:
            return final_actions, trajectory
        else:
            return final_actions

def generate_training_data(env, num_samples: int, episode_length: int):
    states, actions = [], []
    
    for _ in range(num_samples):
        state = np.random.uniform(0, env.capacity, env.num_products)
        for t in range(episode_length):
            seasonality = 1 + 0.3 * np.sin(2 * np.pi * t / 12)  # Seasonal pattern
            
            # Simple heuristic policy: order up to 3/4 capacity if below 1/4 capacity
            action = np.where(state < env.capacity/4, 
                              np.clip(0.75*env.capacity - state, 0, env.max_order), 
                              np.zeros(env.num_products))
            
            # Add some noise to the action to encourage exploration
            action += np.random.normal(0, env.max_order/10, env.num_products)
            action = np.clip(action, 0, env.max_order)
            
            states.append(np.concatenate([state, [seasonality]]))  # Include seasonality in state
            actions.append(action)
            
            state, _, _ = env.step(state, action, seasonality)
    
    states = torch.tensor(np.array(states)).float()
    actions = torch.tensor(np.array(actions)).float() / env.max_order  # Normalize actions to [0, 1] range
    
    return states, actions

def custom_loss(pred, target, state):
    mse_loss = nn.MSELoss()(pred, target)
    stockout_penalty = torch.mean(torch.exp(-state[:, :2]))  # Only consider inventory levels
    overstocking_penalty = torch.mean(torch.relu(state[:, :2] - 0.9))  # Penalize if over 90% capacity
    return mse_loss + 0.1 * stockout_penalty + 0.05 * overstocking_penalty

def train_diffusion_policy(policy, train_data, num_epochs=100, batch_size=64, lr=1e-4):
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for batch in torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True):
            states, actions = batch
            
            noise = torch.randn_like(actions)
            t = torch.randint(0, policy.num_steps, (actions.shape[0],), device=states.device)
            
            x_noisy = actions + noise
            x_input = torch.cat([states, x_noisy], dim=-1)
            
            pred_noise = policy(x_input, t)
            loss = custom_loss(pred_noise, noise, states)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(train_data)}")

def simulate_inventory(policy, env, initial_state, num_steps):
    state = initial_state
    states, actions, demands, costs = [state], [], [], []
    
    for t in range(num_steps):
        seasonality = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
        state_tensor = torch.tensor(np.concatenate([state, [seasonality]])).float().unsqueeze(0)
        
        action = policy.sample(state_tensor).squeeze().detach().numpy()
        action = action * env.max_order  # Denormalize action
        
        next_state, cost, demand = env.step(state, action, seasonality)
        
        states.append(next_state)
        actions.append(action)
        demands.append(demand)
        costs.append(cost)
        state = next_state
    
    return np.array(states), np.array(actions), np.array(demands), np.array(costs)

def visualize_results(states, actions, demands, costs):
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    axs[0].plot(states[:, 0], label='Inventory 1')
    axs[0].plot(states[:, 1], label='Inventory 2')
    axs[0].set_ylabel('Inventory Level')
    axs[0].legend()
    
    axs[1].plot(actions[:, 0], label='Order 1')
    axs[1].plot(actions[:, 1], label='Order 2')
    axs[1].set_ylabel('Order Quantity')
    axs[1].legend()
    
    axs[2].plot(demands[:, 0], label='Demand 1')
    axs[2].plot(demands[:, 1], label='Demand 2')
    axs[2].set_ylabel('Demand')
    axs[2].legend()
    
    axs[3].plot(costs, label='Total Cost')
    axs[3].set_ylabel('Cost')
    axs[3].legend()
    
    for ax in axs:
        ax.set_xlabel('Time Step')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    env = InventoryEnvironment(num_products=2, capacity=250, max_order=80,
                               holding_cost=0.1, stockout_cost=2.0, order_cost=1.0)
    
    # Generate training data
    train_data = generate_training_data(env, num_samples=1000, episode_length=30)
    
    # Initialize and train the policy
    policy = DiffusionPolicy(state_dim=3, action_dim=2)  # 3 = 2 products + 1 seasonality
    train_diffusion_policy(policy, train_data, num_epochs=500, batch_size=64)
    
    # Simulate inventory management
    initial_state = np.array([125, 125])  # Start with half-full inventories
    states, actions, demands, costs = simulate_inventory(policy, env, initial_state, num_steps=100)
    
    # Visualize results
    visualize_results(states, actions, demands, costs)