import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
        x = torch.randn(batch_size, num_samples, 2, device=state.device)  # Initial noise
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        for t in range(self.num_steps, 0, -1):
            t_tensor = torch.full((batch_size, num_samples), t, device=state.device)
            x_input = torch.cat([state.unsqueeze(1).expand(-1, num_samples, -1), x], dim=-1)
            
            pred_noise = self.forward(x_input.view(-1, x_input.shape[-1]), t_tensor.view(-1))
            x = x - pred_noise.view(batch_size, num_samples, -1)  # Denoising step
            
            if return_trajectory:
                trajectory.append(x.cpu().numpy())
            
            if t > 1:
                noise = torch.randn_like(x)
                sigma = 0.1  # You can adjust this
                x = x + sigma * noise
        
        final_actions = torch.sigmoid(x) * 80  # Scale actions to [0, 80] range
        
        if return_trajectory:
            return final_actions, trajectory
        else:
            return final_actions

def custom_loss(pred, target, state):
    mse_loss = nn.MSELoss()(pred, target)
    stockout_penalty = torch.mean(torch.exp(-state))  # Exponential penalty for low inventory
    overstocking_penalty = torch.mean(torch.relu((state + pred) - 250))
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
        
        if epoch % 10 == 0:  # Print loss every 10 epochs
            print(f"Epoch {epoch}, Avg Loss: {total_loss / len(train_data)}")

def simulate_inventory(policy, initial_state, num_steps, demand_func):
    state = initial_state
    states = [state]
    actions = []
    demands = []
    
    for _ in range(num_steps):
        action = policy.sample(torch.tensor(state).float().unsqueeze(0)).squeeze().detach().numpy()
        action = np.clip(action, 0, 80)  # Apply constraints
        
        demand = demand_func()
        demands.append(demand)
        next_state = state + action - demand
        next_state = np.clip(next_state, 0, 250)  # Apply constraints
        
        states.append(next_state)
        actions.append(action)
        state = next_state
    
    return np.array(states), np.array(actions), np.array(demands)

def generate_training_data(num_samples, demand_func):
    states = []
    actions = []
    
    for _ in range(num_samples):
        state = np.random.uniform(0, 250, 2)
        action = np.random.uniform(0, 80, 2)
        demand = demand_func()
        
        next_state = np.clip(state + action - demand, 0, 250)
        
        states.append(state)
        actions.append(action)
    
    states = torch.tensor(np.array(states)).float()
    actions = torch.tensor(np.array(actions)).float() / 80  # Normalize actions to [0, 1] range
    
    return states, actions

def demand_func(demand_max1=150, demand_max2=75, interval1=4, interval2=3):
    demand1 = np.random.lognormal(mean=np.log(demand_max1), sigma=0.5) / interval1
    demand2 = np.random.lognormal(mean=np.log(demand_max2), sigma=0.5) / interval2
    return np.array([demand1, demand2])

def visualize_action_distribution(policy, state):
    with torch.no_grad():
        actions, trajectory = policy.sample(state, num_samples=1000, return_trajectory=True)
    
    actions = actions.squeeze().numpy()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.kdeplot(actions[:, 0], shade=True)
    plt.title('Distribution of Action 1')
    plt.xlabel('Action Value')
    plt.ylabel('Density')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(actions[:, 1], shade=True)
    plt.title('Distribution of Action 2')
    plt.xlabel('Action Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()
    
    return trajectory

def visualize_diffusion_process(trajectory):
    num_steps = len(trajectory)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Diffusion Process Visualization', fontsize=16)
    
    for i, step_actions in enumerate(reversed(trajectory)):
        if i >= 10:  # We'll show only 10 steps
            break
        
        row = i // 5
        col = i % 5
        
        sns.kdeplot(step_actions[0, :, 0], ax=axes[row, col], shade=True)
        sns.kdeplot(step_actions[0, :, 1], ax=axes[row, col], shade=True)
        
        axes[row, col].set_title(f'Step {num_steps - i}')
        axes[row, col].set_xlim(-0.2, 1.2)  # Adjusted for sigmoid output
        axes[row, col].set_ylim(0, 5)
        
        if col == 0:
            axes[row, col].set_ylabel('Density')
        if row == 1:
            axes[row, col].set_xlabel('Action Value')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    policy = DiffusionPolicy(state_dim=2, action_dim=2)
    
    train_data = generate_training_data(50000, demand_func)  # Increased training data
    train_diffusion_policy(policy, train_data, num_epochs=1000)  # Increased epochs
    
    initial_state = torch.tensor([[125, 125]], dtype=torch.float32)  # Start with half-full inventories
    
    # Visualize action distribution
    trajectory = visualize_action_distribution(policy, initial_state)
    
    # Visualize diffusion process
    visualize_diffusion_process(trajectory)
    
    # Run simulation and plot results
    states, actions, demands = simulate_inventory(policy, initial_state.numpy()[0], num_steps=30, demand_func=demand_func)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(states[:, 0], label='Inventory 1')
    plt.plot(states[:, 1], label='Inventory 2')
    plt.xlabel('Time Step')
    plt.ylabel('Inventory Level')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(actions[:, 0], label='Action 1')
    plt.plot(actions[:, 1], label='Action 2')
    plt.xlabel('Time Step')
    plt.ylabel('Action Level')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(demands[:, 0], label='Demand 1')
    plt.plot(demands[:, 1], label='Demand 2')
    plt.xlabel('Time Step')
    plt.ylabel('Demand Level')
    plt.legend()
    
    plt.tight_layout()
    plt.show()