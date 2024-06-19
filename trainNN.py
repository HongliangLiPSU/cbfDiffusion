import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the MATLAB file
mat = scipy.io.loadmat('safe_trajectories.mat')

# Extract the safe trajectories
safe_trajectories = mat['safe_trajectories']

# Prepare the data for training
X = safe_trajectories[:, :-1].T  # Features (current state)
y = safe_trajectories[:, 1:].T   # Labels (next state)

# Normalize the data if necessary
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std
y_mean = np.mean(y, axis=0)
y_std = np.std(y, axis=0)
y = (y - y_mean) / y_std

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'cbf_model.pth')

import torch
import torch.onnx

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('cbf_model.pth'))
model.eval()

# Dummy input for model tracing (adjust size according to your input size)
dummy_input = torch.randn(1, 2)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "cbf_model.onnx", input_names=['input'], output_names=['output'])

