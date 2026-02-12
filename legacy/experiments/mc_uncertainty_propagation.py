import numpy as np
import matplotlib.pyplot as plt

def system_model(x, y):
    """A simple two-layer system model."""
    layer1 = x ** 2
    layer2 = layer1 + y
    return layer2

# Number of simulations
n_simulations = 10000

# Base values
x_base = 2
y_base = 2.5

# Vary x, keep y constant
x_varied = np.random.normal(loc=x_base, scale=0.5, size=n_simulations)
results_x_varied = [system_model(x, y_base) for x in x_varied]

# Vary y, keep x constant
y_varied = np.random.uniform(low=0, high=5, size=n_simulations)
results_y_varied = [system_model(x_base, y) for y in y_varied]

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(results_x_varied, bins=50, edgecolor='black', alpha=0.7)
plt.title("Output Distribution (x varied)")
plt.xlabel("Output Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(results_y_varied, bins=50, edgecolor='black', alpha=0.7)
plt.title("Output Distribution (y varied)")
plt.xlabel("Output Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Calculate statistics
std_x_varied = np.std(results_x_varied)
std_y_varied = np.std(results_y_varied)

print(f"Standard deviation when varying x: {std_x_varied:.2f}")
print(f"Standard deviation when varying y: {std_y_varied:.2f}")

# Calculate sensitivity indices
total_variance = std_x_varied**2 + std_y_varied**2
sensitivity_x = std_x_varied**2 / total_variance
sensitivity_y = std_y_varied**2 / total_variance

print(f"Sensitivity index for x: {sensitivity_x:.2f}")
print(f"Sensitivity index for y: {sensitivity_y:.2f}")