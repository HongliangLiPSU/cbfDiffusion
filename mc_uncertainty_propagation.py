import numpy as np
import matplotlib.pyplot as plt

def system_model(x, y):
    """A simple two-layer system model."""
    layer1 = x ** 2
    layer2 = layer1 + y
    return layer2

# Number of simulations
n_simulations = 10000

# Input distributions
x = np.random.normal(loc=2, scale=0.5, size=n_simulations)
y = np.random.uniform(low=0, high=5, size=n_simulations)

# Run simulations
results = [system_model(x[i], y[i]) for i in range(n_simulations)]

# Analyze results
mean_output = np.mean(results)
std_output = np.std(results)

print(f"Mean output: {mean_output:.2f}")
print(f"Standard deviation of output: {std_output:.2f}")

# %%Plot histogram of results
plt.hist(results, bins=50, edgecolor='black')
plt.title("Distribution of System Outputs")
plt.xlabel("Output Value")
plt.ylabel("Frequency")
plt.show()
# %%
