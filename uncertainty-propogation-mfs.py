import numpy as np

def simulate_production_line(num_machines, num_simulations):
    results = []
    for _ in range(num_simulations):
        wip = 0
        throughput = 0
        for machine in range(num_machines):
            # Sample processing time
            proc_time = np.random.normal(loc=10, scale=2)  # Changed 'mean' to 'loc' and 'std' to 'scale'
            # Sample breakdown probability
            if np.random.random() < 0.05:  # 5% chance of breakdown
                repair_time = np.random.exponential(scale=30)
                proc_time += repair_time
            wip += proc_time
        throughput = 480 / wip  # Assuming 480 minutes work day
        results.append(throughput)
    return results

# Run simulation
throughputs = simulate_production_line(num_machines=5, num_simulations=10000)

# Analyze results
mean_throughput = np.mean(throughputs)
std_throughput = np.std(throughputs)
print(f"Mean throughput: {mean_throughput:.2f}")
print(f"Throughput standard deviation: {std_throughput:.2f}")