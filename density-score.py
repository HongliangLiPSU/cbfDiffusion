import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define a Gaussian Mixture Model
def gaussian_mixture(x, means, covariances, weights):
    n = len(weights)
    pdf = np.zeros(x.shape[0])
    for i in range(n):
        pdf += weights[i] * multivariate_normal.pdf(x, mean=means[i], cov=covariances[i])
    return pdf

# Compute the score function for the mixture
def score_function(x, means, covariances, weights):
    n = len(weights)
    pdf = gaussian_mixture(x, means, covariances, weights)
    score = np.zeros_like(x)
    for i in range(n):
        mean_diff = x - means[i]
        inv_cov = np.linalg.inv(covariances[i])
        component_pdf = multivariate_normal.pdf(x, mean=means[i], cov=covariances[i])
        score += (weights[i] * component_pdf)[:, np.newaxis] * (-inv_cov @ mean_diff.T).T
    return score / pdf[:, np.newaxis]

# Parameters for Gaussian Mixture Model
means = [np.array([2, 2]), np.array([-2, -2])]
covariances = [np.eye(2), np.eye(2)]
weights = [0.5, 0.5]

# Generate a grid of points
x, y = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))  # Reduced grid density
pos = np.dstack((x, y)).reshape(-1, 2)

# Compute PDF and score function on the grid
pdf = gaussian_mixture(pos, means, covariances, weights).reshape(30, 30)
score = score_function(pos, means, covariances, weights).reshape(30, 30, 2)

# Plot the density and score function
fig, ax = plt.subplots(figsize=(8, 8))
contour = ax.contourf(x, y, pdf, levels=20, cmap='plasma', alpha=0.8)  # Changed color scheme
cbar = fig.colorbar(contour, ax=ax)
cbar.ax.set_ylabel('Density')
ax.quiver(x, y, score[:, :, 0], score[:, :, 1], color='black', scale=20)  # Reduced scale for arrows
ax.set_title('Score vs. Density Function')
plt.show()