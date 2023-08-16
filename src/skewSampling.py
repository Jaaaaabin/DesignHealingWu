import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, truncnorm

# Set the parameters of the skew normal distribution
delta_loc = 0.3

alpha = 1 # Shape parameter (controls skewness)
loc = 53.5    # Location parameter (mean)
scale = delta_loc/3 # Scale parameter (standard deviation)

# Generate random samples from the skew normal distribution
num_samples = 100
samples = skewnorm.rvs(alpha, loc=loc, scale=scale, size=num_samples)

# Plot a histogram of the generated samples
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='Sampled Data')

# Plot the probability density function (PDF) of the skew normal distribution
x = np.linspace(min(samples), max(samples), 100)
pdf = skewnorm.pdf(x, alpha, loc=loc, scale=scale)
plt.plot(x, pdf, 'r', label='Skew Normal PDF')
plt.axvline(x=loc-delta_loc)
plt.axvline(x=loc+delta_loc)
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Sampling from Skew Normal Distribution')
plt.legend()
plt.show()

# # Set the parameters of the truncated normal distribution
# a = 2.0   # Lower bound
# b = 5.0   # Upper bound
# loc = 3.0 # Mean
# scale = 1.0 # Standard deviation

# # Generate random samples from the truncated normal distribution
# num_samples = 1000
# samples = truncnorm.rvs(a, b, loc=loc, scale=scale, size=num_samples)

# # Plot a histogram of the generated samples
# plt.hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Sampled Data')

# # Plot the probability density function (PDF) of the truncated normal distribution
# x = np.linspace(min(samples), max(samples), 100)
# pdf = truncnorm.pdf(x, a, b, loc=loc, scale=scale)
# plt.plot(x, pdf, 'r', label='Truncated Normal PDF')

# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Sampling from Truncated Normal Distribution')
# plt.legend()
# plt.show()

# # Set the parameters of the truncated normal distribution
# center = 5
# delta_c = 0.3
# scale = 5 # Standard deviation

# # Generate random samples from the truncated normal distribution
# num_samples = 1000
# truncated_samples = truncnorm.rvs(-delta_c, delta_c, loc=center, scale=scale, size=num_samples)

# # Apply skewness to the truncated samples
# skewness = 4.0
# skewed_samples = truncated_samples

# # Plot a histogram of the generated samples
# plt.hist(skewed_samples, bins=30, density=True, alpha=0.6, color='g', label='Sampled Data')

# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.title('Sampling from Skew Truncated Normal Distribution')
# plt.legend()
# plt.show()
