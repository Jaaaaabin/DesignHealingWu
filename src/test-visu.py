import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Generate random data points
np.random.seed(0)
num_points = 10
data = np.random.rand(num_points, 2)

# Calculate pairwise Euclidean distances
distances = distance_matrix(data, data)

# Create a heatmap of Euclidean distances
plt.figure(figsize=(8, 6))
plt.imshow(distances, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Euclidean Distance')
plt.title('Euclidean Distance Heatmap')
plt.xticks(range(num_points), [f'Point {i+1}' for i in range(num_points)], rotation=45)
plt.yticks(range(num_points), [f'Point {i+1}' for i in range(num_points)])
plt.tight_layout()
plt.show()