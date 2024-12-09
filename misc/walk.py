import matplotlib.pyplot as plt
import numpy as np


def smooth(data, window_size):
    """
    Smooths the input data using a moving average.

    Args:
      data: A NumPy array of shape (n,) representing the input data.
      window_size: An integer specifying the size of the moving average window.

    Returns:
      A NumPy array of the same shape as the input data, containing the smoothed values.
    """

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode="same")
    return smoothed


def random_walk_with_drift(n_steps, drift, var):
    """
    Generates a random walk with drift.

    Args:
      n_steps: The number of steps in the random walk.
      drift: The drift parameter (average step size).
        var: The variance of the step size.

    Returns:
      A NumPy array representing the random walk.
    """

    steps = np.random.normal(drift, var, n_steps) + drift
    walk = np.cumsum(steps)
    return walk


# Parameters
n_walks = 1000  # Number of random walks to generate
n_steps = 100  # Number of steps in each walk
n_slices = 5  # Number of slices to divide the walk into
drift = 0.01  # Drift parameter
var = 0.05  # Variance of the step size

walks = []
for _ in range(n_walks):
    walk = random_walk_with_drift(n_steps, drift, var)
    walks.append(walk)

# ---  Plot 1: Exponential of Random Walks ---
plt.figure(figsize=(10, 6))
exp_walks = [np.exp(walk) for walk in walks]
for walk in exp_walks:
    plt.plot(walk)

plt.xlabel("Steps")
plt.ylabel("Value")
plt.show()

# --- Plot 2: Exponential Random Walk Slices ---
plt.figure(figsize=(10, 6))

slice_indices = np.linspace(0, n_steps, n_slices + 1, dtype=int)
for i in range(n_slices):
    slice_vals = []
    for walk in exp_walks:
        slice_vals.append(walk[slice_indices[i]])

    # Calculate histogram and normalize to get distribution
    hist, bin_edges = np.histogram(slice_vals, bins=100, density=True)
    hist = smooth(hist, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot as line plot
    plt.plot(bin_centers, hist, label=f"After {slice_indices[i+1]} Steps")

plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# --- Plot 3: Linear Random Walks ---
plt.figure(figsize=(10, 6))
for walk in walks:
    plt.plot(walk)

plt.xlabel("Steps")
plt.ylabel("Value")
plt.show()

# --- Plot 4: Slices of Linear Random Walks ---
plt.figure(figsize=(10, 6))

slice_indices = np.linspace(0, n_steps, n_slices + 1, dtype=int)
for i in range(n_slices):
    slice_vals = []
    for walk in walks:
        slice_vals.append(walk[slice_indices[i]])

    # Calculate histogram and normalize to get distribution
    hist, bin_edges = np.histogram(slice_vals, bins=100, density=True)
    hist = smooth(hist, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot as line plot
    plt.plot(bin_centers, hist, label=f"After {slice_indices[i+1]} Steps")

plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# --- Plot 4: Distributions After Differencing ---

plt.figure(figsize=(10, 6))

slice_indices = np.linspace(1, n_steps, n_slices + 1, dtype=int)
for i in range(n_slices):
    slice_vals = []
    for walk in walks:
        slice_vals.append(walk[slice_indices[i]] - walk[slice_indices[i] - 1])

    # Calculate histogram and normalize to get distribution
    # slic_vals = [val for val in slice_vals if abs(val) < var]
    hist, bin_edges = np.histogram(slice_vals, bins=100, density=True)
    hist = smooth(hist, 10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot as line plot
    plt.plot(bin_centers, hist, label=f"After {slice_indices[i+1]} Steps")

plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
