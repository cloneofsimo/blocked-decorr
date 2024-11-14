import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from run import DecorrelationLayer

# Assuming the optimized DecorrelationLayer is already defined as above

def generate_correlated_data(batch_size, num_features, correlation=0.7):
    """
    Generates a batch of correlated data.
    
    Args:
        batch_size (int): Number of samples in the batch.
        num_features (int): Number of features.
        correlation (float): Correlation coefficient between features.
    
    Returns:
        torch.Tensor: Correlated data tensor of shape (batch_size, num_features).
    """
    mean = torch.zeros(num_features)
    cov = torch.full((num_features, num_features), correlation)
    cov.diagonal().fill_(1.0)  # Variance along the diagonal is 1
    data = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((batch_size,))
    return data

# Parameters
batch_size = 1000
num_features = 128  # Must be a multiple of block_size
block_size = 64
correlation = 0.7

# Generate correlated input data
input_data = generate_correlated_data(batch_size, num_features, correlation)

# Instantiate the DecorrelationLayer
decor_layer = DecorrelationLayer(
    num_features=num_features,
    block_size=block_size,
    momentum=0.1,
    eps=1e-5,
    update_every=1,
)

# Switch to training mode to update running statistics
decor_layer.train()

# Forward pass through the DecorrelationLayer
for _ in range(10):
    output_data = decor_layer(input_data)

# Compute covariance matrices
def compute_covariance(data):
    data_mean = data.mean(dim=0, keepdim=True)
    data_centered = data - data_mean
    cov_matrix = (data_centered.t() @ data_centered) / (data.size(0) - 1)
    return cov_matrix

input_cov = compute_covariance(input_data)
output_cov = compute_covariance(output_data.detach())

# Print covariance matrices
print("Input Covariance Matrix:")
print(input_cov)

print("\nOutput Covariance Matrix:")
print(output_cov)

# Visualize the covariance matrices
def plot_covariance_matrix(cov_matrix, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cov_matrix.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.savefig(f"{title}.png")

plot_covariance_matrix(input_cov, "Input Covariance Matrix")
plot_covariance_matrix(output_cov, "Output Covariance Matrix")

