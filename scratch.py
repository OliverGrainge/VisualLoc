import matplotlib.pyplot as plt
import numpy as np
import torch

# Set the parameters for the Beta distribution
alpha = 1.4  # shape parameter
beta = 1.1  # shape parameter

# Create a Beta distribution
beta_distribution = torch.distributions.beta.Beta(alpha, beta)

# Sample from the Beta distribution
sampled_values = beta_distribution.sample().item()
print(sampled_values)
