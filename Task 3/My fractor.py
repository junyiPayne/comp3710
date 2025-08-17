import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Total number of points
N = 500_000
# Number of parallel chains
chains = 1024
# Steps per chain
steps = N // chains

# Affine transformation parameters: [a,b,c,d,e,f]
params = torch.tensor([
    [0.0,    0.0,    0.0, 0.16, 0.0, 0.0],   # f1
    [0.85,  0.04,  -0.04, 0.85, 0.0, 1.6],   # f2
    [0.2,  -0.26,  0.23, 0.22, 0.0, 1.6],    # f3
    [-0.15, 0.28,  0.26, 0.24, 0.0, 0.44]    # f4
], device=device)

# Probabilities
probs = torch.tensor([0.01, 0.85, 0.07, 0.07], device=device)
cum_probs = torch.cumsum(probs, dim=0)

# Initialize all chains at (0,0)
x = torch.zeros(chains, device=device)
y = torch.zeros(chains, device=device)

# Store all points
points = torch.zeros((N, 2), device=device)

idx = 0
for _ in range(steps):
    # Choose transformations for all chains in parallel
    r = torch.rand(chains, device=device)
    func_idx = torch.searchsorted(cum_probs, r)

    # Fetch affine parameters for all chains
    a = params[func_idx, 0]
    b = params[func_idx, 1]
    c = params[func_idx, 2]
    d = params[func_idx, 3]
    e = params[func_idx, 4]
    f = params[func_idx, 5]

    # Update all points in parallel
    x_new = a*x + b*y + e
    y_new = c*x + d*y + f
    points[idx:idx+chains, 0] = x_new
    points[idx:idx+chains, 1] = y_new

    x, y = x_new, y_new
    idx += chains

# Plot
plt.figure(figsize=(6,10))
plt.scatter(points[:,0].cpu(), points[:,1].cpu(), s=0.1, color="green")
plt.axis('off')
plt.show()
