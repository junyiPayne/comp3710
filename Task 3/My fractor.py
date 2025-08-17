import torch
import matplotlib.pyplot as plt
import numpy as np

# ---------------- DEVICE SELECTION ----------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ---------------- PARAMETERS ----------------
N = 500_000        # total points
chains = 1024      # parallel chains
steps = N // chains

# Affine transformation parameters: [a,b,c,d,e,f]
params = torch.tensor([
    [0.0,    0.0,    0.0, 0.16, 0.0, 0.0],   # f1
    [0.85,  0.04,  -0.04, 0.85, 0.0, 1.6],   # f2
    [0.2,  -0.26,  0.23, 0.22, 0.0, 1.6],    # f3
    [-0.15, 0.28,  0.26, 0.24, 0.0, 0.44]    # f4
], device=device)

probs = torch.tensor([0.01, 0.85, 0.07, 0.07], device=device)
cum_probs = torch.cumsum(probs, dim=0)

# ---------------- GENERATE POINTS ----------------
x = torch.zeros(chains, device=device)
y = torch.zeros(chains, device=device)
points = torch.zeros((N, 2), device=device)

idx = 0
for _ in range(steps):
    r = torch.rand(chains, device=device)
    func_idx = torch.searchsorted(cum_probs, r)
    a, b, c, d, e, f = [params[func_idx, i] for i in range(6)]
    x_new = a*x + b*y + e
    y_new = c*x + d*y + f
    points[idx:idx+chains, 0] = x_new
    points[idx:idx+chains, 1] = y_new
    x, y = x_new, y_new
    idx += chains

points_cpu = points.cpu().numpy()

# ---------------- FRACTAL DIMENSION (BOX COUNTING) ----------------
def fractal_dimension(Z, threshold=1e-5):
    Z = np.array(Z)
    x_min, x_max = Z[:,0].min(), Z[:,0].max()
    y_min, y_max = Z[:,1].min(), Z[:,1].max()
    sizes = 2**np.arange(1, 10)
    counts = []
    for size in sizes:
        nx = int(np.ceil((x_max-x_min)/size))
        ny = int(np.ceil((y_max-y_min)/size))
        grid = np.zeros((nx, ny))
        ix = ((Z[:,0]-x_min)/size).astype(int)
        iy = ((Z[:,1]-y_min)/size).astype(int)
        grid[ix, iy] = 1
        counts.append(np.sum(grid>0))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

fd = fractal_dimension(points_cpu)
print(f"Estimated fractal dimension: {fd:.4f}")

# ---------------- VISUALIZATION ----------------
plt.figure(figsize=(12,10))

# Classic green fern
plt.subplot(2,2,1)
plt.scatter(points_cpu[:,0], points_cpu[:,1], s=0.1, color='green')
plt.title("Classic Green Fern")
plt.axis('off')

# Color by iteration (index)
plt.subplot(2,2,2)
plt.scatter(points_cpu[:,0], points_cpu[:,1], s=0.1, c=np.linspace(0,1,N), cmap='viridis')
plt.title("Colored by Index")
plt.axis('off')

# Density-based coloring
plt.subplot(2,2,3)
plt.hexbin(points_cpu[:,0], points_cpu[:,1], gridsize=300, cmap='inferno', mincnt=1)
plt.title("Density Visualization")
plt.axis('off')

# Height-based log coloring
plt.subplot(2,2,4)
plt.scatter(points_cpu[:,0], points_cpu[:,1], s=0.1, c=np.log1p(np.abs(points_cpu[:,1])), cmap='plasma')
plt.title("Log-scale Height Coloring")
plt.axis('off')

plt.suptitle(f"Barnsley Fern - Estimated Fractal Dimension: {fd:.4f}", fontsize=16)
plt.tight_layout()
plt.show()
