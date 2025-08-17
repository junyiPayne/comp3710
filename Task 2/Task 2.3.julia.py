import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def julia(width, height, c, x_min=-2.0, x_max=1.0, y_min=-1.3, y_max=1.3, max_iter=200):
    # Generate a coordinate grid and iterate Z_{n+1} = Z_n^2 + c at each grid point (Julia set)
    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    Z = X + 1j*Y
    C = torch.full_like(Z, complex(c.real, c.imag))

    counts = torch.zeros(Z.shape, dtype=torch.float32, device=device)
    for i in range(max_iter):
        Z = Z*Z + C
        mask = (counts == 0) & (torch.abs(Z) > 4.0)
        counts[mask] = i
    counts[counts == 0] = max_iter
    return counts.cpu().numpy()

# Compute with same resolution as your version
width = int((1 - (-2)) / 0.002)      # 600
height = int((1.3 - (-1.3)) / 0.002) # 520

start = time.perf_counter()

# Julia Set Parameters: Adjusting c to observe different forms
c = complex(-0.8, 0.156)
# Optional: Zoom in by passing x_min, x_max, y_min, y_max, e.g. x_min=-1.5, x_max=1.5,...
ns = julia(width, height, c, x_min=-2.0, x_max=1.0, y_min=-1.3, y_max=1.3, max_iter=200)

elapsed = time.perf_counter() - start
print(f"computation time: {elapsed:.2f} seconds")

# Same coloring function
def processFractal(a):
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20*np.cos(a_cyclic),
        30 + 50*np.sin(a_cyclic),
        155 - 80*np.cos(a_cyclic)
    ], axis=2)
    img[a == a.max()] = 0
    return np.uint8(np.clip(img, 0, 255))

# Plot
plt.figure(figsize=(16,10))
plt.imshow(processFractal(ns))
plt.tight_layout(pad=0)
plt.show()

