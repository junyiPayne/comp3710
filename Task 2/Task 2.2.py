import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

def mandelbrot(width, height, max_iter=200):
    # Match your grid: X in [-2,1], Y in [-1.3,1.3]
    x = torch.linspace(-2.0, 1.0, width, device=device)
    y = torch.linspace(-1.3, 1.3, height, device=device)
    Y, X = torch.meshgrid(y, x, indexing="ij")
    C = X + 1j*Y

    Z = torch.zeros_like(C)
    counts = torch.zeros(C.shape, dtype=torch.float32, device=device)

    for i in range(max_iter):
        Z = Z*Z + C
        mask = (counts == 0) & (torch.abs(Z) > 4.0)
        counts[mask] = i
    counts[counts == 0] = max_iter
    return counts.cpu().numpy()

# Compute with same resolution as your version
width = int((1 - (-2)) / 0.005)      # 600
height = int((1.3 - (-1.3)) / 0.005) # 520

start = time.perf_counter()

ns = mandelbrot(width, height, max_iter=200)

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

