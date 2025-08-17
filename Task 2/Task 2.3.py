import torch
import time
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



def mandelbrot(width, height, x_min=-2.0, x_max=1.0, y_min=-1.3, y_max=1.3, max_iter=200):
    # Match your grid: X in [-2,1], Y in [-1.3,1.3]
    x = torch.linspace(x_min, x_max, width, device=device)
    y = torch.linspace(y_min, y_max, height, device=device)
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
width = int((1 - (-2)) / 0.002)      #600
height = int((1.3 - (-1.3)) / 0.002) # 520

# Set the zoom area (try different magnifications by modifying the following three values)
cx, cy = -0.75, 0.1        # Zoom center (real, imaginary)
half_width = 0.02          # Half-width in x direction (decrease to zoom in)
half_height = half_width * (height / width)

x_min, x_max = cx - half_width, cx + half_width
y_min, y_max = cy - half_height, cy + half_height

start = time.perf_counter()

# When calling, pass in the range and the required max_iter (for example, increase the number of iterations when magnifying)
ns = mandelbrot(width, height, x_min, x_max, y_min, y_max, max_iter=200)

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

