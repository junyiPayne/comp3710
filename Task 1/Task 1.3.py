import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Grid for computing image
X, Y = np.mgrid[-4.0:4.0:0.01, -4.0:4.0:0.01]

# Load into PyTorch tensors
x = torch.tensor(X, dtype=torch.float32, device=device)
y = torch.tensor(Y, dtype=torch.float32, device=device)

# ----- Gaussian -----
sigma = 1.0
gaussian = torch.exp(-(x**2 + y**2) / (2 * sigma**2))

# ----- Sine -----
freq_x = 3.0
freq_y = 2.0
phase = 0.0
sine_pattern = torch.sin(freq_x * x + freq_y * y + phase)

# ----- Combined (Gabor-like) -----
combined = gaussian * sine_pattern

# ----- Plot -----
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Gaussian
axs[0].imshow(gaussian.cpu().numpy(), extent=[-4, 4, -4, 4],
              origin='lower', cmap='viridis', aspect='equal')
axs[0].set_title("2D Gaussian")
axs[0].axis('off')

# Sine
axs[1].imshow(sine_pattern.cpu().numpy(), extent=[-4, 4, -4, 4],
              origin='lower', cmap='viridis', aspect='equal')
axs[1].set_title("2D Sine")
axs[1].axis('off')

# Combined
axs[2].imshow(combined.cpu().numpy(), extent=[-4, 4, -4, 4],
              origin='lower', cmap='viridis', aspect='equal')
axs[2].set_title("Gaussian × Sine")
axs[2].axis('off')

plt.tight_layout()
plt.show()
