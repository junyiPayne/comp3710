import numpy as np
import matplotlib.pyplot as plt
import torch

# NumPy 2D Gaussian
def gaussian_2d_np(x, y, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0, rho=0.0):
    coef = 1.0 / (2.0 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    x_std = (x - mu_x) / sigma_x
    y_std = (y - mu_y) / sigma_y
    exponent = -1.0 / (2.0 * (1 - rho**2)) * (x_std**2 - 2.0 * rho * x_std * y_std + y_std**2)
    return coef * np.exp(exponent)

# PyTorch 2D Gaussian (uses tensor ops)
def gaussian_2d_torch(x, y, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0, rho=0.0):
    dtype = x.dtype
    device = x.device
    mu_x_t = torch.tensor(mu_x, dtype=dtype, device=device)
    mu_y_t = torch.tensor(mu_y, dtype=dtype, device=device)
    sigma_x_t = torch.tensor(sigma_x, dtype=dtype, device=device)
    sigma_y_t = torch.tensor(sigma_y, dtype=dtype, device=device)
    rho_t = torch.tensor(rho, dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)

    norm = 1.0 / (2.0 * torch.pi * sigma_x_t * sigma_y_t * torch.sqrt(one - rho_t**2))
    x_std = (x - mu_x_t) / sigma_x_t
    y_std = (y - mu_y_t) / sigma_y_t
    exponent = -1.0 / (2.0 * (one - rho_t**2)) * (x_std**2 - 2.0 * rho_t * x_std * y_std + y_std**2)
    return norm * torch.exp(exponent)

def main():
    # grid
    n = 200
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X_np, Y_np = np.meshgrid(x, y, indexing='xy')

    # NumPy gaussian
    Z_np = gaussian_2d_np(X_np, Y_np, sigma_x=1.0, sigma_y=1.0, rho=0.0)

    # Torch grid (use float64 to match NumPy precision)
    xt = torch.from_numpy(x).to(dtype=torch.float64)
    yt = torch.from_numpy(y).to(dtype=torch.float64)
    X_t, Y_t = torch.meshgrid(xt, yt, indexing='xy')

    Z_t = gaussian_2d_torch(X_t, Y_t, sigma_x=1.0, sigma_y=1.0, rho=0.0)
    Z_t_np = Z_t.cpu().numpy()

    # Sine pattern (PyTorch) and product
    sine_t = torch.sin(3.0 * X_t + 3.0 * Y_t)
    product_t = Z_t * sine_t
    product_np = product_t.cpu().numpy()
    sine_np = sine_t.cpu().numpy()

    # Plot 1x3 (with colorbars)
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))

    im0 = axs[0].contourf(X_np, Y_np, Z_np, levels=50, cmap='viridis')
    axs[0].set_title('NumPy Gaussian')
    axs[0].set_aspect('equal')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].contourf(X_np, Y_np, Z_t_np, levels=50, cmap='viridis')
    axs[1].set_title('PyTorch Gaussian')
    axs[1].set_aspect('equal')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].contourf(X_np, Y_np, product_np, levels=50, cmap='RdBu')
    axs[2].set_title('PyTorch Gaussian × Sine')
    axs[2].set_aspect('equal')
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].contourf(X_np, Y_np, sine_np, levels=50, cmap='twilight_shifted')
    axs[3].set_title('PyTorch Sine Wave')
    axs[3].set_aspect('equal')
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
