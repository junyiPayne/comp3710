import torch
import numpy as np
#plot
import matplotlib.pyplot as plt


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = torch.exp(-(x**2+y**2)/2.0)
print("Computed Gaussian shape:", z.shape)

plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.show()

# # 生成网格
# X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# # 计算二维高斯分布
# Z = np.exp(-(X**2 + Y**2) / 2.0)

# # 绘图
# plt.imshow(Z, extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
# # plt.colorbar(label='Density')
# plt.title('2D Gaussian Function')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.tight_layout()
# plt.show()


# # 生成网格
# X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
# x = torch.tensor(X, dtype=torch.float32, device=device)
# y = torch.tensor(Y, dtype=torch.float32, device=device)

# # 生成二维正弦函数，角度依赖于 x 和 y
# # 例如：sin(sqrt(x^2 + y^2))，你也可以换成 sin(x) 或 sin(x+y) 等
# z = torch.sin(torch.sqrt(x**2 + y**2))

# # 可视化
# plt.imshow(z.cpu().numpy(), extent=[-4, 4, -4, 4], origin='lower', cmap='viridis')
# plt.colorbar(label='Value')
# plt.title('2D Sine Function: sin(sqrt(x^2 + y^2))')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.tight_layout()
# plt.show()

# # 生成条纹状二维正弦函数
# k = 5.0  # 控制条纹密度
# l = 3.0  # 控制条纹方向
# z_stripes = torch.sin(k * x + l * y)

# # 可视化条纹
# plt.figure()
# plt.imshow(z_stripes.cpu().numpy(), extent=[-4, 4, -4, 4], origin='lower', cmap='plasma')
# plt.colorbar(label='Value')
# plt.title('2D Sine Stripes: sin(5x + 3y)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.tight_layout()
# plt.show()

# Use NumPy to create a 2D array of complex numbers on [-2,2]x[-2,2]
# Y, X = np.mgrid[-1.3:1.3:0.005,
# -2:1:0.005]
# # load into PyTorch tensors
# x = torch.Tensor(X)
# y = torch.Tensor(Y)
# z = torch.complex(x, y) #important!
# zs = z.clone() #Updated!
# ns = torch.zeros_like(z)
# # transfer to the GPU device
# z = z.to(device)
# zs = zs.to(device)
# ns = ns.to(device)
# #Mandelbrot Set
# for i in range(200):
# #Compute the new values of z: z^2 + x
#     zs_ = zs*zs + z
#     #Have we diverged with this new value?
#     not_diverged = torch.abs(zs_) < 4.0
#     #Update variables to compute
#     ns += not_diverged
#     zs = zs_
# #plot
# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(16,10))
# def processFractal(a):
#     """Display an array of iteration counts as a
#     colorful picture of a fractal."""
#     a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
#     img = np.concatenate([10+20*np.cos(a_cyclic),
#     30+50*np.sin(a_cyclic),
#     155-80*np.cos(a_cyclic)], 2)
#     img[a==a.max()] = 0
#     a = img
#     a = np.uint8(np.clip(a, 0, 255))
#     return a
# plt.imshow(processFractal(ns.cpu().numpy()))
# plt.tight_layout(pad=0)
# plt.show()