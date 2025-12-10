import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'QtAgg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from datetime import datetime

# 创建保存图片的目录
save_dir = 'saving/3d_acc'
os.makedirs(save_dir, exist_ok=True)

# 原始数据
p = np.array([10, 5, 3, 10, 5, 3, 10, 5, 3, 10, 5, 3])
alpha = np.array([0.3]*3 + [0.6]*3 + [0.8]*3 + [1.0]*3)
accuracy = np.array([
    93.8799, 91.5199, 88.9499,
    94.3499, 93.4899, 91.33,
    94.3399, 93.2099, 91.69,
    94.58,   93.8899, 92.23
])

# 构建插值网格
alpha_unique = np.linspace(0.3, 1.0, 10)
p_unique = np.linspace(3, 10, 10)
P, A = np.meshgrid(p_unique, alpha_unique)

Z = griddata((p, alpha), accuracy, (P, A), method='cubic')

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 表面图，使用 plasma 色图
surf = ax.plot_surface(P, A, Z, cmap='plasma', edgecolor='none', linewidth=0, antialiased=True)

# 添加颜色条
cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Accuracy[%]')

# 黑点：提高 zorder，关闭 depthshade，避免遮挡
ax.scatter(p, alpha, accuracy, c='black', s=30, depthshade=False, zorder=10)


# 设置坐标轴标签
ax.set_xlabel('Heterogeneity')
ax.set_ylabel('non-IID Degree')

ax.set_zlim(85, 96)  # 手动设置 Z 轴显示范围

# **反转 x 和 y 轴方向**
ax.set_xlim(ax.get_xlim()[::-1])  # 反转 x 轴 (Diameter)
ax.set_ylim(ax.get_ylim()[::-1])  # 反转 y 轴 (non-IID Degree)

# 设置视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(save_dir, f'3d_acc_{current_time}.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {save_path}")

plt.show()
