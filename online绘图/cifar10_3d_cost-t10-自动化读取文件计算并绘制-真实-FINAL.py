import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import pickle
import glob
from datetime import datetime

# ========== 配置 ==========
T = 10                      # OnlineFL周期
TARGET_ACC = 55.0           # 目标准确率(%)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
pkl_dir = os.path.join(BASE_DIR, 'results', f'online_{T}')
cost_dir = os.path.join(BASE_DIR, '绘图文件', '每轮OnlineFL成本')
save_dir = os.path.join(BASE_DIR, '绘图文件', '3d-cost')
os.makedirs(save_dir, exist_ok=True)

# ========== 12组实验参数 ==========
# 顺序: p=20,alpha=0.3/0.6/0.8/1.0; p=10,alpha=0.3/0.6/0.8/1.0; p=5,alpha=0.3/0.6/0.8/1.0
p_list = [20, 10, 5]
alpha_list = [0.3, 0.6, 0.8, 1.0]

# 构建12组参数: [(p, alpha), ...]
params = []
for p_val in p_list:
    for alpha_val in alpha_list:
        params.append((p_val, alpha_val))

print("=" * 70)
print(f"  目标准确率: {TARGET_ACC}%  |  OnlineFL周期 T={T}")
print("=" * 70)



# --------------------- 核心修复：强制映射 numpy 模块 ---------------------
# 修复 numpy 2.0 → numpy 1.x 的模块不兼容问题
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 把新版本的 numpy._core 全部映射到旧版能用的路径
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            # 兜底：所有找不到的 numpy 内部模块都自动修正
            if 'numpy' in module:
                module = module.replace('_core', 'core')
            return super().find_class(module, name)




# ========== Step1: 从pkl文件获取到达目标准确率所需的epoch数 ==========
p_values = []
alpha_values = []
total_costs = []
epoch_to_target = []

for p_val, alpha_val in params:
    # 查找匹配的pkl文件
    pattern = os.path.join(pkl_dir, f'cifar10_vgg11_NOIID_P_{p_val}_Alpha_{alpha_val}_*.pkl')
    matches = glob.glob(pattern)
    
    if not matches:
        print(f"\n[警告] 未找到 pkl 文件: p={p_val}, alpha={alpha_val}")
        print(f"  搜索路径: {pattern}")
        continue
    
    pkl_path = matches[0]
    print(f"\n[Step1] p={p_val}, alpha={alpha_val}")
    print(f"  pkl文件: {os.path.basename(pkl_path)}")
    
    
    
    
    
    
    # with open(pkl_path, 'rb') as f:
    #     data = pickle.load(f)
        
    # --------------------- 正常加载你的文件 ---------------------
    with open(pkl_path, "rb") as f:
        data = CustomUnpickler(f).load()

    
    round_accs = data['global_round_accuracies'][0]
    total_rounds = len(round_accs)
    print(f"  总轮次: {total_rounds}, 最终准确率: {round_accs[-1]:.2f}%")
    
    # 找到第一个达到目标准确率的epoch (索引从0开始, epoch从1开始)
    target_epoch = None
    for i, acc in enumerate(round_accs):
        if acc >= TARGET_ACC:
            target_epoch = i + 1  # epoch编号从1开始
            break
    
    if target_epoch is None:
        print(f"  [未达标] {total_rounds}轮内未达到{TARGET_ACC}%，使用全部{total_rounds}轮")
        target_epoch = total_rounds
    else:
        print(f"  [达标] 第 {target_epoch} 轮首次达到 {TARGET_ACC}% (实际: {round_accs[target_epoch-1]:.2f}%)")
    
    
    # ========== Step2: 从csv/npy中读取前x个时隙的成本并累加 ==========
    cost_file = os.path.join(cost_dir, f'cost_T{T}_p{p_val}.csv')
    
    if not os.path.exists(cost_file):
        print(f"  [警告] 成本文件不存在: {cost_file}")
        continue
    
    # 读取csv: [time_slot, worker_cost, transmission_cost, total_explicit_cost]
    cost_data = np.loadtxt(cost_file, delimiter=',', skiprows=1)
    total_slots_available = len(cost_data)
    
    # 取前 target_epoch 个时隙的 total_explicit_cost (第4列, 索引3)
    slots_to_use = min(target_epoch, total_slots_available)
    total_cost = np.sum(cost_data[:slots_to_use, 3])  # total_explicit_cost列
    
    # 分别累加worker和transmission成本（用于调试）
    worker_cost_sum = np.sum(cost_data[:slots_to_use, 1])
    trans_cost_sum = np.sum(cost_data[:slots_to_use, 2])
    
    print(f"  成本文件: cost_T{T}_p{p_val}.csv (共{total_slots_available}行)")
    print(f"  使用前 {slots_to_use} 个时隙的成本累加:")
    print(f"    Worker成本累加: {worker_cost_sum:.4f}")
    print(f"    传输成本累加:   {trans_cost_sum:.4f}")
    print(f"    总成本:         {total_cost:.4f}")
    
    p_values.append(p_val)
    alpha_values.append(alpha_val)
    total_costs.append(round(total_cost, 2))
    epoch_to_target.append(target_epoch)

# ========== 汇总 ==========
print("\n" + "=" * 70)
print("  汇总: 12组参数的总成本")
print("=" * 70)
print(f"{'p':<6} {'alpha':<8} {'达标epoch':<10} {'总cost':<14}")
print("-" * 40)
for i in range(len(p_values)):
    print(f"{p_values[i]:<6} {alpha_values[i]:<8} {epoch_to_target[i]:<10} {total_costs[i]:<14.2f}")

# ========== Step3: 绘制3D图 ==========
p = np.array(p_values)
alpha = np.array(alpha_values)
cost = np.array(total_costs)

print(f"\n[绘图数据]")
print(f"  p = {p.tolist()}")
print(f"  alpha = {alpha.tolist()}")
print(f"  cost = {cost.tolist()}")

# 构建插值网格
alpha_unique = np.linspace(0.3, 1.0, 10)
p_unique = np.linspace(5, 20, 10)
P, A = np.meshgrid(p_unique, alpha_unique)
Z = griddata((p, alpha), cost, (P, A), method='cubic')

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 禁用自动深度排序，让 zorder 手动控制绘制层级
ax.computed_zorder = False

# 表面图
surf = ax.plot_surface(P, A, Z, cmap='plasma', edgecolor='none', linewidth=0, antialiased=True, zorder=1)

# 颜色条
cbar = fig.colorbar(surf, shrink=0.6, aspect=15, pad=0.1)
cbar.set_label('Cost')

# 黑点
ax.scatter(p, alpha, cost, c='black', s=30, depthshade=False, zorder=10)

# 坐标轴标签
ax.set_xlabel('Heterogeneity')
ax.set_ylabel('non-IID Degree')

# **反转 x 和 y 轴方向**
ax.set_xlim(ax.get_xlim()[::-1])  # 反转 x 轴 (Diameter)
ax.set_ylim(ax.get_ylim()[::-1])  # 反转 y 轴 (non-IID Degree)

# 设置视角
ax.view_init(elev=30, azim=45)

plt.tight_layout()

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(save_dir, f'3d_cost_T_{T}_{current_time}.pdf')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存至: {save_path}")

plt.show()
