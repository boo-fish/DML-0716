import matplotlib

# 适配服务器无桌面环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os
import numpy as np
from scipy import stats
from datetime import datetime

# --------------- 全局配置（核心新增：固定绘制前50个epoch）---------------
GLOBAL_EPOCH = 50  # 所有方法只绘制前50个epoch
save_dir = r'D:\project\DML-0716\results-mnist-Final-tao'
os.makedirs(save_dir, exist_ok=True)

# --------------- 工具函数 ---------------
sns.set_style("whitegrid")
sns.set_palette("pastel")


# 转换张量为Python数值
def convert_to_python(value):
    return value.item() if hasattr(value, 'item') else value


# 读取同组的3个pkl文件（仅通过H值、是否IID、方法目录筛选）
def load_group_pkls(method_dir, h, is_iid):
    """
    读取同组的3个pkl文件（核心筛选条件）：
    - method_dir: 方法对应的文件夹名称（如'DML'/'benchmark'/'RAMFL'）
    - h: 节点数（5/10）
    - is_iid: 是否IID（True/False）
    返回：3次实验的准确率列表（每个列表是一轮轮的acc）
    """
    # 拼接方法目录路径
    full_dir = os.path.join('../results-mnist-Final-tao', method_dir)
    if not os.path.exists(full_dir):
        raise FileNotFoundError(f"方法目录不存在: {full_dir}")

    # 定义筛选关键词（适配不同命名格式）
    h_key = f'_{h}_'  # 匹配H=5/10的关键词（如'_5_'/'_10_'）
    iid_key = f'is_iid_{str(is_iid).lower()}' if 'is_iid' in os.listdir(full_dir)[0] else f'isIID_{is_iid}'

    # 筛选符合条件的pkl文件
    pkl_paths = []
    for fname in os.listdir(full_dir):
        if (fname.endswith('.pkl') and
                h_key in fname and
                (iid_key in fname or str(is_iid) in fname)):  # 兼容不同的IID命名格式
            pkl_paths.append(os.path.join(full_dir, fname))
            if len(pkl_paths) == 5:  # 只取前5个符合条件的pkl
                break

    # 校验pkl数量
    if len(pkl_paths) < 5:
        raise ValueError(
            f"方法目录[{method_dir}]下，H={h}、IID={is_iid}的pkl文件不足5个！\n"
            f"找到的文件：{[os.path.basename(p) for p in pkl_paths]}\n"
            f"目录路径：{full_dir}"
        )

    # 加载5个pkl的准确率数据（截断到前GLOBAL_EPOCH个epoch）
    acc_data_list = []
    for path in pkl_paths:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                # 读取并截断到前GLOBAL_EPOCH个epoch
                raw_accs = [convert_to_python(v) for v in data['global_round_accuracies'][0]]
                truncated_accs = raw_accs[:GLOBAL_EPOCH]  # 强制截断到前50个epoch

                # 可选：若数据不足50个epoch，填充NaN并提示
                if len(truncated_accs) < GLOBAL_EPOCH:
                    print(
                        f"⚠️ 警告：{os.path.basename(path)} 仅包含{len(raw_accs)}个epoch，不足{GLOBAL_EPOCH}个，剩余部分填充NaN")
                    truncated_accs += [np.nan] * (GLOBAL_EPOCH - len(truncated_accs))

                acc_data_list.append(truncated_accs)
        except Exception as e:
            raise RuntimeError(f"加载pkl文件失败: {path}\n错误信息: {str(e)}")

    return acc_data_list


# 计算均值和95%置信区间（适配前50个epoch）
def compute_mean_ci(data_list):
    """输入：5次实验的准确率列表（已截断到50个epoch）；输出：均值、置信区间下限、置信区间上限"""
    # 转换为数组（确保长度为GLOBAL_EPOCH）
    aligned_data = np.array(data_list)  # shape: (5, GLOBAL_EPOCH)

    # 计算均值和95%置信区间（基于t分布）
    mean = np.nanmean(aligned_data, axis=0)  # 忽略NaN值
    sem = stats.sem(aligned_data, axis=0, nan_policy='omit')  # 标准误（忽略NaN）
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(aligned_data) - 1)  # 95%置信区间

    # 处理NaN值（避免绘图报错）
    ci_lower = np.where(np.isnan(mean), np.nan, mean - ci)
    ci_upper = np.where(np.isnan(mean), np.nan, mean + ci)
    return mean, ci_lower, ci_upper


# --------------- 定义实验配置（先H=10，后H=5）---------------
method_configs = [
    # Proposed (DML) - 先H=10，后H=5（颜色调换）
    ('Proposed (H=10)', 'DML', '#1515ff', 10),
    ('Proposed (H=5)', 'DML', '#ff7f50', 5),
    # Conventional FL (benchmark) - 先H=10，后H=5（颜色调换）
    ('Conventional FL (H=10)', 'benchmark', '#ff0000', 10),
    ('Conventional FL (H=5)', 'benchmark', '#04bdfb', 5),
    # RAMFL - 先H=10，后H=5（颜色调换）
    ('RAMFL (H=10)', 'RAMFL', '#9370db', 10),
    ('RAMFL (H=5)', 'RAMFL', '#32cd32', 5),
]

# 数据分布配置：(是否IID, 子图标题)
data_distributions = [
    (True, '(a) MNIST dataset (IID data)'),
    (False, '(b) MNIST dataset (non-IID data)'),
]

# --------------- 绘制曲线与置信区间 ---------------
num_ais = 1  # 根据实际Ai数量调整
# 用于存储所有最终准确率的字典（方便汇总打印）
final_accuracy_summary = {}

for ai_idx in range(num_ais):
    # 创建1行2列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    axes = [ax1, ax2]

    # 遍历IID/non-IID两种分布
    for (is_iid, subplot_title), ax in zip(data_distributions, axes):
        # 生成前50个epoch的x轴刻度
        rounds = list(range(1, GLOBAL_EPOCH + 1))
        # 标记数据分布类型（IID/non-IID）
        dist_type = "IID" if is_iid else "non-IID"
        final_accuracy_summary[dist_type] = {}

        print(f"\n{'=' * 60}")
        print(f"📊 数据分布: {dist_type} (子图: {subplot_title})")
        print(f"{'=' * 60}")

        # 遍历所有方法（先H=10，后H=5）
        for (method_name, method_dir, color, h) in method_configs:
            # 加载同组5个pkl的准确率数据（已截断到50个epoch）
            acc_data_list = load_group_pkls(
                method_dir=method_dir,
                h=h,
                is_iid=is_iid
            )

            # 计算均值和置信区间
            mean_acc, ci_lower, ci_upper = compute_mean_ci(acc_data_list)

            # 获取最终准确率（第50个epoch）
            final_epoch_idx = GLOBAL_EPOCH - 1  # 索引从0开始
            final_mean = mean_acc[final_epoch_idx]
            final_ci_lower = ci_lower[final_epoch_idx]
            final_ci_upper = ci_upper[final_epoch_idx]

            # 存储到汇总字典
            final_accuracy_summary[dist_type][method_name] = {
                'mean': final_mean,
                'ci_lower': final_ci_lower,
                'ci_upper': final_ci_upper,
                'ci_range': final_ci_upper - final_ci_lower
            }

            # 打印当前方法的最终准确率（格式化输出）
            print(f"\n🔹 {method_name}:")
            print(f"   ├─ 第{GLOBAL_EPOCH}个epoch平均准确率: {final_mean:.4f} %")
            print(f"   ├─ 95%置信区间: [{final_ci_lower:.4f}, {final_ci_upper:.4f}] %")
            print(f"   └─ 置信区间宽度: {final_ci_upper - final_ci_lower:.4f} %")

            # 绘制均值曲线（仅前50个epoch）
            ax.plot(rounds, mean_acc, label=method_name, color=color, linewidth=1.5)
            # 绘制置信区间阴影（半透明）
            ax.fill_between(rounds, ci_lower, ci_upper, color=color, alpha=0.1)

        # 子图样式配置（固定x轴范围为0~51）
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_ylim(10, 100)  # 准确率范围
        ax.set_xlim(0, GLOBAL_EPOCH + 1)  # 固定x轴为0~51（适配前50个epoch）
        # 图例顺序与绘制顺序一致（先H=10后H=5）
        ax.legend(fontsize=10, loc="lower right", bbox_to_anchor=(1, 0))
        # 子图标题（底部）
        ax.text(0.5, -0.15, subplot_title,
                fontsize=14, ha='center', va='top', transform=ax.transAxes)
        # 美化边框
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        ax.tick_params(axis='both', width=2, length=6)

    # 布局调整与保存
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'2d-mnist_epoch_acc_with_CI_{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图片已保存至: {save_path}")

# 打印汇总表格（更清晰的对比）
print(f"\n{'=' * 80}")
print(f"📈 所有方法最终准确率汇总表（第{GLOBAL_EPOCH}个epoch）")
print(f"{'=' * 80}")

# 打印表头
print(f"{'数据分布':<12} {'方法名称':<25} {'平均准确率(%)':<15} {'95%置信区间(%)':<30}")
print(f"{'-' * 12:<12} {'-' * 25:<25} {'-' * 15:<15} {'-' * 30:<30}")

# 打印每个分布下的所有方法
for dist_type in ['IID', 'non-IID']:
    for method_name, acc_info in final_accuracy_summary[dist_type].items():
        ci_str = f"[{acc_info['ci_lower']:.4f}, {acc_info['ci_upper']:.4f}]"
        # 最终正确的打印语句
        print(f"{dist_type:<12} {method_name:<25} {acc_info['mean']:<15.4f} {ci_str:<30}")
print(f"{'=' * 80}")