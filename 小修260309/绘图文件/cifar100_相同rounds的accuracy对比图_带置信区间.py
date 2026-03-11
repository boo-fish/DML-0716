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
GLOBAL_EPOCH = 100  # 所有方法只绘制前50个epoch
save_dir = r'./小修实验图/'
os.makedirs(save_dir, exist_ok=True)

# --------------- 工具函数 ---------------
sns.set_style("whitegrid")
sns.set_palette("pastel")


# 打印分隔线的辅助函数
def print_separator(title=""):
    """打印带标题的分隔线，增强日志可读性"""
    print("\n" + "=" * 80)
    if title:
        print(f"📌 {title}")
        print("=" * 80)


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
    print_separator(f"开始加载数据 | 方法: {method_dir} | 节点数H: {h} | IID: {is_iid}")

    # 拼接方法目录路径
    full_dir = os.path.join('../绘图所需的数据/相同轮数下的准确率对比数据/', method_dir)
    print(f"🔍 查找目录: {full_dir}")

    if not os.path.exists(full_dir):
        raise FileNotFoundError(f"方法目录不存在: {full_dir}")

    # 定义筛选关键词（适配不同命名格式）
    h_key = f'_{h}_'  # 匹配H=5/10的关键词（如'_5_'/'_10_'）
    iid_key = f'is_iid_{str(is_iid).lower()}' if 'is_iid' in os.listdir(full_dir)[0] else f'isIID_{is_iid}'
    print(f"🔤 筛选关键词 | H关键词: {h_key} | IID关键词: {iid_key}")

    # 筛选符合条件的pkl文件
    pkl_paths = []
    for fname in os.listdir(full_dir):
        if (fname.endswith('.pkl') and
                h_key in fname and
                (iid_key in fname or str(is_iid) in fname)):  # 兼容不同的IID命名格式
            pkl_paths.append(os.path.join(full_dir, fname))

    # 打印找到的文件信息
    print(f"📂 找到符合条件的PKL文件数量: {len(pkl_paths)}")
    if pkl_paths:
        print(f"📄 找到的文件列表:")
        for i, path in enumerate(pkl_paths, 1):
            print(f"   {i}. {os.path.basename(path)}")

    # 校验pkl数量
    if len(pkl_paths) < 5:
        raise ValueError(
            f"方法目录[{method_dir}]下，H={h}、IID={is_iid}的pkl文件不足5个！\n"
            f"找到的文件：{[os.path.basename(p) for p in pkl_paths]}\n"
            f"目录路径：{full_dir}"
        )

    # 加载3个pkl的准确率数据（截断到前GLOBAL_EPOCH个epoch）
    acc_data_list = []
    for idx, path in enumerate(pkl_paths, 1):
        try:
            print(f"\n📖 正在加载第{idx}/{len(pkl_paths)}个文件: {os.path.basename(path)}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
                # 读取并截断到前GLOBAL_EPOCH个epoch
                raw_accs = [convert_to_python(v) for v in data['global_round_accuracies'][0]]
                print(f"   📊 原始数据长度（epoch数）: {len(raw_accs)}")

                truncated_accs = raw_accs[:GLOBAL_EPOCH]  # 强制截断到前50个epoch
                print(f"   ✂️  截断后数据长度: {len(truncated_accs)} (前{GLOBAL_EPOCH}个epoch)")

                # 可选：若数据不足50个epoch，填充NaN并提示
                if len(truncated_accs) < GLOBAL_EPOCH:
                    missing = GLOBAL_EPOCH - len(truncated_accs)
                    print(f"   ⚠️  警告：文件仅包含{len(raw_accs)}个epoch，不足{GLOBAL_EPOCH}个，将填充{missing}个NaN值")
                    truncated_accs += [np.nan] * missing

                acc_data_list.append(truncated_accs)
                # 打印关键统计信息
                valid_accs = [x for x in truncated_accs if not np.isnan(x)]
                if valid_accs:
                    print(
                        f"   📈 数据统计 | 最小值: {min(valid_accs):.4f} | 最大值: {max(valid_accs):.4f} | 平均值: {np.mean(valid_accs):.4f}")

        except Exception as e:
            raise RuntimeError(f"加载pkl文件失败: {path}\n错误信息: {str(e)}")

    print(
        f"\n✅ 数据加载完成 | 共加载{len(acc_data_list)}个文件 | 每个文件数据长度: {len(acc_data_list[0]) if acc_data_list else 0}")
    return acc_data_list


# 计算均值和95%置信区间（适配前50个epoch）
def compute_mean_ci(data_list):
    """输入：3次实验的准确率列表（已截断到50个epoch）；输出：均值、置信区间下限、置信区间上限"""
    print_separator("计算均值和95%置信区间")

    # 转换为数组（确保长度为GLOBAL_EPOCH）
    aligned_data = np.array(data_list)  # shape: (3, GLOBAL_EPOCH)
    print(f"📐 输入数据形状: {aligned_data.shape} (实验次数 × Epoch数)")

    # 计算均值和95%置信区间（基于t分布）
    mean = np.nanmean(aligned_data, axis=0)  # 忽略NaN值
    sem = stats.sem(aligned_data, axis=0, nan_policy='omit')  # 标准误（忽略NaN）
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(data_list) - 1)  # 95%置信区间

    # 处理NaN值（避免绘图报错）
    ci_lower = np.where(np.isnan(mean), np.nan, mean - ci)
    ci_upper = np.where(np.isnan(mean), np.nan, mean + ci)

    # 打印统计摘要
    valid_mean = mean[~np.isnan(mean)]
    print(f"📊 均值统计 | 有效数据点: {len(valid_mean)} | 均值范围: {np.min(valid_mean):.4f} ~ {np.max(valid_mean):.4f}")
    print(f"🎯 置信区间 | 平均区间宽度: {np.nanmean(ci) * 2:.4f} (95%置信水平)")

    return mean, ci_lower, ci_upper


# --------------- 主程序开始 ---------------
if __name__ == "__main__":
    print_separator("程序启动 | 开始绘制CIFAR100实验曲线图")
    print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  全局配置 | 绘制Epoch数: {GLOBAL_EPOCH} | 保存目录: {save_dir}")

    method_configs = [
        # Proposed (DML) - 先H=10，后H=5（颜色调换）
        ('Proposed (H=10)', 'DML_cifar100', '#1515ff', 10),
        ('Proposed (H=5)', 'DML_cifar100', '#ff7f50', 5),
        # Conventional FL (benchmark) - 先H=10，后H=5（颜色调换）
        ('Conventional FL (H=10)', 'Bench_cifar100', '#ff0000', 10),
        ('Conventional FL (H=5)', 'Bench_cifar100', '#04bdfb', 5),
        # RAMFL - 先H=10，后H=5（颜色调换）
        ('RAMFL (H=10)', 'RAMFL_cifar100', '#9370db', 10),
        ('RAMFL (H=5)', 'RAMFL_cifar100', '#32cd32', 5),
    ]

    # 数据分布配置：(是否IID, 子图标题)
    data_distributions = [
        (True, '(a) CIFAR-100 dataset (IID data)'),
        (False, '(b) CIFAR-100 dataset (non-IID data)'),
    ]

    # 打印方法配置信息
    print_separator("实验方法配置")
    print(f"🔧 共配置{len(method_configs)}个方法:")
    for i, (name, dir_, color, h) in enumerate(method_configs, 1):
        print(f"   {i}. {name} | 目录: {dir_} | 颜色: {color} | H: {h}")

    # --------------- 绘制曲线与置信区间 ---------------
    num_ais = 1  # 根据实际Ai数量调整
    print_separator(f"开始绘制图表 | AI数量: {num_ais}")

    for ai_idx in range(num_ais):
        print(f"\n🎨 正在绘制第{ai_idx + 1}/{num_ais}组图表")
        # 创建1行2列的子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        axes = [ax1, ax2]

        # 遍历IID/non-IID两种分布
        for (is_iid, subplot_title), ax in zip(data_distributions, axes):
            print_separator(f"绘制子图 | 数据分布: {'IID' if is_iid else 'non-IID'} | 标题: {subplot_title}")

            # 生成前50个epoch的x轴刻度
            rounds = list(range(1, GLOBAL_EPOCH + 1))

            # 遍历所有方法（先H=10，后H=5）
            for idx, (method_name, method_dir, color, h) in enumerate(method_configs, 1):
                print(f"\n🔹 处理第{idx}/{len(method_configs)}个方法: {method_name}")

                # 加载同组3个pkl的准确率数据（已截断到50个epoch）
                acc_data_list = load_group_pkls(
                    method_dir=method_dir,
                    h=h,
                    is_iid=is_iid
                )

                # 计算均值和置信区间
                mean_acc, ci_lower, ci_upper = compute_mean_ci(acc_data_list)

                # 绘制均值曲线（仅前50个epoch）
                ax.plot(rounds, mean_acc, label=method_name, color=color, linewidth=1.5)
                # 绘制置信区间阴影（半透明）
                ax.fill_between(rounds, ci_lower, ci_upper, color=color, alpha=0.1)

                print(f"✅ 已绘制 {method_name} 曲线")

            # 子图样式配置（固定x轴范围为0~51）
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_ylim(0, 60)  # 准确率范围
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
    print_separator("保存图表文件")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'2d-CIFAR100_epoch_acc_with_CI_{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {save_path}")
    print(f"📊 文件信息 | DPI: 300 | 格式: PDF | 保存目录: {save_dir}")

    print_separator("程序执行完成")
    print(f"🎉 所有图表绘制完成！")
    print(f"📁 结果文件保存在: {save_dir}")
    print(f"🕒 程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")