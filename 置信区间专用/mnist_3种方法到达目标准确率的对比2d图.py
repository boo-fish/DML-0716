import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.options import args_parser
import seaborn as sns
import pickle
import pandas as pd
import os
from datetime import datetime

# ===================== 基础配置 =====================
# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 解析参数
args = args_parser()
p_value = args.p
ai = args.ai
lr = args.lr
epoch_value = args.epochs

# 核心配置
TARGET_ACCURACY = 92  # 目标准确率（保留原代码目标值）
SAVE_DIR = r'D:\project\DML-0716\大修实验图\MNIST_到达目标准确率的时间对比图'  # 图片保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 自定义颜色（对应3个模型，新增颜色适配RAMFL模型）
custom_colors = [
    '#1515ff',
    '#ff7f50',
    '#04bdfb',
    '#32cd32'  # 新增绿色，用于RAMFL模型
]

# 图表尺寸配置（和参考代码保持一致）
FIG_SIZE = (16, 8)  # 一行两列的整体尺寸
SUBPLOT_WIDTH_RATIO = 1
SUBPLOT_HEIGHT_RATIO = 1

# 数据文件路径（新增ramfl_iid和ramfl_noniid）
DATA_PATHS = {
    'proposed_iid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_True_DML_alpha_0.3.pkl',
    'proposed_noniid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_False_DML_alpha_0.3.pkl',
    'conventional_iid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_True_local_alpha_0.3.pkl',
    'conventional_noniid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_False_local_alpha_0.3.pkl',
    'ramfl_iid': r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_60_is_iid_True_local_alpha_0.3_Final_Acc_92.4300_2025-12-15-12-23-41.pkl',
    'ramfl_noniid': r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_155_is_iid_False_local_alpha_0.3_Final_Acc_92.4600_2025-12-16-01-13-51.pkl'
}

# ===================== 工具函数 =====================
def convert_to_python(value):
    """张量转标量函数"""
    if hasattr(value, 'item'):
        return value.item()
    return value

def calculate_cumulative_time(times):
    """将每轮时间转换为累计时间"""
    cumulative = []
    total = 0
    for time in times:
        total += convert_to_python(time)
        cumulative.append(total)
    return cumulative

def truncate_to_target(times, accuracies, target):
    """截断数据，只保留达到目标准确率之前（包括达到时）的数据点"""
    truncated_times = []
    truncated_accs = []

    for t, acc in zip(times, accuracies):
        truncated_times.append(t)
        truncated_accs.append(acc)

        # 如果达到或超过目标准确率，停止添加更多数据点
        if acc >= target:
            break

    return truncated_times, truncated_accs

def load_single_pkl_file(file_path):
    """加载单个pkl文件，返回数据或None（异常处理）"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"警告：读取文件 {file_path} 失败 - {e}")
        return None

# ===================== 加载数据 =====================
# 加载所有数据文件（包含新增的ramfl数据）
data_dict = {
    key: load_single_pkl_file(path) for key, path in DATA_PATHS.items()
}

# 校验数据是否加载成功
for key, data in data_dict.items():
    if data is None:
        print(f"❌ 关键错误：{key} 数据加载失败，无法继续绘图！")
        exit(1)

# 确定AI数量（更新为包含ramfl数据的最小AI数）
num_ais = min(
    len(data_dict['proposed_iid']['ais']),
    len(data_dict['proposed_noniid']['ais']),
    len(data_dict['conventional_iid']['ais']),
    len(data_dict['conventional_noniid']['ais']),
    len(data_dict['ramfl_iid']['ais']),  # 新增校验ramfl_iid的AI数量
    len(data_dict['ramfl_noniid']['ais'])  # 新增校验ramfl_noniid的AI数量
)
if num_ais == 0:
    print("❌ 未找到有效AI数据，无法绘图！")
    exit(1)

# ===================== 绘制组合图表 =====================
def plot_combined_chart(ai_idx):
    """
    绘制一行两列子图：
    左图：IID数据对比（Proposed IID vs Conventional FL IID vs RAMFL IID）
    右图：non-IID数据对比（Proposed non-IID vs Conventional FL non-IID vs RAMFL non-IID）
    """
    # 创建一行两列的子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)


    # 定义子图配置（左：IID，右：non-IID，新增RAMFL模型）
    subplot_configs = [
        {
            'ax': ax1,
            'plot_type': 'iid',
            'title': '(a) MNIST dataset (IID data)',
            'models': [
                ('proposed_iid', 'Proposed (IID)'),
                ('conventional_iid', 'Conventional FL  (IID)'),
                ('ramfl_iid', 'RAMFL (IID)')  # 新增RAMFL IID模型
            ]
        },
        {
            'ax': ax2,
            'plot_type': 'noniid',
            'title': '(b) MNIST dataset (non-IID data)',
            'models': [
                ('proposed_noniid', 'Proposed (non-IID)'),
                ('conventional_noniid', 'Conventional FL  (non-IID)'),
                ('ramfl_noniid', 'RAMFL (non-IID)')  # 新增RAMFL non-IID模型
            ]
        }
    ]

    # 遍历绘制每个子图
    for config in subplot_configs:
        ax = config['ax']
        plot_type = config['plot_type']
        title = config['title']
        models = config['models']

        # 准备当前子图的数据
        plot_data = []
        method_names = []
        color_index = 0  # 对应自定义颜色索引

        for data_key, model_name in models:
            data = data_dict[data_key]
            # 计算累计时间
            times = calculate_cumulative_time(data['global_round_times'][ai_idx])
            # 提取并转换准确率
            accs = [convert_to_python(v) for v in data['global_round_accuracies'][ai_idx]]
            # 截断数据到目标准确率
            truncated_times, truncated_accs = truncate_to_target(times, accs, TARGET_ACCURACY)

            if truncated_times:
                plot_data.append({
                    'Time': truncated_times,
                    'Accuracy': truncated_accs,
                    'Model': model_name
                })
                method_names.append(model_name)
                color_index += 1

        # 处理无数据的情况
        if not plot_data:
            print(f"警告：AI {ai_idx} 的 {plot_type} 数据为空，该子图将为空！")
            # 统一子图基础样式
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_ylim(10, 100)
            ax.set_xlim(0, 1)
        else:
            # 构建DataFrame
            df_list = []
            for item in plot_data:
                df = pd.DataFrame({
                    'Time': item['Time'],
                    'Accuracy': item['Accuracy'],
                    'Model': item['Model']
                })
                df_list.append(df)
            data = pd.concat(df_list, ignore_index=True)

            # 绘制折线图
            sns.lineplot(
                data=data,
                x='Time',
                y='Accuracy',
                hue='Model',
                style='Model',
                markers=False,
                dashes=False,
                linewidth=1.5,
                palette=custom_colors[:len(method_names)],
                ax=ax
            )

            # 添加目标准确率虚线
            max_time = max(data['Time']) if not data.empty else 0
            ax.axhline(
                y=TARGET_ACCURACY,
                color='gray',
                linestyle='--',
                linewidth=1.5,
                label=f'Target Accuracy ({TARGET_ACCURACY}%)'
            )

            # 设置x轴范围
            ax.set_xlim(0, max_time * 1.05 if max_time > 0 else 1)

        # 统一子图样式（和参考代码完全对齐）
        # 设置坐标轴标签
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        # 设置y轴范围（保留原代码的10-100）
        ax.set_ylim(10, 100)
        # 边框加粗设置
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        # 刻度样式
        ax.tick_params(axis='both', width=2, length=6)
        # 设置图例
        ax.legend(fontsize=10, title_fontsize=11, loc="lower right", frameon=True, framealpha=0.9)
        # 设置子图标题（和参考代码对齐，y轴向下偏移）
        ax.set_title(title, fontsize=14, y=-0.15)

    # 统一布局调整（和参考代码保持一致）
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, wspace=0.25)

    # 保存图片
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        SAVE_DIR,
        f'con-time_Ai_{ai_idx}_lr_{lr}_{current_time}.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ AI {ai_idx} 的组合图已保存至: {save_path}")
    plt.show()

# ===================== 主逻辑：遍历AI并绘图 =====================
print(f"📌 开始绘制 {num_ais} 个AI的一行两列组合图表...")
for ai_idx in range(num_ais):
    plot_combined_chart(ai_idx)