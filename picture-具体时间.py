import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils.options import args_parser
import seaborn as sns
import pickle
import pandas as pd
import os
from datetime import datetime

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("pastel")

args = args_parser()
p_value = args.p
ai = args.ai
lr = args.lr
epoch_value = args.epochs

# 目标准确率
TARGET_ACCURACY = 92

# 创建保存图片的目录
save_dir = 'saving/time/plot'
os.makedirs(save_dir, exist_ok=True)

# 加载数据
with open('saving/time/Ai_6_P_10_epoch_100_is_iid_True_DML_alpha_0.3.pkl', 'rb') as f:
    dml_data_1 = pickle.load(f)
with open('saving/time/Ai_6_P_10_epoch_100_is_iid_False_DML_alpha_0.3.pkl', 'rb') as f:
    dml_data_2 = pickle.load(f)
with open('saving/time/Ai_6_P_10_epoch_100_is_iid_True_Local_alpha_0.3.pkl', 'rb') as f:
    local_data_3 = pickle.load(f)
with open('saving/time/Ai_6_P_10_epoch_100_is_iid_False_Local_alpha_0.3.pkl', 'rb') as f:
    local_data_4 = pickle.load(f)

# 确保Ai数量一致
num_ais = min(len(dml_data_1['ais']), len(dml_data_1['ais']))


# 张量转标量函数
def convert_to_python(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


# 计算累计时间函数
def calculate_cumulative_time(times):
    """将每轮时间转换为累计时间"""
    cumulative = []
    total = 0
    for time in times:
        total += convert_to_python(time)
        cumulative.append(total)
    return cumulative


# 截断数据到目标准确率的函数，并返回达到目标的累计时间
def truncate_to_target(times, accuracies, target):
    """截断数据，只保留达到目标准确率之前（包括达到时）的数据点，
    并返回达到目标准确率时的累计时间（如果达到的话）"""
    truncated_times = []
    truncated_accs = []
    target_time = None

    for t, acc in zip(times, accuracies):
        truncated_times.append(t)
        truncated_accs.append(acc)

        # 如果达到或超过目标准确率，记录时间并停止添加更多数据点
        if acc >= target and target_time is None:
            target_time = t
            break

    return truncated_times, truncated_accs, target_time


# 自定义颜色
custom_colors = ['#1515ff', '#ff7f50', '#ff0000', '#04bdfb']

# 为每个Ai绘制曲线图
for ai_idx in range(num_ais):
    print(f"\n===== Ai {ai_idx} 达到目标准确率({TARGET_ACCURACY}%)的累计时间 =====")

    # 提取并计算累计时间
    dml_times_1 = calculate_cumulative_time(dml_data_1['global_round_times'][ai_idx])
    dml_times_2 = calculate_cumulative_time(dml_data_2['global_round_times'][ai_idx])
    local_times_3 = calculate_cumulative_time(local_data_3['global_round_times'][ai_idx])
    local_times_4 = calculate_cumulative_time(local_data_4['global_round_times'][ai_idx])

    # 提取准确率并转换格式
    dml_accuracies_1 = [convert_to_python(v) for v in dml_data_1['global_round_accuracies'][ai_idx]]
    dml_accuracies_2 = [convert_to_python(v) for v in dml_data_2['global_round_accuracies'][ai_idx]]
    local_accuracies_3 = [convert_to_python(v) for v in local_data_3['global_round_accuracies'][ai_idx]]
    local_accuracies_4 = [convert_to_python(v) for v in local_data_4['global_round_accuracies'][ai_idx]]

    # 截断数据到目标准确率，并获取达到目标的时间
    dml_times_1, dml_accuracies_1, dml_time_1 = truncate_to_target(dml_times_1, dml_accuracies_1, TARGET_ACCURACY)
    dml_times_2, dml_accuracies_2, dml_time_2 = truncate_to_target(dml_times_2, dml_accuracies_2, TARGET_ACCURACY)
    local_times_3, local_accuracies_3, local_time_3 = truncate_to_target(local_times_3, local_accuracies_3,
                                                                         TARGET_ACCURACY)
    local_times_4, local_accuracies_4, local_time_4 = truncate_to_target(local_times_4, local_accuracies_4,
                                                                         TARGET_ACCURACY)

    # 输出每条曲线达到目标准确率的累计时间
    print(f"Proposed (IID): {dml_time_1:.4f} 秒" if dml_time_1 is not None else "Proposed (IID): 未达到目标准确率")
    print(
        f"Proposed (non-IID): {dml_time_2:.4f} 秒" if dml_time_2 is not None else "Proposed (non-IID): 未达到目标准确率")
    print(f"FedL (IID): {local_time_3:.4f} 秒" if local_time_3 is not None else "FedL (IID): 未达到目标准确率")
    print(f"FedL (non-IID): {local_time_4:.4f} 秒" if local_time_4 is not None else "FedL (non-IID): 未达到目标准确率")

    # 创建数据框
    data = pd.DataFrame({
        'Time': dml_times_1 + dml_times_2 + local_times_3 + local_times_4,
        'Accuracy': dml_accuracies_1 + dml_accuracies_2 + local_accuracies_3 + local_accuracies_4,
        'Model': ['Proposed (IID)'] * len(dml_times_1)
                 + ['Proposed (non-IID)'] * len(dml_times_2)
                 + ['FedL (IID)'] * len(local_times_3)
                 + ['FedL (non-IID)'] * len(local_times_4)
    })

    # 绘制曲线图
    plt.figure(figsize=(6, 5))

    # 绘制曲线
    sns.lineplot(
        data=data,
        x='Time',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=False,
        dashes=False,
        linewidth=1.5,
        palette=custom_colors
    )

    # 添加目标准确率的虚线
    max_time = max(data['Time']) if not data.empty else 0
    plt.axhline(y=TARGET_ACCURACY, color='gray', linestyle='--', linewidth=1.5,
                label=f'Target Accuracy ({TARGET_ACCURACY}%)')

    # 边框加粗设置
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')

    ax.tick_params(axis='both', width=2)
    ax.tick_params(axis='both', length=6)

    # 设置图表其他属性
    plt.xlabel('Time (seconds)', fontsize=14)  # x轴改为累计时间
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(10, 100)
    plt.xlim(0, max_time * 1.05 if max_time > 0 else 1)  # x轴范围调整为最大时间的1.05倍
    plt.legend(fontsize=13, title_fontsize=13, loc="lower right")
    plt.tight_layout()

    # 保存图片
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'time_Ai_{ai_idx}_lr_{lr}_{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")

    plt.show()
