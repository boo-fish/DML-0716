import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # 用于添加外边框
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

# 创建保存图片的目录
save_dir = 'saving/0918/2d-iid'
os.makedirs(save_dir, exist_ok=True)

# 加载数据
# with open('saving/iid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_True_DML.pkl', 'rb') as f:
#     dml_data_1 = pickle.load(f)
# with open('saving/iid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_True_DML.pkl', 'rb') as f:
#     dml_data_2 = pickle.load(f)
# with open('saving/iid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_True_Local.pkl', 'rb') as f:
#     local_data_3 = pickle.load(f)
# with open('saving/iid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_True_Local.pkl', 'rb') as f:
#     local_data_4 = pickle.load(f)

# with open('saving/0918/本文方法/Ai_6_P_10_epoch_50_is_iid_True_DML_alpha_0.3.pkl', 'rb') as f:
#     dml_data_1 = pickle.load(f)
# with open('saving/0918/本文方法/Ai_6_P_5_epoch_50_is_iid_True_DML_alpha_0.3.pkl', 'rb') as f:
#     dml_data_2 = pickle.load(f)
# with open('saving/0918/基准方法/Ai_6_P_10_epoch_50_is_iid_True_local_alpha_0.3.pkl', 'rb') as f:
#     local_data_3 = pickle.load(f)
# with open('saving/0918/基准方法/Ai_6_P_5_epoch_50_is_iid_True_local_alpha_0.3.pkl', 'rb') as f:
#     local_data_4 = pickle.load(f)

with open('saving/0918/本文方法/Ai_6_P_10_epoch_50_is_iid_False_DML_alpha_0.3.pkl', 'rb') as f:
    dml_data_1 = pickle.load(f)
with open('saving/0918/本文方法/Ai_6_P_5_epoch_50_is_iid_False_DML_alpha_0.3.pkl', 'rb') as f:
    dml_data_2 = pickle.load(f)
with open('saving/0918/基准方法/Ai_6_P_10_epoch_50_is_iid_False_local_alpha_0.3.pkl', 'rb') as f:
    local_data_3 = pickle.load(f)
with open('saving/0918/基准方法/Ai_6_P_5_epoch_50_is_iid_False_local_alpha_0.3.pkl', 'rb') as f:
    local_data_4 = pickle.load(f)

# 确保Ai数量一致
num_ais = min(len(dml_data_1['ais']), len(dml_data_1['ais']))


# 张量转标量函数
def convert_to_python(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


# 自定义颜色
# custom_colors = ['#0d13aa', '#fe7e21', '#0172bd', '#fec900']
custom_colors = ['#1515ff', '#ff7f50',  '#ff0000','#04bdfb']
# custom_colors = ['#04bdfb', '#ff7f50',  '#ff0000','#1515ff']


# 为每个Ai绘制曲线图
for ai_idx in range(num_ais):
    # 提取轮数
    max_rounds = max(len(dml_data_1['global_round_accuracies'][ai_idx]),
                     len(dml_data_1['global_round_accuracies'][ai_idx]))
    rounds = list(range(1, max_rounds + 1))

    # 提取准确率并转换格式
    dml_accuracies_1 = [convert_to_python(v) for v in dml_data_1['global_round_accuracies'][ai_idx]]
    dml_accuracies_2 = [convert_to_python(v) for v in dml_data_2['global_round_accuracies'][ai_idx]]
    local_accuracies_3 = [convert_to_python(v) for v in local_data_3['global_round_accuracies'][ai_idx]]
    local_accuracies_4 = [convert_to_python(v) for v in local_data_4['global_round_accuracies'][ai_idx]]

    # 创建数据框
    N = len(rounds)
    data = pd.DataFrame({
        'Rounds': rounds * 4,
        'Accuracy': dml_accuracies_1 + dml_accuracies_2 + local_accuracies_3 + local_accuracies_4,
        'Model': ['Proposed (H=10)'] * N
                 + ['Proposed (H=5)'] * N
                 + ['Conventional FL Approach (H=10)'] * N
                 + ['Conventional FL Approach (H=5)'] * N
    })

    # 绘制曲线图（正方形画布）
    plt.figure(figsize=(6, 5))

    # 绘制曲线
    sns.lineplot(
        data=data,
        x='Rounds',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=False,
        dashes=False,
        linewidth=1.5,
        palette=custom_colors
    )

    # --------------------------
    # 核心：边框加粗设置
    # --------------------------
    # 获取当前坐标轴
    ax = plt.gca()

    # 1. 加粗坐标轴边框（上、下、左、右）
    for spine in ax.spines.values():
        spine.set_linewidth(1)  # 边框线宽（默认1，设置为2即加粗）
        spine.set_color('black')  # 边框颜色（确保清晰）

    # 2. 加粗坐标轴刻度线
    ax.tick_params(axis='both', width=2)  # 刻度线宽
    ax.tick_params(axis='both', length=6)  # 刻度线长度（更长更明显）

    # 设置图表其他属性
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(10, 100)
    plt.xlim(0, max_rounds + 1)
    plt.legend(fontsize=13, title_fontsize=13, loc="lower right")
    plt.tight_layout()

    # 保存图片
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'2d-con-iid_Ai_{ai}_lr_{lr}_{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")

    plt.show()