import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.options import args_parser
import seaborn as sns
import pickle
import pandas as pd
import os
from datetime import datetime

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 解析参数
args = args_parser()
p_value = args.p
ai = args.ai
lr = args.lr
epoch_value = args.epochs

# 创建保存图片的目录
save_dir = r'D:\project\DML-0716\大修实验图\MNIST_准确率对比图'
os.makedirs(save_dir, exist_ok=True)

# 加载数据 - IID
with open('../saving/iid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_True_DML.pkl', 'rb') as f:
    iid_dml_data_1 = pickle.load(f)
with open('../saving/iid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_True_DML.pkl', 'rb') as f:
    iid_dml_data_2 = pickle.load(f)
with open('../saving/iid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_True_Local.pkl', 'rb') as f:
    iid_local_data_3 = pickle.load(f)
with open('../saving/iid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_True_Local.pkl', 'rb') as f:
    iid_local_data_4 = pickle.load(f)
with open(
        r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_50_is_iid_True_local_alpha_0.3_Final_Acc_91.1600_2025-12-14-07-05-15.pkl',
        'rb') as f:
    iid_ramfl_data_5 = pickle.load(f)
with open(
        r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_5_epoch_50_is_iid_False_local_alpha_0.3_Final_Acc_88.6400_2025-12-15-09-21-39.pkl',
        'rb') as f:
    iid_ramfl_data_6 = pickle.load(f)

# 加载数据 - non-IID
with open('../saving/noniid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_False_DML.pkl', 'rb') as f:
    noniid_dml_data_1 = pickle.load(f)
with open('../saving/noniid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_False_DML.pkl', 'rb') as f:
    noniid_dml_data_2 = pickle.load(f)
with open('../saving/noniid-lr-0.001/Ai_6_P_10_epoch_50_is_iid_False_Local.pkl', 'rb') as f:
    noniid_local_data_3 = pickle.load(f)
with open('../saving/noniid-lr-0.001/Ai_6_P_5_epoch_50_is_iid_False_Local.pkl', 'rb') as f:
    noniid_local_data_4 = pickle.load(f)
with open(
        r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_50_is_iid_False_local_alpha_0.3_Final_Acc_86.9600_2025-12-14-07-36-16.pkl',
        'rb') as f:
    noniid_ramfl_data_5 = pickle.load(f)
with open(
        r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_5_epoch_50_is_iid_False_local_alpha_0.3_Final_Acc_88.6400_2025-12-15-09-21-39.pkl',
        'rb') as f:
    noniid_ramfl_data_6 = pickle.load(f)

# 确保Ai数量一致（新增RAMFL数据的数量校验）
num_ais = min(
    len(iid_dml_data_1['ais']),
    len(iid_dml_data_2['ais']),
    len(iid_local_data_3['ais']),
    len(iid_local_data_4['ais']),
    len(iid_ramfl_data_5['ais']),  # 新增RAMFL数据校验
    len(iid_ramfl_data_6['ais']),  # 新增RAMFL数据校验
    len(noniid_dml_data_1['ais']),
    len(noniid_dml_data_2['ais']),
    len(noniid_local_data_3['ais']),
    len(noniid_local_data_4['ais']),
    len(noniid_ramfl_data_5['ais']),  # 新增RAMFL数据校验
    len(noniid_ramfl_data_6['ais'])  # 新增RAMFL数据校验
)


def convert_to_python(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


# 自定义颜色（新增2个颜色用于RAMFL方法，保持视觉区分度）
custom_colors = ['#1515ff', '#ff7f50', '#ff0000', '#04bdfb', '#9370db', '#32cd32']

# 为每个Ai绘制组合图
for ai_idx in range(num_ais):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 适当加宽画布，容纳更多图例

    # ---------- 左图：IID ----------
    # 更新最大轮数计算，包含RAMFL数据
    max_rounds_iid = max(
        len(iid_dml_data_1['global_round_accuracies'][ai_idx]),
        len(iid_dml_data_2['global_round_accuracies'][ai_idx]),
        len(iid_local_data_3['global_round_accuracies'][ai_idx]),
        len(iid_local_data_4['global_round_accuracies'][ai_idx]),
        len(iid_ramfl_data_5['global_round_accuracies'][ai_idx]),  # 新增RAMFL数据
        len(iid_ramfl_data_6['global_round_accuracies'][ai_idx])  # 新增RAMFL数据
    )
    rounds_iid = list(range(1, max_rounds_iid + 1))

    # 提取各方法准确率数据（包含RAMFL）
    iid_dml_acc1 = [convert_to_python(v) for v in iid_dml_data_1['global_round_accuracies'][ai_idx]]
    iid_dml_acc2 = [convert_to_python(v) for v in iid_dml_data_2['global_round_accuracies'][ai_idx]]
    iid_local_acc3 = [convert_to_python(v) for v in iid_local_data_3['global_round_accuracies'][ai_idx]]
    iid_local_acc4 = [convert_to_python(v) for v in iid_local_data_4['global_round_accuracies'][ai_idx]]
    iid_ramfl_acc5 = [convert_to_python(v) for v in iid_ramfl_data_5['global_round_accuracies'][ai_idx]]  # 新增RAMFL
    iid_ramfl_acc6 = [convert_to_python(v) for v in iid_ramfl_data_6['global_round_accuracies'][ai_idx]]  # 新增RAMFL
    N_iid = len(rounds_iid)

    # 构造DataFrame，包含RAMFL方法（新增2个模型类别）
    iid_data = pd.DataFrame({
        'Rounds': rounds_iid * 6,  # 从4个方法改为6个方法，对应6倍轮数
        'Accuracy': iid_dml_acc1 + iid_dml_acc2 + iid_local_acc3 + iid_local_acc4 + iid_ramfl_acc5 + iid_ramfl_acc6,
        'Model': ['Proposed (H=10)'] * N_iid + ['Proposed (H=5)'] * N_iid +
                 ['Conventional FL  (H=10)'] * N_iid + ['Conventional FL  (H=5)'] * N_iid +
                 ['RAMFL (H=10)'] * N_iid + ['RAMFL (H=5)'] * N_iid  # 新增RAMFL的两个模型名称
    })

    sns.lineplot(
        data=iid_data,
        x='Rounds',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=False,
        dashes=False,
        linewidth=1.5,
        palette=custom_colors,
        ax=ax1
    )

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(10, 100)
    ax1.set_xlim(0, max_rounds_iid + 1)
    for spine in ax1.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax1.tick_params(axis='both', width=2, length=6)
    # 调整图例大小和位置，避免遮挡
    ax1.legend(fontsize=10, title_fontsize=11, loc="lower right", bbox_to_anchor=(1, 0))

    # ---------- 右图：non-IID ----------
    # 更新最大轮数计算，包含RAMFL数据
    max_rounds_noniid = max(
        len(noniid_dml_data_1['global_round_accuracies'][ai_idx]),
        len(noniid_dml_data_2['global_round_accuracies'][ai_idx]),
        len(noniid_local_data_3['global_round_accuracies'][ai_idx]),
        len(noniid_local_data_4['global_round_accuracies'][ai_idx]),
        len(noniid_ramfl_data_5['global_round_accuracies'][ai_idx]),  # 新增RAMFL数据
        len(noniid_ramfl_data_6['global_round_accuracies'][ai_idx])  # 新增RAMFL数据
    )
    rounds_noniid = list(range(1, max_rounds_noniid + 1))

    # 提取各方法准确率数据（包含RAMFL）
    noniid_dml_acc1 = [convert_to_python(v) for v in noniid_dml_data_1['global_round_accuracies'][ai_idx]]
    noniid_dml_acc2 = [convert_to_python(v) for v in noniid_dml_data_2['global_round_accuracies'][ai_idx]]
    noniid_local_acc3 = [convert_to_python(v) for v in noniid_local_data_3['global_round_accuracies'][ai_idx]]
    noniid_local_acc4 = [convert_to_python(v) for v in noniid_local_data_4['global_round_accuracies'][ai_idx]]
    noniid_ramfl_acc5 = [convert_to_python(v) for v in
                         noniid_ramfl_data_5['global_round_accuracies'][ai_idx]]  # 新增RAMFL
    noniid_ramfl_acc6 = [convert_to_python(v) for v in
                         noniid_ramfl_data_6['global_round_accuracies'][ai_idx]]  # 新增RAMFL
    N_noniid = len(rounds_noniid)

    # 构造DataFrame，包含RAMFL方法（新增2个模型类别）
    noniid_data = pd.DataFrame({
        'Rounds': rounds_noniid * 6,  # 从4个方法改为6个方法，对应6倍轮数
        'Accuracy': noniid_dml_acc1 + noniid_dml_acc2 + noniid_local_acc3 + noniid_local_acc4 + noniid_ramfl_acc5 + noniid_ramfl_acc6,
        'Model': ['Proposed (H=10)'] * N_noniid + ['Proposed (H=5)'] * N_noniid +
                 ['Conventional FL  (H=10)'] * N_noniid + ['Conventional FL  (H=5)'] * N_noniid +
                 ['RAMFL (H=10)'] * N_noniid + ['RAMFL (H=5)'] * N_noniid  # 新增RAMFL的两个模型名称
    })

    sns.lineplot(
        data=noniid_data,
        x='Rounds',
        y='Accuracy',
        hue='Model',
        style='Model',
        markers=False,
        dashes=False,
        linewidth=1.5,
        palette=custom_colors,
        ax=ax2
    )

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(10, 100)
    ax2.set_xlim(0, max_rounds_noniid + 1)
    ax2.legend(fontsize=10, title_fontsize=11, loc="lower right", bbox_to_anchor=(1, 0))
    for spine in ax2.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax2.tick_params(axis='both', width=2, length=6)

    # ---------- 添加子图下方标题 ----------
    text_y_offset = -3  # 根据实际效果可调节
    x1_min, x1_max = ax1.get_xlim()
    x2_min, x2_max = ax2.get_xlim()
    ax1.text((x1_min + x1_max) / 2, text_y_offset, '(a) MNIST dataset (IID data)',
             fontsize=14, ha='center', va='top', transform=ax1.transData)
    ax2.text((x2_min + x2_max) / 2, text_y_offset, '(b) MNIST dataset (non-IID data)',
             fontsize=14, ha='center', va='top', transform=ax2.transData)

    # ---------- 布局与保存 ----------
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3)  # 增加横向间距，避免图例重叠

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'2d-con-combined_lr_{lr}_{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.show()