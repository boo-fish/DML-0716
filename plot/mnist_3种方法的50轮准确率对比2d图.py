# @Date       2425/12/18 下午2:59
# @Author     2424级电子信息计算机方向 艾春慧
# @University MUC
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# 注意：如果utils.options不存在，需要注释或替换为实际参数解析逻辑
try:
    from utils.options import args_parser
except ImportError:
    # 模拟参数解析，避免运行报错
    class Args:
        def __init__(self):
            self.p = 0.5
            self.ai = 1
            self.lr = 0.01
            self.epochs = 50
    def args_parser():
        return Args()
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

# 核心：定义数据根目录（包含benchmark/DML/RAMFL三个子文件夹）
root_dir = r'D:\project\DML-0716\results-1220'
# 创建保存图片的目录
save_dir = r'D:\project\DML-0716\大修实验图\3种方法的50轮准确率对比图'
os.makedirs(save_dir, exist_ok=True)


# ---------------------- 优化：自动遍历文件夹读取文件 ----------------------
def load_data_from_folder(folder_path):
    """从指定文件夹读取所有.pkl文件，返回{文件名: 数据}的字典"""
    data_dict = {}
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在，返回空数据")
        return data_dict
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                data_dict[filename] = pickle.load(f)
    return data_dict


# 读取三个子文件夹的所有数据
benchmark_data = load_data_from_folder(os.path.join(root_dir, 'benchmark'))
dml_data = load_data_from_folder(os.path.join(root_dir, 'DML'))
ramfl_data = load_data_from_folder(os.path.join(root_dir, 'RAMFL'))


# ---------------------- 按条件分类数据（IID/non-IID、H=10/5） ----------------------
def classify_data(data_dict, method_name):
    """将数据按 isIID_True/False、H=10/5 分类"""
    classified = {
        'iid_10': None, 'iid_5': None,
        'noniid_10': None, 'noniid_5': None
    }
    for filename, data in data_dict.items():
        # 从文件名提取特征：isIID_True/False、_24_10_（H=10）/_24_5_（H=5）
        is_iid = 'isIID_True' in filename
        is_h10 = '_24_10_' in filename
        is_h5 = '_24_5_' in filename

        if is_iid and is_h10:
            classified['iid_10'] = data
        elif is_iid and is_h5:
            classified['iid_5'] = data
        elif not is_iid and is_h10:
            classified['noniid_10'] = data
        elif not is_iid and is_h5:
            classified['noniid_5'] = data
    return classified


# 对三个方法的数据分类
benchmark_classified = classify_data(benchmark_data, 'Conventional FL')
dml_classified = classify_data(dml_data, 'Proposed')
ramfl_classified = classify_data(ramfl_data, 'RAMFL')

# ---------------------- 后续代码（原逻辑复用，替换手动加载的变量） ----------------------
# 加载数据 - IID
iid_dml_data_1 = dml_classified['iid_10']  # Proposed (H=10)
iid_dml_data_2 = dml_classified['iid_5']  # Proposed (H=5)
iid_local_data_3 = benchmark_classified['iid_10']  # Conventional FL (H=10)
iid_local_data_4 = benchmark_classified['iid_5']  # Conventional FL (H=5)
iid_ramfl_data_5 = ramfl_classified['iid_10']  # RAMFL (H=10)
iid_ramfl_data_6 = ramfl_classified['iid_5']  # RAMFL (H=5)

# 加载数据 - non-IID
noniid_dml_data_1 = dml_classified['noniid_10']  # Proposed (H=10)
noniid_dml_data_2 = dml_classified['noniid_5']  # Proposed (H=5)
noniid_local_data_3 = benchmark_classified['noniid_10']  # Conventional FL (H=10)
noniid_local_data_4 = benchmark_classified['noniid_5']  # Conventional FL (H=5)
noniid_ramfl_data_5 = ramfl_classified['noniid_10']  # RAMFL (H=10)
noniid_ramfl_data_6 = ramfl_classified['noniid_5']  # RAMFL (H=5)

# 确保Ai数量一致（原代码不变）
# 先过滤掉None值，避免报错
valid_data_list = [
    d for d in [
        iid_dml_data_1, iid_dml_data_2, iid_local_data_3, iid_local_data_4, iid_ramfl_data_5, iid_ramfl_data_6,
        noniid_dml_data_1, noniid_dml_data_2, noniid_local_data_3, noniid_local_data_4, noniid_ramfl_data_5, noniid_ramfl_data_6
    ] if d is not None
]

if valid_data_list:
    num_ais = min(len(d['ais']) for d in valid_data_list)
else:
    num_ais = 0
    print("警告：没有有效数据，将跳过绘图")


def convert_to_python(value):
    if hasattr(value, 'item'):
        return value.item()
    return value


# 自定义颜色（原代码不变）
custom_colors = [
    '#1515ff',  # Proposed (H=10)
    '#ff7f50',  # Proposed (H=5)
    '#ff0000',  # Conventional FL (H=10)
    '#04bdfb',  # Conventional FL (H=5)
    '#9370db',  # RAMFL (H=10)
    '#32cd32'  # RAMFL (H=5)
]

# 为每个Ai绘制组合图（原代码不变）
for ai_idx in range(num_ais):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 适当增加高度，为底部标题预留空间

    # ---------- 左图：IID ----------
    # 获取所有IID数据的最大轮数
    iid_data_list = [d for d in [iid_dml_data_1, iid_dml_data_2, iid_local_data_3, iid_local_data_4, iid_ramfl_data_5, iid_ramfl_data_6] if d is not None]
    if iid_data_list:
        max_rounds_iid = max(len(d['global_round_accuracies'][ai_idx]) for d in iid_data_list)
    else:
        max_rounds_iid = 50
    rounds_iid = list(range(1, max_rounds_iid + 1))


    # 处理各模型准确率数据，统一长度
    def pad_accuracy(acc_list, target_len):
        """补齐准确率列表长度至目标长度"""
        if acc_list is None:
            return [0] * target_len
        padded = [convert_to_python(v) for v in acc_list]
        if len(padded) < target_len:
            padded += [padded[-1]] * (target_len - len(padded))
        return padded[:target_len]


    # 处理IID各模型数据
    iid_dml_acc1 = pad_accuracy(iid_dml_data_1['global_round_accuracies'][ai_idx] if iid_dml_data_1 else None, max_rounds_iid)
    iid_dml_acc2 = pad_accuracy(iid_dml_data_2['global_round_accuracies'][ai_idx] if iid_dml_data_2 else None, max_rounds_iid)
    iid_local_acc3 = pad_accuracy(iid_local_data_3['global_round_accuracies'][ai_idx] if iid_local_data_3 else None, max_rounds_iid)
    iid_local_acc4 = pad_accuracy(iid_local_data_4['global_round_accuracies'][ai_idx] if iid_local_data_4 else None, max_rounds_iid)
    iid_ramfl_acc5 = pad_accuracy(iid_ramfl_data_5['global_round_accuracies'][ai_idx] if iid_ramfl_data_5 else None, max_rounds_iid)
    iid_ramfl_acc6 = pad_accuracy(iid_ramfl_data_6['global_round_accuracies'][ai_idx] if iid_ramfl_data_6 else None, max_rounds_iid)

    N_iid = len(rounds_iid)

    # 构建IID数据DataFrame（包含RAMFL）
    iid_data = pd.DataFrame({
        'Rounds': rounds_iid * 6,
        'Accuracy': iid_dml_acc1 + iid_dml_acc2 + iid_local_acc3 + iid_local_acc4 + iid_ramfl_acc5 + iid_ramfl_acc6,
        'Model': ['Proposed (H=10)'] * N_iid +
                 ['Proposed (H=5)'] * N_iid +
                 ['Conventional FL (H=10)'] * N_iid +
                 ['Conventional FL (H=5)'] * N_iid +
                 ['RAMFL (H=10)'] * N_iid +
                 ['RAMFL (H=5)'] * N_iid
    })

    # 绘制IID曲线
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
    ax1.set_ylim(10, 70)
    ax1.set_xlim(0, max_rounds_iid + 1)
    for spine in ax1.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax1.tick_params(axis='both', width=2, length=6)
    # 优化图例位置和大小
    ax1.legend(fontsize=10, title_fontsize=11, loc="lower right", frameon=True, framealpha=0.9)

    # ---------- 右图：non-IID ----------
    # 获取所有non-IID数据的最大轮数
    noniid_data_list = [d for d in [noniid_dml_data_1, noniid_dml_data_2, noniid_local_data_3, noniid_local_data_4, noniid_ramfl_data_5, noniid_ramfl_data_6] if d is not None]
    if noniid_data_list:
        max_rounds_noniid = max(len(d['global_round_accuracies'][ai_idx]) for d in noniid_data_list)
    else:
        max_rounds_noniid = 50
    rounds_noniid = list(range(1, max_rounds_noniid + 1))

    # 处理non-IID各模型数据，统一长度
    noniid_dml_acc1 = pad_accuracy(noniid_dml_data_1['global_round_accuracies'][ai_idx] if noniid_dml_data_1 else None, max_rounds_noniid)
    noniid_dml_acc2 = pad_accuracy(noniid_dml_data_2['global_round_accuracies'][ai_idx] if noniid_dml_data_2 else None, max_rounds_noniid)
    noniid_local_acc3 = pad_accuracy(noniid_local_data_3['global_round_accuracies'][ai_idx] if noniid_local_data_3 else None, max_rounds_noniid)
    noniid_local_acc4 = pad_accuracy(noniid_local_data_4['global_round_accuracies'][ai_idx] if noniid_local_data_4 else None, max_rounds_noniid)
    noniid_ramfl_acc5 = pad_accuracy(noniid_ramfl_data_5['global_round_accuracies'][ai_idx] if noniid_ramfl_data_5 else None, max_rounds_noniid)
    noniid_ramfl_acc6 = pad_accuracy(noniid_ramfl_data_6['global_round_accuracies'][ai_idx] if noniid_ramfl_data_6 else None, max_rounds_noniid)

    N_noniid = len(rounds_noniid)

    # 构建non-IID数据DataFrame（包含RAMFL）
    noniid_data = pd.DataFrame({
        'Rounds': rounds_noniid * 6,
        'Accuracy': noniid_dml_acc1 + noniid_dml_acc2 + noniid_local_acc3 + noniid_local_acc4 + noniid_ramfl_acc5 + noniid_ramfl_acc6,
        'Model': ['Proposed (H=10)'] * N_noniid +
                 ['Proposed (H=5)'] * N_noniid +
                 ['Conventional FL (H=10)'] * N_noniid +
                 ['Conventional FL (H=5)'] * N_noniid +
                 ['RAMFL (H=10)'] * N_noniid +
                 ['RAMFL (H=5)'] * N_noniid
    })

    # 绘制non-IID曲线
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
    ax2.set_ylim(10, 70)
    ax2.set_xlim(0, max_rounds_noniid + 1)
    for spine in ax2.spines.values():
        spine.set_linewidth(1)
        spine.set_color('black')
    ax2.tick_params(axis='both', width=2, length=6)
    ax2.legend(fontsize=10, title_fontsize=11, loc="lower right", frameon=True, framealpha=0.9)

    # ---------- 修正：添加子图下方标题（关键修改） ----------
    # 方案1：使用set_title并调整y轴偏移（推荐，更简洁）
    ax1.set_title('(a) CIFAR-10 dataset (IID data)', fontsize=14, y=-0.15)  # y=-0.15 表示在子图下方15%的位置
    ax2.set_title('(b) CIFAR-10 dataset (non-IID data)', fontsize=14, y=-0.15)

    # 方案2：若坚持使用ax.text，修改为如下配置（二选一即可，推荐方案1）
    # text_y_offset = 8  # 调整为y轴最小值附近（略低于10，保证可视）
    # x1_min, x1_max = ax1.get_xlim()
    # x2_min, x2_max = ax2.get_xlim()
    # ax1.text((x1_min + x1_max) / 2, text_y_offset, '(a) CIFAR-10 dataset (IID data)',
    #          fontsize=14, ha='center', va='top', transform=ax1.transData)
    # ax2.text((x2_min + x2_max) / 2, text_y_offset, '(b) CIFAR-10 dataset (non-IID data)',
    #          fontsize=14, ha='center', va='top', transform=ax2.transData)

    # ---------- 布局与保存（修正：调整底部间距，适配标题） ----------
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, wspace=0.25)  # 增大底部间距（0.2），容纳下方标题

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f'2d-3种方法的50轮准确率对比图__{current_time}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.show()