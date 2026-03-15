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
TARGET_ACCURACY = 50  # 目标准确率
ROOT_DIR = r'D:\project\DML-0716\results-全50准确率'  # 数据根目录（包含DML/benchmark/RAMFL子文件夹）
SAVE_DIR = r'D:\project\DML-0716\小修260309\绘图文件\收敛时间对比图\cifar10'  # 图片保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 自定义颜色（对应DML、benchmark、RAMFL）
custom_colors = [
    '#1515ff',  # DML (Proposed)
    '#ff0000',  # benchmark (Conventional FL)
    '#04bdfb'  # RAMFL
]

# 图表尺寸配置（和参考代码保持一致，增大高度容纳底部标题）
FIG_SIZE = (16, 8)  # 参考代码是(16,8)，原代码是(16,7)，统一调整
SUBPLOT_WIDTH_RATIO = 1  # 子图宽度比例
SUBPLOT_HEIGHT_RATIO = 1  # 子图高度比例


# ===================== 工具函数 =====================
def load_data_from_folder(folder_path):
    """从指定文件夹读取所有.pkl文件，返回{文件名: 数据}的字典"""
    data_dict = {}
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在！")
        return data_dict
    for filename in os.listdir(folder_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'rb') as f:
                    data_dict[filename] = pickle.load(f)
            except Exception as e:
                print(f"警告：读取文件 {file_path} 失败 - {e}")
    return data_dict


def classify_data(data_dict, method_name):
    """将数据按 isIID_True/False、H=10/5 分类（仅保留H=10）"""
    classified = {
        'iid_10': None, 'iid_5': None,
        'noniid_10': None, 'noniid_5': None
    }
    for filename, data in data_dict.items():
        # 从文件名提取特征（兼容原文件名格式：is_iid_True/False 或 isIID_True/False）
        is_iid = 'is_iid_True' in filename or 'isIID_True' in filename
        is_noniid = 'is_iid_False' in filename or 'isIID_False' in filename
        is_h10 = '_P_10_' in filename or '_24_10_' in filename  # 匹配H=10（P=10/H=10）

        if is_iid and is_h10:
            classified['iid_10'] = data
        elif is_noniid and is_h10:
            classified['noniid_10'] = data
    return classified


def convert_to_python(value):
    """张量转标量"""
    if hasattr(value, 'item'):
        return value.item()
    return value


def calculate_cumulative_time(times):
    """将每轮时间转换为累计时间"""
    cumulative = []
    total = 0
    for time in times:
        total += convert_to_python(time) * 3
        cumulative.append(total)
    return cumulative


def truncate_to_target(times, accuracies, target):
    """截断数据，只保留达到目标准确率之前（包括达到时）的数据点"""
    truncated_times = []
    truncated_accs = []
    for t, acc in zip(times, accuracies):
        truncated_times.append(t)
        # 关键修改：如果当前准确率超过目标，强制改为目标值
        if acc >= target:
            truncated_accs.append(target)
            break  # 截断，不再处理后续点
        else:
            truncated_accs.append(acc)
    return truncated_times, truncated_accs


# ===================== 加载并分类数据 =====================
# 读取三个方法的所有数据
dml_data_dict = load_data_from_folder(os.path.join(ROOT_DIR, 'DML'))
benchmark_data_dict = load_data_from_folder(os.path.join(ROOT_DIR, 'benchmark'))
ramfl_data_dict = load_data_from_folder(os.path.join(ROOT_DIR, 'RAMFL'))

# 兼容：如果benchmark数据在根目录（无benchmark子文件夹），从根目录读取local相关文件
if not benchmark_data_dict and 'local' in [f.split('_')[0] for f in os.listdir(ROOT_DIR) if f.endswith('.pkl')]:
    benchmark_data_dict = load_data_from_folder(ROOT_DIR)
    # 过滤出local相关文件
    benchmark_data_dict = {k: v for k, v in benchmark_data_dict.items() if 'local' in k}

# 分类数据（仅保留H=10）
dml_classified = classify_data(dml_data_dict, 'DML')
benchmark_classified = classify_data(benchmark_data_dict, 'benchmark')
ramfl_classified = classify_data(ramfl_data_dict, 'RAMFL')


# 校验数据是否存在
def check_data_exists(data, method_name, data_type):
    """检查数据是否存在，不存在则提示"""
    if data is None:
        print(f"警告：{method_name} 的 {data_type} (H=10) 数据未找到！")
    return data is not None


# ===================== 绘制图表 =====================
def plot_combined_chart(data_dict, ai_idx):
    """
    绘制组合图表（一行两列：IID + non-IID）
    :param data_dict: 包含dml/benchmark/ramfl数据的字典
    :param ai_idx: AI索引
    """
    # 创建一行两列的子图（图表尺寸和参考代码保持一致）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # 定义子图配置（统一标题文本，和参考代码完全一致）
    subplot_configs = [
        {'ax': ax1, 'plot_type': 'iid', 'title': '(a) CIFAR-100 dataset (IID data)'},
        {'ax': ax2, 'plot_type': 'noniid', 'title': '(b) CIFAR-100 dataset (non-IID data)'}
    ]

    # 遍历绘制每个子图
    for config in subplot_configs:
        ax = config['ax']
        plot_type = config['plot_type']
        title = config['title']

        # 提取该子图类型的数据
        plot_data = []
        method_names = []

        # 1. DML (Proposed)
        method_data = data_dict['dml'].get(f'{plot_type}_10')
        if check_data_exists(method_data, 'DML', plot_type):
            # 计算累计时间
            dml_times = calculate_cumulative_time(method_data['global_round_times'][ai_idx])
            # 提取并转换准确率
            dml_accs = [convert_to_python(v) for v in method_data['global_round_accuracies'][ai_idx]]
            # 截断数据到目标准确率
            dml_times, dml_accs = truncate_to_target(dml_times, dml_accs, TARGET_ACCURACY)
            if dml_times:
                plot_data.append({'Time': dml_times, 'Accuracy': dml_accs, 'Model': 'Proposed'})
                method_names.append('Proposed')

        # 2. benchmark (Conventional FL)
        method_data = data_dict['benchmark'].get(f'{plot_type}_10')
        if check_data_exists(method_data, 'benchmark', plot_type):
            benchmark_times = calculate_cumulative_time(method_data['global_round_times'][ai_idx])
            benchmark_accs = [convert_to_python(v) for v in method_data['global_round_accuracies'][ai_idx]]
            benchmark_times, benchmark_accs = truncate_to_target(benchmark_times, benchmark_accs, TARGET_ACCURACY)
            if benchmark_times:
                plot_data.append({'Time': benchmark_times, 'Accuracy': benchmark_accs, 'Model': 'Conventional FL'})
                method_names.append('Conventional FL')

        # 3. RAMFL
        method_data = data_dict['ramfl'].get(f'{plot_type}_10')
        if check_data_exists(method_data, 'RAMFL', plot_type):
            ramfl_times = calculate_cumulative_time(method_data['global_round_times'][ai_idx])
            ramfl_accs = [convert_to_python(v) for v in method_data['global_round_accuracies'][ai_idx]]
            ramfl_times, ramfl_accs = truncate_to_target(ramfl_times, ramfl_accs, TARGET_ACCURACY)
            if ramfl_times:
                plot_data.append({'Time': ramfl_times, 'Accuracy': ramfl_accs, 'Model': 'RAMFL'})
                method_names.append('RAMFL')

        # 无数据则跳过该子图
        if not plot_data:
            print(f"警告：AI {ai_idx} 的 {plot_type} 数据为空，该子图将为空！")
            # 统一子图样式，和参考代码对齐
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_ylim(10, 60)
            ax.set_xlim(0, 1)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
            ax.tick_params(axis='both', width=2, length=6)
        else:
            ax.set_ylim(10, 60)  # 这才是生效的y轴配置，你未修改这里
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

            # 设置子图样式，和参考代码完全对齐
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_ylim(5,55)
            ax.set_xlim(0, max_time * 1.05 if max_time > 0 else 1)

            # 美化边框和刻度，和参考代码一致
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
            ax.tick_params(axis='both', width=2, length=6)

            # 设置图例，和参考代码一致
            ax.legend(fontsize=10, title_fontsize=11, loc="lower right", frameon=True, framealpha=0.9)

        # ========== 关键修改：统一子图标题设置（和参考代码保持一致） ==========
        # 替换原有的ax.text方式，使用set_title，参数和参考代码完全匹配
        ax.set_title(title, fontsize=14, y=-0.15)

    # ========== 关键修改：统一布局参数（和参考代码保持一致） ==========
    plt.tight_layout()
    # 调整底部间距和子图间距，和参考代码完全一致
    plt.subplots_adjust(bottom=0.2, wspace=0.25)

    # 保存图片
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        SAVE_DIR,
        f'3种方法到达目标准确率{TARGET_ACCURACY}所需的时间_AI{ai_idx}_{current_time}.pdf'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ AI {ai_idx} 的组合图已保存至: {save_path}")
    plt.show()


# ===================== 主逻辑：遍历AI并绘图 =====================
# 准备组合数据字典
combined_data = {
    'dml': dml_classified,
    'benchmark': benchmark_classified,
    'ramfl': ramfl_classified
}

# 确定AI数量（取有数据的最大AI数）
ai_counts = []
for data in [dml_classified['iid_10'], benchmark_classified['iid_10'], ramfl_classified['iid_10']]:
    if data is not None and 'ais' in data:
        ai_counts.append(len(data['ais']))
num_ais = max(ai_counts) if ai_counts else 0

if num_ais == 0:
    print("❌ 未找到有效数据，无法绘图！")
else:
    print(f"📌 开始绘制 {num_ais} 个AI的组合图表...")
    for ai_idx in range(num_ais):
        plot_combined_chart(combined_data, ai_idx)