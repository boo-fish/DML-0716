import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
import pickle
import pandas as pd

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 加载数据
with open('saving/原创结果/p=10--Ai=2--dml--noniid.pkl', 'rb') as f:
    dml_data_10 = pickle.load(f)

with open('saving/原创结果/p=5--Ai=2--dml--noniid.pkl', 'rb') as f:
    dml_data_5 = pickle.load(f)

with open('saving/原创结果/p=10--Ai=2--local--noniid.pkl', 'rb') as f:
    local_data_10 = pickle.load(f)

with open('saving/原创结果/p=5--Ai=2--local--noniid.pkl', 'rb') as f:
    local_data_5 = pickle.load(f)

# 确保两个数据集的Ai数量一致
num_ais = min(len(dml_data_10['ais']), len(local_data_10['ais']))

# 定义辅助函数：将可能的张量转换为Python标量
def convert_to_python(value):
    if hasattr(value, 'item'):
        return value.item()
    return value

# 为每个Ai绘制单独的轮数-准确率曲线图
for ai_idx in range(num_ais):
    # 提取轮数列表
    max_rounds = max(len(dml_data_10['global_round_accuracies'][ai_idx]),
                     len(local_data_10['global_round_accuracies'][ai_idx]))
    rounds = list(range(1, max_rounds + 1))

    # 提取对应的测试集准确率
    dml_accuracies_10 = dml_data_10['global_round_accuracies'][ai_idx]
    local_accuracies_10 = local_data_10['global_round_accuracies'][ai_idx]

    # 扩展准确率列表到相同长度
    dml_accuracies_extended = dml_accuracies_10 + [None] * (max_rounds - len(dml_accuracies_10))
    local_accuracies_extended = local_accuracies_10 + [None] * (max_rounds - len(local_accuracies_10))

    # 转换为普通Python数值
    dml_accuracies_extended = [convert_to_python(v) for v in dml_accuracies_extended]
    local_accuracies_extended = [convert_to_python(v) for v in local_accuracies_extended]

    # 创建数据框
    data = pd.DataFrame({
        'Rounds': rounds * 2,
        'Accuracy': dml_accuracies_extended + local_accuracies_extended,
        'Model': ['Proposed'] * len(rounds) + ['FedL'] * len(rounds)
    })

    # 绘制曲线图
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=data, x='Rounds', y='Accuracy', hue='Model', style='Model', markers=False, dashes=False, linewidth=2.5)

    # 设置图表属性
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, max_rounds + 1)
    plt.legend(fontsize=12, title_fontsize=13, loc="lower right")
    plt.tight_layout()
    plt.show()