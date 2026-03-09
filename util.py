# @Date       2025/7/20 上午10:03
# @Author     2024级电子信息计算机方向 艾春慧
# @University MUC
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch
# matplotlib.use('TkAgg')

def visualize_dirichlet_distribution(dataset, dict_users, num_classes=10, num_users=10):
    """
    可视化 Dirichlet 分布下的客户端类别样本分布
    :param dataset: MNIST 训练集（含 train_labels）
    :param dict_users: 划分结果 dict[int, set[int]]
    """
    labels = dataset.train_labels.numpy()

    # 初始化每个客户端的类别计数
    class_counts = np.zeros((num_users, num_classes), dtype=int)

    for user_id in range(num_users):
        indices = list(dict_users[user_id])
        user_labels = labels[indices]
        for cls in range(num_classes):
            class_counts[user_id, cls] = np.sum(user_labels == cls)

    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(num_users)
    colors = plt.cm.get_cmap('tab10', num_classes)  # 10种颜色，对应0~9

    for cls in range(num_classes):
        plt.bar(np.arange(num_users), class_counts[:, cls], bottom=bottom,
                label=f'Class {cls}', color=colors(cls))
        bottom += class_counts[:, cls]

    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title('iid or Non-iid Distribution')
    plt.legend(loc='upper right', ncol=2)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
