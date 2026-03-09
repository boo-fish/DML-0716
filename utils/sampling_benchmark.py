from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms
import pickle
import logging
def mnist_iid(dataset, num_users, round_idx):
    try:
        # 加载 Ai_actdata.pkl 文件
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)

        # 从文件中读取工人节点容量
        with open('./worker_capacity.pkl','rb') as f:

            worker_capacity = pickle.load(f)

        # 获取当前循环的 Ai 和训练数据量
        Ai, training_sizes = Ai_actdata[round_idx]

        # 从文件中读取total_size
        with open('./total_size.pkl', 'rb') as f:

            total_size = pickle.load(f)

        # 计算每个客户端的数据集大小，不超过 Ai 和客户端容量
        num_items = {i: (Ai * int(len(dataset) / total_size))for i in range(num_users)}
        print(num_items)

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            if total_size > len(dataset):
                logging.warning("根据 Ai 计算的总数据量超过了数据集总大小，将进行调整==============================================================")

            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认数据划分方法。")
        # 如果文件缺失，使用默认的IID划分方法
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users


def calculate_sample_size():
    """计算MNIST数据集中单个样本的存储大小（图像+标签）"""
    # 加载MNIST数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )

    # 获取一个样本
    img, label = train_dataset[0]

    # 计算样本大小（图像和标签）
    img_size = img.numpy().nbytes
    label_size = np.array(label).nbytes

    # 返回单个样本的总大小（字节）
    return img_size + label_size


# =========基于 Dirichlet 分布的 MNIST Non-IID划分方法==============
def mnist_noniid_dirichlet(dataset, num_users, round_idx, alpha=0.5):
    """
    基于 Dirichlet 分布的 MNIST 非IID划分方法
    参数：
    - dataset: MNIST 训练集对象，需有 dataset.train_labels
    - num_users: 客户端数量
    - round_idx: 当前轮数，用于获取 Ai_actdata
    - alpha: Dirichlet 分布的参数（控制非IID程度，越小越不均匀）
    返回：
    - dict_users: dict[int, set[int]]，每个客户端分配到的样本索引集合
    """

    try:
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)
        Ai, training_sizes = Ai_actdata[round_idx]

        with open('./total_size.pkl', 'rb') as f:
            total_size = pickle.load(f)

        # 计算每个客户端应获得的样本数
        num_samples = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认均匀划分方式。")
        total_len = len(dataset)
        num_samples = {i: int(total_len / num_users) for i in range(num_users)}

    labels = dataset.train_labels.numpy()
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # 为每个类生成 Dirichlet 分布用于用户分配比例
    client_indices = defaultdict(list)
    for c in range(num_classes):
        idx_c = class_indices[c]
        np.random.shuffle(idx_c)

        # 为这个类别在不同客户端上的分布生成一个 Dirichlet 向量
        proportions = np.random.dirichlet([alpha] * num_users)

        # 乘以样本总量并四舍五入得到分配样本数量
        proportions = np.array([int(p * len(idx_c)) for p in proportions])

        # 修正总和偏差
        diff = len(idx_c) - np.sum(proportions)
        for i in range(abs(diff)):
            proportions[i % num_users] += 1 if diff > 0 else -1

        start = 0
        for i in range(num_users):
            client_indices[i].extend(idx_c[start:start + proportions[i]])
            start += proportions[i]

    # 最后从每个客户端中根据其“所需样本量”进行下采样
    dict_users = {}
    for i in range(num_users):
        user_idx = client_indices[i]
        if len(user_idx) > num_samples[i]:
            user_idx = np.random.choice(user_idx, num_samples[i], replace=False)
        dict_users[i] = set(user_idx)

    return dict_users
