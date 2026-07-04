from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms
import pickle
import logging


def mnist_iid(dataset, num_users, round_idx, Ai_actdata=None, total_size=None):
    try:
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用默认值")
            Ai = 6
            training_sizes = [0] * num_users
        else:
            Ai, training_sizes = Ai_actdata[round_idx]

        print(Ai, training_sizes)

        if total_size is None:
            logging.warning("total_size未传递，使用默认值179")
            total_size = 179

        # 计算每个客户端的数据集大小（Ai=15兆）
        num_items = {i: (Ai * int(len(dataset) / total_size))for i in range(num_users)}
        print(num_items)

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            if total_size > len(dataset):
                logging.warning("根据 Ai 计算的总数据量超过了数据集总大小，将进行调整")
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
            print(f"客户端{i}分配到的数据数量:{len(dict_users[i])}")


        return dict_users
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认数据划分方法。")
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users


def calculate_sample_size():
    """计算MNIST数据集中单个样本的存储大小（图像+标签）"""
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True,
        transform=transforms.ToTensor()
    )
    img, label = train_dataset[0]
    img_size = img.numpy().nbytes
    label_size = np.array(label).nbytes
    return img_size + label_size


def mnist_noniid_dirichlet(dataset, num_users, round_idx, alpha=0.5, Ai_actdata=None, total_size=None):
    try:
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用默认值")
            Ai = 6
            training_sizes = [0] * num_users
        else:
            Ai, training_sizes = Ai_actdata[round_idx]

        if total_size is None:
            logging.warning("total_size未传递，使用默认值179")
            total_size = 179

        # 计算每个客户端应获得的样本数（Ai=15兆）
        num_samples = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认均匀划分方式。")
        total_len = len(dataset)
        num_samples = {i: int(total_len / num_users) for i in range(num_users)}

    # 兼容train_labels/train_targets
    labels = dataset.train_labels.numpy() if hasattr(dataset, 'train_labels') else dataset.train_targets.numpy()
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    client_indices = defaultdict(list)
    for c in range(num_classes):
        idx_c = class_indices[c]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * num_users)
        proportions = np.array([int(p * len(idx_c)) for p in proportions])
        diff = len(idx_c) - np.sum(proportions)
        for i in range(abs(diff)):
            proportions[i % num_users] += 1 if diff > 0 else -1
        start = 0
        for i in range(num_users):
            client_indices[i].extend(idx_c[start:start + proportions[i]])
            start += proportions[i]

    dict_users = {}
    for i in range(num_users):
        user_idx = client_indices[i]
        if len(user_idx) > num_samples[i]:
            user_idx = np.random.choice(user_idx, num_samples[i], replace=False)
        dict_users[i] = set(user_idx)
        print(f"客户端{i}分配到的数据数量:{len(dict_users[i])}")

    return dict_users