#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
import pickle
import os
import logging
import math
import numpy as np
import pickle
import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import torch
import random
import numpy as np
import pickle
import logging
from collections import defaultdict

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def mnist_iid(dataset, num_users, round_idx,offloading_data):
    try:
        # 加载 Ai_actdata.pkl 文件
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)

        # 获取当前循环的 Ai 和训练数据量
        Ai, training_sizes = Ai_actdata[round_idx]

        print(Ai, training_sizes)

        # 从文件中读取total_size
        with open('./total_size.pkl','rb') as f:

            total_size = pickle.load(f)

        #这行代码的逻辑就是训练集的总样本个数/总样本存储大小。就得到了一兆数据量会有多少样本数量，
        # 然后再乘以每个客户端的实际训练存储大小就得到了实际要得到多少样本数量
        num_items = {i: int(training_sizes[i] * len(dataset) / total_size) for i in range(num_users)}
        # print(num_items)

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
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


def mnist_noniid(dataset, num_users, round_idx):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    try:
        # 加载 Ai_actdata.pkl 文件
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)

        # 获取当前循环的 Ai 和训练数据量
        Ai, training_sizes = Ai_actdata[round_idx]

        print(Ai, training_sizes)



        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs = np.arange(num_shards * num_imgs)
        labels = dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # 根据训练数据量分配数据
        for i in range(num_users):
            # 根据训练数据量计算应该分配的 shard 数量
            num_shards_to_take = int(training_sizes[i] / (num_imgs * (30, 60)[0]))  # 假设 (30, 60) 是数据大小范围
            rand_set = set(np.random.choice(idx_shard, num_shards_to_take, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        return dict_users
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认数据划分方法。")
        # 如果文件缺失，使用默认的非 IID 划分方法
        num_shards, num_imgs = 200, 300
        idx_shard = [i for i in range(num_shards)]
        dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
        idxs = np.arange(num_shards * num_imgs)
        labels = dataset.train_labels.numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        return dict_users


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CustomMNIST(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform:
            img = self.transform(img)

        return img, target


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


def iid_partition(dataset, num_clients, samples_per_client, seed=42):
    """
    对数据集进行独立同分布划分，每个客户端分配固定数量的样本

    参数:
    - dataset: 要划分的数据集
    - num_clients: 客户端数量
    - samples_per_client: 每个客户端分配的样本数量
    - seed: 随机种子

    返回:
    - 每个客户端的样本索引列表
    """
    set_random_seed(seed)

    # 创建样本索引的随机排列
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # 为每个客户端分配样本索引
    client_indices = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = min(start_idx + samples_per_client, len(dataset))
        client_indices.append(indices[start_idx:end_idx])

    return client_indices


def non_iid_partition_by_label(dataset, num_clients, samples_per_client, seed=42):
    """
    按标签分布进行非独立同分布划分，同时确保每个客户端的数据量接近目标值

    参数:
    - dataset: 要划分的数据集
    - num_clients: 客户端数量
    - samples_per_client: 每个客户端分配的样本数量
    - seed: 随机种子

    返回:
    - 每个客户端的样本索引列表
    """
    set_random_seed(seed)

    # 获取每个样本的标签
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # 收集每个标签的样本索引
    label_indices = {label: np.where(labels == label)[0].tolist() for label in range(10)}

    # 打乱每个标签内的样本顺序
    for label in label_indices:
        random.shuffle(label_indices[label])

    # 为每个客户端分配特定标签的样本
    client_indices = [[] for _ in range(num_clients)]

    # 为每个客户端分配2个主要标签
    labels_per_client = 2
    samples_per_label = samples_per_client // labels_per_client

    for client_id in range(num_clients):
        # 确定该客户端的主要标签
        main_labels = [(client_id * labels_per_client + i) % 10 for i in range(labels_per_client)]

        # 为每个主要标签分配样本
        for label in main_labels:
            # 确保有足够的样本可供分配
            available_samples = len(label_indices[label])
            if available_samples >= samples_per_label:
                client_indices[client_id].extend(label_indices[label][:samples_per_label])
                label_indices[label] = label_indices[label][samples_per_label:]
            else:
                # 如果标签样本不足，则全部分配
                client_indices[client_id].extend(label_indices[label])
                label_indices[label] = []

    # 分配剩余样本，使每个客户端达到目标样本数量
    for client_id in range(num_clients):
        current_size = len(client_indices[client_id])
        needed_samples = samples_per_client - current_size

        if needed_samples > 0:
            # 从剩余样本中随机选择
            remaining_indices = []
            for label in label_indices:
                remaining_indices.extend(label_indices[label])

            if remaining_indices:
                # 随机选择所需数量的样本
                random.shuffle(remaining_indices)
                client_indices[client_id].extend(remaining_indices[:needed_samples])

                # 更新剩余样本
                remaining_indices = remaining_indices[needed_samples:]
                label_indices = {label: [] for label in range(10)}  # 清空标签索引
                for idx in remaining_indices:
                    label = labels[idx]
                    label_indices[label].append(idx)

    return client_indices


def create_client_datasets(dataset, client_indices):
    """
    根据索引列表为每个客户端创建数据集

    参数:
    - dataset: 原始数据集
    - client_indices: 每个客户端的样本索引列表

    返回:
    - 客户端数据集列表
    """
    client_datasets = []

    for indices in client_indices:
        client_dataset = Subset(dataset, indices)
        client_datasets.append(client_dataset)

    return client_datasets


def calculate_client_samples(sample_size_bytes, target_size_mb):
    """
    计算每个客户端应分配的样本数量

    参数:
    - sample_size_bytes: 单个样本的大小（字节）
    - target_size_mb: 目标数据量大小（兆字节）

    返回:
    - 每个客户端应分配的样本数量
    """
    # 将兆字节转换为字节
    target_size_bytes = target_size_mb * 1024 * 1024

    # 计算应分配的样本数量
    samples_per_client = int(target_size_bytes / sample_size_bytes)

    return samples_per_client


def load_actual_data_sizes(file_path):
    """
    从pkl文件中加载每个客户端的实际训练数据量大小

    参数:
    - file_path: pkl文件路径

    返回:
    - 每个客户端的实际训练数据量大小列表（兆字节）
    """
    # print(file_path)
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

            # 查找Ai=2.0的条目
            for item in data:
                Ai, actual_sizes = item
                if Ai == 2.0:  # 找到匹配的Ai值
                    return actual_sizes

            # 如果未找到匹配的Ai值
            # print(f"未找到Ai=2.0的条目，返回None")
            return None

    except FileNotFoundError:
        # print(f"文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


def random_partition_by_size(dataset, client_sizes_mb, sample_size_bytes, seed=42):
    """
    根据实际数据量大小随机划分数据集

    参数:
    - dataset: 要划分的数据集
    - client_sizes_mb: 每个客户端的实际数据量大小（兆字节）列表
    - sample_size_bytes: 单个样本的大小（字节）
    - seed: 随机种子

    返回:
    - 每个客户端的样本索引列表
    - 划分信息字典
    """
    set_random_seed(seed)

    # 计算每个客户端应分配的样本数量
    client_samples = [
        int(size_mb * 1024 * 1024 / sample_size_bytes)
        for size_mb in client_sizes_mb
    ]

    # 检查总样本数量是否超过数据集大小
    total_samples = sum(client_samples)
    if total_samples > len(dataset):
        raise ValueError(
            f"错误: 所有客户端分配的数据总量({total_samples})超过数据集大小({len(dataset)})，请重新设置Ai的值")

    # 创建样本索引的随机排列
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # 为每个客户端分配样本索引
    client_indices = []
    start_idx = 0

    for samples in client_samples:
        end_idx = start_idx + samples
        client_indices.append(indices[start_idx:end_idx])
        start_idx = end_idx

    # 创建划分信息
    partition_info = {
        'random_seed': seed,
        'sample_size_bytes': sample_size_bytes,
        'client_data_sizes_mb': client_sizes_mb,
        'client_sample_counts': client_samples,
        'client_indices': client_indices
    }

    return client_indices, partition_info


def save_partition_info(partition_info, file_path='partition_info.pkl'):
    """
    保存划分信息到文件

    参数:
    - partition_info: 划分信息字典
    - file_path: 保存文件路径
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(partition_info, f)
        # print(f"划分信息已保存到 {file_path}")
    except Exception as e:
        print(f"保存划分信息时出错: {e}")


def mnist_noniid(train_dataset, num_users, round_idx):
    # 设置随机种子
    set_random_seed(42)

    # 计算单个样本的存储大小
    sample_size = calculate_sample_size()
    # print(f"单个样本的存储大小: {sample_size} 字节")

    # 加载 Ai_actdata.pkl 文件
    with open('Ai_actdata.pkl', 'rb') as f:
        Ai_actdata = pickle.load(f)

    # 获取当前循环的 Ai 和训练数据量
    Ai, training_sizes = Ai_actdata[round_idx]

    # 计算每个客户端应分配的样本数量（基于2兆目标）
    samples_per_client = calculate_client_samples(sample_size, Ai)
    # print(f"每个客户端应分配的样本数量: {samples_per_client}")

    # 客户端数量
    num_clients = num_users

    # 按标签进行非独立同分布划分，同时确保每个客户端的数据量接近2兆
    client_indices = non_iid_partition_by_label(train_dataset, num_clients, samples_per_client)

    # 创建客户端数据集
    client_datasets = create_client_datasets(train_dataset, client_indices)

    # 检查每个客户端的样本数量和数据量
    # print("\n初始分配结果:")
    for i, dataset in enumerate(client_datasets):
        # dataset_size = len(dataset) * sample_size / (1024 * 1024)  # 转换为兆字节
        dataset_size = math.ceil(len(dataset) * sample_size / (1024 * 1024))  # 向上取整为兆字节
        # print(f"客户端 {i}: 样本数量 = {len(dataset)}, 数据量 = {dataset_size:.4f} MB")

    # 加载实际数据量大小
    actual_sizes = load_actual_data_sizes('./Ai_actdata.pkl')

    if actual_sizes is not None:
        # print("\n每个客户端的实际训练数据量大小 (MB):")
        # for i, size in enumerate(actual_sizes):
        #     print(f"客户端 {i}: {size} MB")

        try:
            # 根据实际数据量重新划分数据集
            new_client_indices, partition_info = random_partition_by_size(
                train_dataset, actual_sizes, sample_size
            )

            # 保存划分信息
            save_partition_info(partition_info)

            # 输出新的划分结果
            # print("\n重新划分后的结果:")
            for i, indices in enumerate(new_client_indices):
                # dataset_size = len(indices) * sample_size / (1024 * 1024)  # 转换为兆字节
                dataset_size = math.ceil(len(indices) * sample_size / (1024 * 1024))  # 向上取整为兆字节
                # print(f"客户端 {i}: 样本数量 = {len(indices)}, 数据量 = {dataset_size:.4f} MB")

            dict_users = {i: indices for i, indices in enumerate(new_client_indices)}

            return dict_users

        except ValueError as e:
            print(e)





# =========基于 Dirichlet 分布的 MNIST Non-IID划分方法==============
def mnist_noniid_dirichlet(dataset, num_users, round_idx, offloading_data,alpha=0.5):
    """
    基于 Dirichlet 分布的 MNIST 非IID划分方法
    参数：
    - dataset: MNIST 训练集对象，需有 dataset.train_labels
    - num_users: 客户端数量
    - round_idx: 当前轮数，用于获取 Ai_actdata
    - alpha: Dirichlet 分布的参数（控制非IID程度，越小越不均匀）
    - xxxxx: dml的每次卸载字典
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
        num_samples = {i: int(training_sizes[i] * len(dataset) / total_size) for i in range(num_users)}
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



def cifar_iid(dataset, num_users, round_idx):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    try:
        # 加载 Ai_actdata.pkl 文件
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)

        # 获取当前循环的 Ai 和训练数据量
        Ai, training_sizes = Ai_actdata[round_idx]
        # print(f"Ai:{Ai}, training_sizes:{training_sizes}")
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        total_size_file = os.path.join(script_dir, 'total_size.pkl')

        # 从文件中读取total_size
        with open('./total_size.pkl', 'rb') as f:
            total_size = pickle.load(f)

        # 获取客户端应该获取的数据量（根据训练数据量进行调整）
        #这里的逻辑感觉问题很大，在可行情况下，应该是合理的，在不可行情况下，逻辑应该就是错误的
        num_items = {i: int(training_sizes[i] * len(dataset) / total_size) for i in range(num_users)}

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
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