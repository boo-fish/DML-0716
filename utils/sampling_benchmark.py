from collections import defaultdict
import numpy as np
from torchvision import datasets, transforms
import pickle
import logging
def mnist_iid(dataset, num_users, round_idx, Ai_actdata=None, total_size=None):

    try:
        # =========关键修改1：移除从文件加载Ai_actdata的逻辑==========
        # with open('Ai_actdata.pkl', 'rb') as f:
        #     Ai_actdata = pickle.load(f)


        # =========关键修改2：增加索引边界检查==========
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用默认值")
            # 给默认值避免报错
            Ai = 6  # 与GetFlow默认ai值一致
            training_sizes = [0] * num_users

        else:
            # 获取当前循环的 Ai 和训练数据量
            Ai, training_sizes = Ai_actdata[round_idx]



        print(Ai, training_sizes)


        # =========关键修改3：移除从文件加载total_size的逻辑，优先使用传入的参数==========
        # with open('./total_size.pkl', 'rb') as f:
        #     total_size = pickle.load(f)
        if total_size is None:
            logging.warning("total_size未传递，使用默认值179（原total_size.pkl的默认值）")
            total_size = 179  # 原total_size.pkl的默认值，可根据实际情况调整


        # 计算每个客户端的数据集大小，不超过 Ai 和客户端容量
        num_items = {i: (Ai * int(len(dataset) / total_size))for i in range(num_users)}
        print(num_items)

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            if total_size > len(dataset):
                logging.warning("根据 Ai 计算的总数据量超过了数据集总大小，将进行调整==============================================================")

            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
            print(f"客户端{i}分配到的数据数量:{len(dict_users[i])}")

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


def cifar10_iid(dataset, num_users, round_idx, Ai_actdata=None, total_size=None):
    """
    适配CIFAR10数据集的独立同分布(IID)划分方法
    参数：
    - dataset: CIFAR10 训练集对象（PyTorch Dataset）
    - num_users: 客户端数量
    - round_idx: 当前轮数，用于获取Ai_actdata
    - Ai_actdata: 每轮的Ai和training_sizes，None则用默认值
    - total_size: 总样本量参考值，None则用CIFAR10默认值
    返回：
    - dict_users: dict[int, set[int]]，每个客户端分配到的样本索引集合
    """
    try:
        # 1. 处理Ai和training_sizes（逻辑不变，调整CIFAR10默认值）
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用CIFAR10默认值")
            Ai = 6  # 适配CIFAR10样本量的默认Ai值（原MNIST为6）
            training_sizes = [0] * num_users
        else:
            # 获取当前循环的 Ai 和训练数据量（逻辑完全复用）
            Ai, training_sizes = Ai_actdata[round_idx]

        print(f"CIFAR10 - 当前轮Ai值: {Ai}, 训练尺寸: {training_sizes}")

        # 2. 处理total_size（调整CIFAR10默认值）
        if total_size is None:
            logging.warning("total_size未传递，使用CIFAR10默认值200（原MNIST为179）")
            total_size = 200  # 适配CIFAR10 50000训练样本的默认值

        # 3. 计算每个客户端的数据集大小（逻辑完全复用）
        num_items = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}
        print(f"CIFAR10 - 各客户端目标样本数: {num_items}")

        # 4. IID核心划分逻辑（完全复用，仅调整日志提示）
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            # 检查目标样本数是否超过剩余数据量
            if num_items[i] > len(all_idxs):
                logging.warning(f"客户端{i}目标样本数({num_items[i]})超过剩余数据量({len(all_idxs)})，调整为剩余全部")
                target_num = len(all_idxs)
            else:
                target_num = num_items[i]

            # 随机采样（IID核心逻辑）
            dict_users[i] = set(np.random.choice(all_idxs, target_num, replace=False))
            # 移除已分配的索引
            all_idxs = list(set(all_idxs) - dict_users[i])
            print(f"CIFAR10客户端{i}最终分配到的数据数量:{len(dict_users[i])}")

        return dict_users

    except Exception as e:  # 扩大异常捕获范围，适配CIFAR10场景
        logging.error(f"参数计算失败: {e}. 使用CIFAR10默认IID划分方法。")
        # 异常时使用基础IID划分（均分）
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            # 最后一个客户端兜底，避免余数问题
            target_num = num_items if i < num_users - 1 else len(all_idxs)
            dict_users[i] = set(np.random.choice(all_idxs, target_num, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
            print(f"CIFAR10默认IID划分 - 客户端{i}分配数量:{len(dict_users[i])}")

        return dict_users


# =========基于 Dirichlet 分布的 MNIST Non-IID划分方法==============
def mnist_noniid_dirichlet(dataset, num_users, round_idx, alpha=0.5, Ai_actdata=None, total_size=None):
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
        # =========关键修改1：移除从文件加载Ai_actdata的逻辑==========
        # with open('Ai_actdata.pkl', 'rb') as f:
        #     Ai_actdata = pickle.load(f)

        # =========关键修改2：增加索引边界检查==========
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用默认值")
            Ai = 6  # 与GetFlow默认ai值一致
            training_sizes = [0] * num_users
        else:
            Ai, training_sizes = Ai_actdata[round_idx]

        # =========关键修改3：移除从文件加载total_size的逻辑，优先使用传入的参数==========
        # with open('./total_size.pkl', 'rb') as f:
        #     total_size = pickle.load(f)
        if total_size is None:
            logging.warning("total_size未传递，使用默认值179（原total_size.pkl的默认值）")
            total_size = 179  # 原total_size.pkl的默认值，可根据实际情况调整



        # 计算每个客户端应获得的样本数
        num_samples = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认均匀划分方式。")
        total_len = len(dataset)
        num_samples = {i: int(total_len / num_users) for i in range(num_users)}

    # 4. 核心修改：通用标签获取逻辑（兼容MNIST/CIFAR10）
    if hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()  # 兼容旧版MNIST
    elif hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)  # CIFAR10/新版MNIST
    else:
        labels = np.array([label for _, label in dataset])  # 兜底通用



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
        print(f"客户端{i}分配到的数据数量:{len(dict_users[i])}")

    return dict_users

def cifar10_noniid_dirichlet(dataset, num_users, round_idx, alpha=0.5, Ai_actdata=None, total_size=None):
    """
    适配CIFAR10的Dirichlet非IID划分（基于MNIST版本修改）
    参数：
    - dataset: CIFAR10训练集对象（PyTorch Dataset）
    - num_users: 客户端数量
    - round_idx: 当前轮数，用于获取Ai_actdata
    - alpha: Dirichlet分布参数（越小越不均匀）
    - Ai_actdata: 每轮的Ai和training_sizes，None则用默认值
    - total_size: 总样本量参考值，None则用CIFAR10默认值
    返回：
    - dict_users: 每个客户端的样本索引集合
    """
    try:
        # 1. 处理Ai和training_sizes（逻辑不变，仅调整默认值）
        if Ai_actdata is None or round_idx >= len(Ai_actdata):
            logging.warning(f"Ai_actdata为空或round_idx({round_idx})超出范围，使用CIFAR10默认值")
            Ai = 8  # 适配CIFAR10样本量的默认值
            training_sizes = [0] * num_users
        else:
            Ai, training_sizes = Ai_actdata[round_idx]

        # 2. 处理total_size（调整CIFAR10默认值）
        if total_size is None:
            logging.warning("total_size未传递，使用CIFAR10默认值200")
            total_size = 200  # 替换原MNIST的179

        # 3. 计算每个客户端目标样本数（逻辑不变）
        num_samples = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}

    except Exception as e:  # 扩大异常捕获范围，适配CIFAR10场景
        logging.error(f"参数计算失败: {e}. 使用均匀划分默认值。")
        total_len = len(dataset)
        num_samples = {i: int(total_len / num_users) for i in range(num_users)}

    # 4. 核心修改：通用标签获取逻辑（兼容MNIST/CIFAR10）
    if hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()  # 兼容旧版MNIST
    elif hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)    # CIFAR10/新版MNIST
    else:
        labels = np.array([label for _, label in dataset])  # 兜底通用

    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    # 5. Dirichlet分配逻辑（完全复用）
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
            end = start + proportions[i]
            client_indices[i].extend(idx_c[start:end])
            start += end

    # 6. 下采样到目标数量（完全复用）
    dict_users = {}
    for i in range(num_users):
        user_idx = client_indices[i]
        if len(user_idx) > num_samples[i]:
            user_idx = np.random.choice(user_idx, num_samples[i], replace=False)
        dict_users[i] = set(user_idx)
        print(f"CIFAR10客户端{i}分配到的数据数量:{len(dict_users[i])}")

    return dict_users