import numpy as np
import pickle
import logging
from collections import defaultdict
from util import visualize_dirichlet_distribution

def perform_offloading(dataset, dict_users, offloading_data, max_attempts=1000):
    """
    根据offloading_data字典执行数据卸载

    参数:
    - dataset: MNIST数据集
    - dict_users: 原始客户端样本分配字典
    - offloading_data: 数据卸载字典
    - max_attempts: 最大尝试次数

    返回:
    - 卸载后的dict_users
    """

    total_samples_before = 0

    # 复制原始分配，避免修改原始数据
    new_dict_users = {i: set(indices) for i, indices in dict_users.items()}

    # 存储待处理的卸载请求
    pending_offloads = []
    for src, dest_dict in offloading_data.items():
        for dest, megabytes in dest_dict.items():
            pending_offloads.append((src, dest, int(megabytes)))

    print("[INFO] 待处理的卸载请求",pending_offloads)

    attempt = 0
    while pending_offloads and attempt < max_attempts:
        attempt += 1
        remaining_offloads = []

        for src, dest, megabytes in pending_offloads:
            # 检查源客户端和目标客户端是否存在
            if src not in new_dict_users or dest not in new_dict_users:
                continue

            samples_of_need_offload = megabytes * 335
            # # 计算源客户端的样本大小
            src_indices = list(new_dict_users[src])

            # 如果源客户端数据量小于要卸载的数据量，跳过此次卸载
            if len(src_indices) < samples_of_need_offload :
                remaining_offloads.append((src, dest, megabytes))
                continue

            # 随机选择样本进行卸载
            offload_indices = np.random.choice(src_indices, samples_of_need_offload, replace=False)

            # 执行卸载
            new_dict_users[src] -= set(offload_indices)
            new_dict_users[dest].update(offload_indices)

        # 更新待处理的卸载请求
        pending_offloads = remaining_offloads

    if pending_offloads:
        logging.warning(f"经过{max_attempts}次尝试后，仍有{len(pending_offloads)}个卸载请求未完成")

    # 输出卸载后的字典统计信息
    print("\n[DEBUG] 卸载后字典统计信息:")
    total_samples_after = 0
    for client_id, indices in new_dict_users.items():
        print(f"  客户端 {client_id}: {len(indices)} 个样本")
        total_samples_after += len(indices)
    print(f"  总样本数: {total_samples_after}")
    print(f"  样本总数变化: {total_samples_after - total_samples_before} (应该为0)\n")

    return new_dict_users
# def perform_offloading(dataset, dict_users, offloading_data, max_attempts=1000):
#     """
#     根据offloading_data字典执行数据卸载
#
#     参数:
#     - dataset: MNIST数据集
#     - dict_users: 原始客户端样本分配字典
#     - offloading_data: 数据卸载字典
#     - max_attempts: 最大尝试次数
#
#     返回:
#     - 卸载后的dict_users
#     """
#
#     total_samples_before = sum(len(indices) for indices in dict_users.values())
#
#     # 复制原始分配，避免修改原始数据
#     new_dict_users = {i: set(indices) for i, indices in dict_users.items()}
#
#     # 存储待处理的卸载请求
#     pending_offloads = []
#     for src, dest_dict in offloading_data.items():
#         for dest, megabytes in dest_dict.items():
#             pending_offloads.append((src, dest, int(megabytes)))
#
#     print("[INFO] 待处理的卸载请求", pending_offloads)
#
#     attempt = 0
#     while pending_offloads and attempt < max_attempts:
#         attempt += 1
#         remaining_offloads = []
#         executed_any = False
#
#         for src, dest, megabytes in pending_offloads:
#             # 检查源客户端和目标客户端是否存在
#             if src not in new_dict_users or dest not in new_dict_users:
#                 continue
#
#             samples_of_need_offload = megabytes * 335
#             # 计算源客户端的样本大小
#             src_indices = list(new_dict_users[src])
#
#             # 如果源客户端数据量小于要卸载的数据量，暂存此次卸载
#             if len(src_indices) < samples_of_need_offload:
#                 remaining_offloads.append((src, dest, megabytes))
#                 continue
#
#             # 随机选择样本进行卸载
#             offload_indices = np.random.choice(src_indices, samples_of_need_offload, replace=False)
#
#             # 执行卸载
#             new_dict_users[src] -= set(offload_indices)
#             new_dict_users[dest].update(offload_indices)
#             executed_any = True
#
#         # 如果本轮没有执行任何卸载，说明陷入死循环，退出
#         if not executed_any:
#             logging.warning(f"经过{attempt}次尝试后，无法再执行任何卸载，仍有{len(pending_offloads)}个请求未完成")
#             break
#
#         # 更新待处理的卸载请求
#         pending_offloads = remaining_offloads
#
#     if pending_offloads:
#         logging.warning(f"经过{max_attempts}次尝试后，仍有{len(pending_offloads)}个卸载请求未完成")
#
#     # 输出卸载后的字典统计信息
#     print("\n[DEBUG] 卸载后字典统计信息:")
#     total_samples_after = 0
#     for client_id, indices in new_dict_users.items():
#         print(f"  客户端 {client_id}: {len(indices)} 个样本")
#         total_samples_after += len(indices)
#     print(f"  总样本数: {total_samples_after}")
#     print(f"  样本总数变化: {total_samples_after - total_samples_before} (应该为0)\n")
#
#     return new_dict_users

# def perform_offloading(dataset, dict_users, offloading_data, max_attempts=10):
#     """
#     根据offloading_data字典执行数据卸载
#
#     参数:
#     - dataset: MNIST数据集
#     - dict_users: 原始客户端样本分配字典
#     - offloading_data: 数据卸载字典
#     - max_attempts: 最大尝试次数
#
#     返回:
#     - 卸载后的dict_users
#     """
#     # 复制原始分配，避免修改原始数据
#     new_dict_users = {i: set(indices) for i, indices in dict_users.items()}
#
#     # 存储待处理的卸载请求
#     pending_offloads = []
#     for src, dest_dict in offloading_data.items():
#         for dest, megabytes in dest_dict.items():
#             pending_offloads.append((src, dest, float(megabytes)))
#
#     attempt = 0
#     while pending_offloads and attempt < max_attempts:
#         attempt += 1
#         remaining_offloads = []
#
#         for src, dest, megabytes in pending_offloads:
#             # 检查源客户端和目标客户端是否存在
#             if src not in new_dict_users or dest not in new_dict_users:
#                 continue
#
#             # 计算源客户端的样本大小
#             src_indices = list(new_dict_users[src])
#             src_size_bytes = calculate_sample_size(dataset, src_indices)
#             src_size_mb = bytes_to_megabytes(src_size_bytes)
#
#             # 如果源客户端数据量小于要卸载的数据量，跳过此次卸载
#             if src_size_mb < megabytes:
#                 remaining_offloads.append((src, dest, megabytes))
#                 continue
#
#             # 计算需要卸载的样本数量
#             samples_to_offload = max(1, int(len(src_indices) * (megabytes / src_size_mb)))
#
#             # 随机选择样本进行卸载
#             offload_indices = np.random.choice(src_indices, samples_to_offload, replace=False)
#
#             # 执行卸载
#             new_dict_users[src] -= set(offload_indices)
#             new_dict_users[dest].update(offload_indices)
#
#         # 更新待处理的卸载请求
#         pending_offloads = remaining_offloads
#
#     if pending_offloads:
#         logging.warning(f"经过{max_attempts}次尝试后，仍有{len(pending_offloads)}个卸载请求未完成")
#
#     return new_dict_users


def mnist_iid(dataset, num_users, round_idx, offloading_data):
    try:
        # 加载 Ai_actdata.pkl 文件
        with open('Ai_actdata.pkl', 'rb') as f:
            Ai_actdata = pickle.load(f)

        # 获取当前循环的 Ai 和训练数据量
        Ai, training_sizes = Ai_actdata[round_idx]

        print(Ai, training_sizes)

        # 从文件中读取total_size
        with open('./total_size.pkl', 'rb') as f:
            total_size = pickle.load(f)

        # 计算每个客户端应获得的样本数
        num_items = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}

        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items[i], replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        # 执行数据卸载
        if offloading_data:
            dict_users = perform_offloading(dataset, dict_users, offloading_data)

        return dict_users
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认数据划分方法。")
        # 如果文件缺失，使用默认的IID划分方法
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        # 执行数据卸载
        if offloading_data:
            dict_users = perform_offloading(dataset, dict_users, offloading_data)

        return dict_users


# =========基于 Dirichlet 分布的 MNIST Non-IID划分方法==============
def mnist_noniid_dirichlet(dataset, num_users, round_idx, offloading_data, alpha=0.5):
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
            # 总m数，179m
            total_size = pickle.load(f)

        samples_of_1m = len(dataset) / total_size
        print("[INFO] 1M对应样本数:", samples_of_1m)

        # 计算每个客户端应获得的样本数   dict{key:对应的样本总数}
        num_samples = {i: (Ai * int(len(dataset) / total_size)) for i in range(num_users)}
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}. 使用默认均匀划分方式。")

    # 获取数据集标签
    if hasattr(dataset, 'train_labels'):  # 兼容不同版本的MNIST加载方式
        labels = dataset.train_labels.numpy()
    else:
        labels = np.array([label for _, label in dataset])

    # 10
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == y)[0] for y in range(num_classes)]

    print("[INFO] 不同标签拥有的index集合:",class_indices)

    # 为每个类生成 Dirichlet 分布用于用户分配比例
    client_indices = defaultdict(list)
    for c in range(num_classes):
        # 获取第c类的所有index
        idx_c = class_indices[c]
        # shuffle
        np.random.shuffle(idx_c)


        # 为这个类别在不同客户端上的分布生成一个 Dirichlet 向量
        proportions = np.random.dirichlet([alpha] * num_users)

        # 乘以样本总量并四舍五入得到分配样本数量
        proportions = np.array([int(p * len(idx_c)) for p in proportions])

        # print(f"[INFO] 初始分配 len(idx_c):{len(idx_c)} np.sum(proportions):{np.sum(proportions)} ")

        # 修正总和偏差
        diff = len(idx_c) - np.sum(proportions)
        for i in range(abs(diff)):
            proportions[i % num_users] += 1 if diff > 0 else -1

        # print(f"[CHECK] 修复后 类别{c} 分配前: {len(idx_c)}, 分配后: {np.sum(proportions)}")

        start = 0
        for i in range(num_users):
            client_indices[i].extend(idx_c[start:start + proportions[i]])
            start += proportions[i]

    dict_users = {}
    selected_indices = set()
    shortage_record = {}

    all_indices = set(range(len(dataset)))

    # 第一步：按 client_indices 分配一部分（能分多少分多少）
    for i in range(num_users):
        user_idx = list(set(client_indices[i]))  # 去重
        target = num_samples[i]

        if len(user_idx) >= target:
            selected = np.random.choice(user_idx, target, replace=False)
        else:
            selected = user_idx  # 所有能用的都用上
            shortage_record[i] = target - len(user_idx)  # 记录还差多少

        dict_users[i] = set(selected)
        selected_indices.update(selected)

        # print(f"[INFO] 第{i}个worker: 初选样本 {len(selected)}，目标数量 {target}")

    # 第二步：统一补充不够的客户端
    remaining_indices = list(all_indices - selected_indices)

    for i, shortage in shortage_record.items():
        if len(remaining_indices) >= shortage:
            supplement = np.random.choice(remaining_indices, shortage, replace=False)
        else:
            logging.warning(f"[WARNING] 可供补充的样本不足，仅剩 {len(remaining_indices)}，将启用重复采样")
            supplement = np.random.choice(list(all_indices), shortage, replace=True)

        dict_users[i].update(supplement)
        selected_indices.update(supplement)
        remaining_indices = list(all_indices - selected_indices)  # 实时更新可用补充集

        # print(f"[FIXED] 补充第{i}个worker: 增加 {len(supplement)} 样本，总计 {len(dict_users[i])}")


    print("\n[DEBUG] 卸载前字典统计信息:")
    total_samples_before = 0
    for client_id, indices in dict_users.items():
        print(f"  客户端 {client_id}: {len(indices)} 个样本")
        total_samples_before += len(indices)
    print(f"  总样本数: {total_samples_before}\n")


    # visualize_dirichlet_distribution(dataset,dict_users,10,10)

    # 执行数据卸载
    if offloading_data:
        dict_users = perform_offloading(dataset, dict_users, offloading_data)

    return dict_users