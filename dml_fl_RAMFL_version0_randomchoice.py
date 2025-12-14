import matplotlib
import os
from utils.sampling_benchmark import mnist_noniid_dirichlet
matplotlib.use('TkAgg')
import sys  # 新增：导入sys模块用于终止程序
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.network_environment import NetworkEnvironment
from utils.sampling_benchmark import mnist_iid
from utils.options import args_parser
from models.Update_benchmark import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
import torch.multiprocessing as mp
from utils.get_offload_dict_dml import GetFlow
from datetime import datetime

# 获取当前时间
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

# ========== 新增：多服务器配置 ==========
SERVER_NUM = 3  # 服务器个数设置为3
MAP_UPDATE_INTERVAL = 5  # 每5轮全局训练更新客户端-服务器映射（论文核心逻辑）

def client_train(args, dataset, idxs, server_w_param, client_data_size, worker_capacity, client_id):
    """客户端训练函数，修改为：使用分配的服务器参数作为初始参数"""
    client_start_time = time.time()
    idxs = list(idxs)
    data_size = len(idxs)
    w = server_w_param.copy()  # 初始参数来自对应服务器，而非全局统一参数
    loss = 0.0

    # 模型初始化（保持原有逻辑）
    if args.model == 'cnn' and args.dataset == 'cifar':
        net = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        img_size = dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError('Error: unrecognized model')
    net.load_state_dict(server_w_param)  # 加载服务器参数

    # 无数据客户端直接返回（保持原有逻辑）
    if data_size == 0:
        client_end_time = time.time()
        client_elapsed_time = client_end_time - client_start_time
        return w, loss, client_elapsed_time, client_id

    # 客户端容量计算（保持原有逻辑）
    sample, label = dataset[0]
    sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
    label_size = 4 / (1024 * 1024)
    num_samples_in_capacity = int(worker_capacity[client_id] / (sample_size + label_size))

    start_idx = 0
    batch_num = 0
    while start_idx < data_size:
        remaining_samples = data_size - start_idx
        end_idx = start_idx + min(num_samples_in_capacity, remaining_samples)
        local_idxs = idxs[start_idx:end_idx]

        # 本地训练（保持原有逻辑）
        local = LocalUpdate(args=args, dataset=dataset, idxs=local_idxs, client_data_size=num_samples_in_capacity)
        w, loss = local.train(net=net)
        net.load_state_dict(w)
        start_idx = end_idx
        batch_num += 1

    client_end_time = time.time()
    client_elapsed_time = client_end_time - client_start_time
    torch.cuda.empty_cache()
    return w, loss, client_elapsed_time, client_id

def main():
    args = args_parser()
    alpha = args.alpha
    p_value = args.p
    ai = args.ai
    epoch_value = args.epochs
    total_size = args.total_mb

    # 原有数据加载逻辑（保持不变）
    offloading_data, worker_capacity, Ai_actdata, training_data = GetFlow(p=p_value, ai=ai)
    print("main中的卸载字典", offloading_data)
    print("main中Ai的len", len(Ai_actdata))
    print("main中Ai", Ai_actdata)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 原有结果存储逻辑（保持不变）
    test_accuracies = []
    global_round_accuracies = []
    global_round_times = []

    # 遍历每个Ai（保持原有逻辑）
    for round_idx, (Ai, training_sizes) in enumerate(Ai_actdata):
        # 数据集加载（保持原有逻辑）
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            if args.iid:
                print("iid")
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx, Ai_actdata=Ai_actdata, total_size=total_size)
            else:
                print("non-iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, alpha=alpha, Ai_actdata=Ai_actdata, total_size=total_size)
        else:
            exit('Error: unrecognized dataset')

        # 模型初始化（保持原有逻辑）
        img_size = dataset_train[0][0].shape
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        net_glob.train()
        init_w = net_glob.state_dict()  # 初始模型参数

        # ========== 新增：多服务器初始化 ==========
        # 1. 服务器参数字典：key=服务器ID(0/1/2)，value=模型参数
        server_w = {s_id: copy.deepcopy(init_w) for s_id in range(SERVER_NUM)}
        # 2. 客户端-服务器映射表：初始随机分配（论文"随机聚类"核心）
        client_server_map = {client_id: np.random.randint(0, SERVER_NUM) for client_id in range(args.num_users)}
        print(f"初始客户端-服务器映射：{client_server_map}")

        # 原有客户端数据大小计算（保持不变）
        client_dataset_sizes = {}
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
        label_size = 4 / (1024 * 1024)
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))

        # 全局训练轮次（保持原有逻辑）
        accuracies_per_round = []
        times_per_round = []
        for epoch in range(args.epochs):
            # ========== 新增：每5轮更新客户端-服务器映射（论文核心逻辑） ==========
            if epoch % MAP_UPDATE_INTERVAL == 0 and epoch != 0:
                client_server_map = {client_id: np.random.randint(0, SERVER_NUM) for client_id in range(args.num_users)}
                print(f"第{epoch+1}轮更新映射：{client_server_map}")

            global_start_time = time.time()
            loss_locals = []
            client_processing_times = {}

            # 选中参与训练的客户端（保持原有逻辑）
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            # ========== 修改：多进程训练时传递对应服务器的参数 ==========
            with mp.Pool(processes=m) as pool:
                # 为每个选中的客户端分配对应的服务器参数
                task_args = []
                for idx in idxs_users:
                    s_id = client_server_map[idx]  # 当前客户端分配的服务器ID
                    s_param = server_w[s_id]  # 该服务器的当前模型参数
                    task_args.append((args, dataset_train, dict_users[idx], s_param,
                                     client_dataset_sizes[idx], worker_capacity, idx))
                results = pool.starmap(client_train, task_args)

            # ========== 新增：按服务器分组聚合参数（论文多服务器聚合逻辑） ==========
            # 1. 按服务器分组收集客户端参数
            server_client_params = {s_id: [] for s_id in range(SERVER_NUM)}
            for w, loss, elapsed_time, client_id in results:
                if w is not None and loss is not None:
                    s_id = client_server_map[client_id]  # 客户端对应的服务器ID
                    server_client_params[s_id].append(copy.deepcopy(w)) # 将该客户端的参数append对应的服务器
                    loss_locals.append(copy.deepcopy(loss))
                    adjusted_time = elapsed_time / worker_capacity[client_id]
                    client_processing_times[client_id] = adjusted_time
            print("各个服务器收集到的参数个数：")
            print(f"服务器0:{len(server_client_params[0])} \n服务器1:{len(server_client_params[1])}\n服务器2:{len(server_client_params[2])}")

            # 2. 每个服务器独立聚合（FedAvg）
            for s_id in range(SERVER_NUM):
                params = server_client_params[s_id]
                if len(params) > 0:
                    server_w[s_id] = FedAvg(params)  # 服务器更新自身模型参数

            # ========== 新增：生成全局测试模型（所有服务器参数平均） ==========
            # 为了保持原有测试逻辑，取3个服务器参数的平均值作为全局模型
            w_glob = copy.deepcopy(init_w)
            for key in w_glob.keys():
                w_glob[key] = torch.zeros_like(w_glob[key])
                for s_id in range(SERVER_NUM):
                    w_glob[key] += server_w[s_id][key]
                w_glob[key] /= SERVER_NUM  # 平均所有服务器参数

            # 测试全局模型（保持原有逻辑）
            net_glob.load_state_dict(w_glob)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            # ==============================================
            # 核心新增：判断是否满足终止条件
            # ==============================================
            if args.iid:
                # iid模式：检查第14轮（epoch=13）准确率是否超过90%
                if epoch == 14 and acc_test > 90:
                    print(f"iid模式第14轮准确率{acc_test:.2f}%超过90%，终止运行")
                    sys.exit(0)  # 终止程序
            else:
                # no-iid模式：检查第10轮（epoch=9）准确率是否超过83%
                if epoch == 10 and acc_test > 83:
                    print(f"no-iid模式第10轮准确率{acc_test:.2f}%超过83%，终止运行")
                    sys.exit(0)  # 终止程序







            # 记录结果（保持原有逻辑）
            max_client_time = max(client_processing_times.values()) if client_processing_times else 0
            global_elapsed_time = max_client_time
            times_per_round.append(global_elapsed_time)
            print(f"Ai={Ai}, Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, Time: {global_elapsed_time}")
            accuracies_per_round.append(acc_test)
            torch.cuda.empty_cache()

        # 存储结果（保持原有逻辑）
        global_round_accuracies.append(accuracies_per_round)
        global_round_times.append(times_per_round)
        test_accuracies.append(accuracies_per_round[-1])

    # 保存结果（保持原有逻辑）
    save_dir = 'results/RAMFL'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'Ai_{Ai}_P_{p_value}_epoch_{epoch_value}_is_iid_{args.iid}_local_alpha_{alpha}_Final_Acc_{accuracies_per_round[-1]:.4f}_{formatted_time}.pkl')
    data_to_save = {
        'ais': [x[0] for x in Ai_actdata],
        'test_accuracies': test_accuracies,
        'global_round_accuracies': global_round_accuracies,
        'global_round_times': global_round_times
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print("成功保存数据pkl文件")
    return data_to_save

if __name__ == "__main__":
    mp.set_start_method('spawn')
    final_results = main()
    print("训练完成，最终结果：", final_results)