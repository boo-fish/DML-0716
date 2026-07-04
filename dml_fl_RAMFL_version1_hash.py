import matplotlib
import os
from utils.sampling_benchmark import mnist_noniid_dirichlet, cifar10_noniid_dirichlet, mnist_iid, cifar10_iid

matplotlib.use('TkAgg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.network_environment import NetworkEnvironment
from utils.options import args_parser
from models.Update_benchmark import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, VGG11, ResNet18Cifar
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
import torch.multiprocessing as mp
from utils.get_offload_dict_dml import GetFlow
from datetime import datetime
import sys  # 补充缺失的导入（如果用到终止逻辑）

# 获取当前时间
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")

# ========== 多服务器配置（与论文一致） ==========
SERVER_NUM = 3  # 服务器个数（可调整，论文中为动态适配）
MAP_UPDATE_INTERVAL = 5  # 每5轮更新客户端-服务器映射（论文核心逻辑）


def client_train(args, dataset, idxs, server_w_param, client_data_size, worker_capacity, client_id):
    """客户端训练函数，使用分配的服务器参数作为初始参数"""
    client_start_time = time.time()
    idxs = list(idxs)
    data_size = len(idxs)
    w = server_w_param.copy()
    loss = 0.0

    # 模型初始化（保持原有逻辑）
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net = CNNCifar(args=args).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'mnist':
        net = CNNMnist(args=args).to(args.device)

    elif args.model == 'vgg11' and args.dataset == 'cifar10':
        net = VGG11(args=args).to(args.device)

    elif args.model == 'vgg11' and args.dataset == 'mnist':
        net = VGG11(args=args).to(args.device)
        
    ###### 新增 ResNet-18 的逻辑 ######
    elif args.model == 'resnet18' and args.dataset == 'cifar100':
        # 注意：如果你的 args.num_classes 默认不是 100，这里可以直接强制传入 100
        net = ResNet18Cifar(args=args, num_classes=100).to(args.device)

    elif args.model == 'mlp':
        img_size = dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError('Error: unrecognized model')
    
    net.load_state_dict(server_w_param)

    # 无数据客户端直接返回
    if data_size == 0:
        client_end_time = time.time()
        client_elapsed_time = client_end_time - client_start_time
        return w, loss, client_elapsed_time, client_id

    # 客户端容量计算
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

        # 本地训练
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
    torch.cuda.empty_cache()

    args = args_parser()
    alpha = args.alpha
    p_value = args.p
    ai = args.ai
    epoch_value = args.epochs
    total_size = args.total_mb

    # 数据加载（保持原有逻辑）
    offloading_data, worker_capacity, Ai_actdata, training_data = GetFlow(p=p_value, ai=ai,dataset=args.dataset)
    # print("main中的卸载字典", offloading_data)
    # print("main中Ai的len", len(Ai_actdata))
    # print("main中Ai", Ai_actdata)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 设定数据集对应的目标准确率
    if args.dataset == 'mnist':
        target_accuracy = 92.0  # mnist数据集目标准确率92%
        target_epoch = 50
    elif args.dataset == 'cifar10':
        target_accuracy = 55.0  # cifar10数据集目标准确率55%
        target_epoch = 100
        
    ###### DML小修0308 ######
    elif args.dataset == 'cifar100':
        target_accuracy = 45.0  # cifar100数据集目标准确率55%
        target_epoch = 100
        
    else:
        target_accuracy = 0.0  # 未知数据集默认值
        print(f"警告：未识别的数据集 {args.dataset}，未设置目标准确率")

    print(f"目标准确率：{target_accuracy}，目标epoch：{target_epoch}")

    # 结果存储（保持原有逻辑）
    test_accuracies = []
    global_round_accuracies = []
    global_round_times = []

    # 遍历每个Ai
    for round_idx, (Ai, training_sizes) in enumerate(Ai_actdata):
        # 数据集加载
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
            if args.iid:
                print("iid")
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx, Ai_actdata=Ai_actdata,
                                       total_size=total_size)
            else:
                print("non-iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, alpha=alpha,
                                                    Ai_actdata=Ai_actdata, total_size=total_size)

        elif args.dataset == 'cifar10':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=False, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=trans_cifar)



            if args.iid:
                print("iid")
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx, Ai_actdata=Ai_actdata,
                                         total_size=total_size)
            else:
                print("non-iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, alpha=alpha,
                                                      Ai_actdata=Ai_actdata, total_size=total_size)

        elif args.dataset == 'cifar100':
            # 1. 训练集 Transform（加入数据增强）
            trans_cifar_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            # 2. 测试集 Transform（纯净转换）
            trans_cifar_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

            # 3. 分别应用不同的 transform
            dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar_train)
            dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar_test)
        
            if args.iid:
                print("iid")
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx, Ai_actdata=Ai_actdata,
                                         total_size=total_size)
            else:
                print("non-iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, alpha=alpha,
                                                      Ai_actdata=Ai_actdata, total_size=total_size)
        
        else:
            exit('Error: unrecognized dataset')

        # 模型初始化
        img_size = dataset_train[0][0].shape
        if args.model == 'cnn' and args.dataset == 'cifar10':
            net_glob = CNNCifar(args=args).to(args.device)

        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)

        elif args.model == 'vgg11' and args.dataset == 'cifar10':
            net_glob = VGG11(args=args).to(args.device)

        elif args.model == 'vgg11' and args.dataset == 'mnist':
            net_glob = VGG11(args=args).to(args.device)
            
            
        ###### 新增 ResNet-18 的逻辑 ######
        elif args.model == 'resnet18' and args.dataset == 'cifar100':
            # 注意：如果你的 args.num_classes 默认不是 100，这里可以直接强制传入 100
            net_glob = ResNet18Cifar(args=args, num_classes=100).to(args.device)

        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        net_glob.train()
        init_w = net_glob.state_dict()

        # ========== 核心修改：每个Ai轮次重新初始化多服务器参数 ==========
        # 确保每个Ai轮次都从全新的初始参数开始，而非复用前一轮的参数
        server_w = {s_id: copy.deepcopy(init_w) for s_id in range(SERVER_NUM)}

        # ========== 核心修改1：初始客户端-服务器映射（论文公式：ServerId_i = i % S） ==========
        # 基于哈希取模实现初始分配，确保客户端与服务器的固定映射（初始状态）
        client_server_map = {client_id: client_id % SERVER_NUM for client_id in range(args.num_users)}
        print(f"初始客户端-服务器映射（哈希取模）：{client_server_map}")

        # 客户端数据大小计算（保持不变）
        client_dataset_sizes = {}
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
        label_size = 4 / (1024 * 1024)
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))

        # 全局训练轮次
        accuracies_per_round = []
        times_per_round = []
        for epoch in range(args.epochs):
            # ========== 核心修改2：动态更新映射（论文随机聚类策略） ==========
            # 每MAP_UPDATE_INTERVAL轮打乱客户端顺序，重新基于哈希取模分配服务器
            if epoch % MAP_UPDATE_INTERVAL == 0 and epoch != 0:
                # 1. 随机打乱客户端ID顺序（实现动态聚类）
                shuffled_client_ids = np.random.permutation(args.num_users)
                # 2. 基于新索引重新哈希取模分配服务器（保持ServerId = 新索引 % SERVER_NUM）
                new_client_server_map = {}
                for new_idx, client_id in enumerate(shuffled_client_ids):
                    new_client_server_map[client_id] = new_idx % SERVER_NUM
                client_server_map = new_client_server_map
                # print(f"第{epoch+1}轮更新映射（随机打乱+哈希取模）：{client_server_map}")
                print(f"第{epoch + 1}轮更新映射（随机打乱+哈希取模）")

            global_start_time = time.time()
            loss_locals = []
            client_processing_times = {}

            # 选中参与训练的客户端
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            # 多进程训练（传递对应服务器参数）
            with mp.Pool(processes=m) as pool:
                task_args = []
                for idx in idxs_users:
                    s_id = client_server_map[idx]  # 获取客户端当前分配的服务器ID
                    s_param = server_w[s_id]  # 加载该服务器的最新参数
                    task_args.append((args, dataset_train, dict_users[idx], s_param,
                                      client_dataset_sizes[idx], worker_capacity, idx))
                results = pool.starmap(client_train, task_args)

            # 按服务器分组聚合参数（论文多服务器聚合逻辑）
            server_client_params = {s_id: [] for s_id in range(SERVER_NUM)}
            for w, loss, elapsed_time, client_id in results:
                if w is not None and loss is not None:
                    s_id = client_server_map[client_id]
                    server_client_params[s_id].append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                    adjusted_time = elapsed_time / worker_capacity[client_id]
                    client_processing_times[client_id] = adjusted_time
            # print("各个服务器收集到的参数个数：")
            # for s_id in range(SERVER_NUM):
            # print(f"服务器{s_id}: {len(server_client_params[s_id])}")

            # 每个服务器独立执行FedAvg聚合
            for s_id in range(SERVER_NUM):
                params = server_client_params[s_id]
                if len(params) > 0:
                    server_w[s_id] = FedAvg(params)

            # 生成全局测试模型（所有服务器参数平均，保持原有测试逻辑）
            w_glob = copy.deepcopy(init_w)
            for key in w_glob.keys():
                # 1. 强制使用 float32 初始化零张量，避免累加时类型冲突
                w_glob[key] = torch.zeros_like(w_glob[key], dtype=torch.float32)
                
                for s_id in range(SERVER_NUM):
                    # 2. 累加时，确保加进来的 tensor 也是 float32
                    w_glob[key] += server_w[s_id][key].to(torch.float32)
                
                # 3. 计算平均值
                w_glob[key] = torch.div(w_glob[key], SERVER_NUM)
                
                # 4. 【关键修复】将计算完的张量类型，还原回 init_w 对应层原本的数据类型
                # 这样 Float 类型的权重保持为 Float，Long 类型的 num_batches_tracked 会变回 Long
                w_glob[key] = w_glob[key].to(init_w[key].dtype)

            # 测试全局模型
            net_glob.load_state_dict(w_glob)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)



            # 记录结果
            max_client_time = max(client_processing_times.values()) if client_processing_times else 0
            global_elapsed_time = max_client_time

            # 监测测试集准确率，达到目标则提前停止训练
            if acc_test > target_accuracy and epoch >= target_epoch:
                print(f"Epoch{epoch} 测试集准确率 {acc_test:.2f} 达到目标准确率 {target_accuracy}，提前终止训练！")

                # 立即保存当前轮次结果（替代原有的仅保存最后一轮）
                accuracies_per_round.append(acc_test)
                times_per_round.append(global_elapsed_time)

                # 手动释放CUDA缓存
                torch.cuda.empty_cache()

                # 跳出epoch循环，不再继续训练
                break


            times_per_round.append(global_elapsed_time)
            print(
                f"Ai={Ai}, Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, Time: {global_elapsed_time}")
            accuracies_per_round.append(acc_test)
            torch.cuda.empty_cache()

        # 存储结果
        global_round_accuracies.append(accuracies_per_round)
        global_round_times.append(times_per_round)
        test_accuracies.append(accuracies_per_round[-1])

    # 构造保存路径
    save_dir = f'results/RAMFL_{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建
    # 构造文件名（注意添加 save_dir 前缀）
    filename = os.path.join(save_dir, f'{args.model}_isIID_{args.iid}_{Ai}_{p_value}_alpha_{alpha}_Acc_{accuracies_per_round[-1]:.4f}_{formatted_time}_epoch_{epoch_value}.pkl')
    
    
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