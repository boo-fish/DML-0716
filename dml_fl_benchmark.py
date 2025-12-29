import matplotlib
import os
from utils.sampling_benchmark import mnist_noniid_dirichlet
matplotlib.use('TkAgg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.network_environment import NetworkEnvironment
from utils.sampling_benchmark import mnist_iid
from utils.options import args_parser
from models.Update_benchmark import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar,VGG11
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
import torch.multiprocessing as mp  # 导入多进程模块，用于并行训练
from utils.get_offload_dict_dml import GetFlow
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")


def client_train(args, dataset, idxs, w_glob, client_data_size, worker_capacity, client_id):
    """客户端训练函数，考虑客户端容量"""
    client_start_time = time.time()
    idxs = list(idxs)
    data_size = len(idxs)  # 客户端本地数据量

    # 初始化w为全局模型参数（关键：确保w始终有值）
    w = w_glob.copy()
    loss = 0.0  # 初始化损失

    # 若客户端无数据，直接返回全局模型和默认损失
    if data_size == 0:
        client_end_time = time.time()
        client_elapsed_time = client_end_time - client_start_time
        return w, loss, client_elapsed_time, client_id

    # 创建模型（先在CPU上创建，稍后移到GPU）
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net = CNNMnist(args=args)
    elif args.model == 'vgg11' and args.dataset == 'cifar10':
        net = VGG11(args=args)
    elif args.model == 'vgg11' and args.dataset == 'mnist':
        net = VGG11(args=args)
    elif args.model == 'mlp':
        img_size = dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        net = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)
    else:
        raise ValueError('Error: unrecognized model')

    # 加载全局权重到模型
    net.load_state_dict(w_glob)

    # 将模型移到设备（GPU或CPU）
    net.to(args.device)

    # 计算客户端容量对应的样本数
    sample, label = dataset[0]
    sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
    label_size = 4 / (1024 * 1024)
    num_samples_in_capacity = int(worker_capacity[client_id] / (sample_size + label_size))

    start_idx = 0
    batch_num = 0

    # 分批训练
    while start_idx < data_size:
        # 计算当前批次的结束索引
        remaining_samples = data_size - start_idx
        end_idx = start_idx + min(num_samples_in_capacity, remaining_samples)
        local_idxs = idxs[start_idx:end_idx]

        # 创建LocalUpdate实例并训练
        local = LocalUpdate(args=args, dataset=dataset, idxs=local_idxs,
                            client_data_size=num_samples_in_capacity)

        # 执行本地训练，更新w和loss
        w, loss = local.train(net=net)  # 覆盖初始化的w和loss

        # 清理当前批次的梯度
        net.zero_grad()

        # 重新加载权重到模型（为了内存优化）
        if torch.cuda.is_available() and args.device.type == 'cuda':
            # 如果使用GPU，先移到CPU减少GPU内存占用
            net.cpu()
            net.load_state_dict(w)
            net.to(args.device)
        else:
            # 如果使用CPU，直接加载
            net.load_state_dict(w)

        # 清理CUDA缓存（如果是GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新索引
        start_idx = end_idx
        batch_num += 1

    client_end_time = time.time()
    client_elapsed_time = client_end_time - client_start_time

    # 最终清理
    del net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return w, loss, client_elapsed_time, client_id

def main():
    args = args_parser()
    alpha = args.alpha
    p_value = args.p
    ai = args.ai
    epoch_value = args.epochs
    total_size = args.total_mb

    # 修改：调用GetFlow并接收4个返回值
    offloading_data, worker_capacity, Ai_actdata, training_data = GetFlow(p=p_value, ai=ai)
    print("main中的卸载字典", offloading_data)
    print("main中Ai的len", len(Ai_actdata))
    print("main中Ai", Ai_actdata)


    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # 存储测试集准确率
    test_accuracies = []
    # 存储每个 Ai 下的全局轮次准确率
    global_round_accuracies = []
    # 存储每个 Ai 下每一轮全局训练的时间
    global_round_times = []

    # 使用 Ai_actdata.pkl 文件中列表元素的个数，进行限定循环轮数
    for round_idx, (Ai, training_sizes) in enumerate(Ai_actdata):

        # load dataset and split users
        if args.dataset =='mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            if args.iid:
                print("iid")
                # 将 round_idx 传递给 mnist_iid 函数
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx,Ai_actdata=Ai_actdata, total_size=total_size)

            else:
                print("non-iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, alpha=alpha,Ai_actdata=Ai_actdata, total_size=total_size)


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


        else:

            exit('Error: unrecognized dataset')



        # 打印训练集总元素的大小和所占用的存储空间大小
        total_elements = len(dataset_train)
        # 估算占用空间大小
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)  # 转换为兆
        label_size = 4 / (1024 * 1024)  # 其标签为整数，整数的字节大小是固定的（通常为 4 字节）

        img_size = dataset_train[0][0].shape

        # build model
        if args.model == 'cnn' and args.dataset == 'cifar10':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset =='mnist':
            net_glob = CNNMnist(args=args).to(args.device)

        elif args.model == 'vgg11' and args.dataset == 'cifar10':
            net_glob = VGG11(args=args).to(args.device)

        elif args.model == 'vgg11' and args.dataset == 'mnist':
            net_glob = VGG11(args=args).to(args.device)
        elif args.model =='mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()

        if args.all_clients:
            # print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

        client_dataset_sizes = {}
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))



        # 全局训练轮次
        accuracies_per_round = []
        times_per_round = []
        for epoch in range(args.epochs):
            # 记录全局训练开始时间
            global_start_time = time.time()
            w_locals = []
            loss_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            with mp.Pool(processes=m) as pool:  # 使用 with 语句确保资源正确释放
                results = pool.starmap(client_train, [(args, dataset_train, dict_users[idx], w_glob,
                                                       client_dataset_sizes[idx], worker_capacity, idx)
                                                      for idx in idxs_users])

            # 存储每个客户端的处理时间
            client_processing_times = {}
            for w, loss, elapsed_time, client_id in results:
                if w is not None and loss is not None:
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                    # 根据客户端计算能力调整训练时间
                    adjusted_time = elapsed_time / worker_capacity[client_id]
                    client_processing_times[client_id] = adjusted_time

            # 聚合权重
            w_glob = FedAvg(w_locals)

            # 更新全局模型
            net_glob.load_state_dict(w_glob)

            # 找出处理时间最长的客户端，作为本轮全局训练的时间
            max_client_time = max(client_processing_times.values()) if client_processing_times else 0

            # 记录全局训练结束时间
            global_end_time = time.time()
            # 计算全局训练时间
            global_elapsed_time = global_end_time - global_start_time
            # 使用调整后的最大客户端处理时间作为全局训练时间
            global_elapsed_time = max_client_time

            # 测试全局模型
            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            times_per_round.append(global_elapsed_time)
            print(f"Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, Global training time: {global_elapsed_time}")
            # 添加测试准确率到列表
            accuracies_per_round.append(acc_test)
            # 手动释放 CUDA 缓存
            torch.cuda.empty_cache()

        # 存储每个 Ai 下的全局轮次准确率
        global_round_accuracies.append(accuracies_per_round)
        # 存储每个 Ai 下每一轮全局训练的时间
        global_round_times.append(times_per_round)
        # 存储测试集准确率
        test_accuracies.append(accuracies_per_round[-1])

    # 构造保存路径
    save_dir = 'results/benchmark'
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建
    # 构造文件名（注意添加 save_dir 前缀）
    filename = os.path.join(save_dir, f'{args.dataset}_{args.model}_{Ai}_{p_value}_epoch_{epoch_value}_isIID_{args.iid}_alpha_{alpha}_Final_Acc_{accuracies_per_round[-1]:.4f}_{formatted_time}.pkl')

    # 保存数据
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
    mp.set_start_method('spawn')  # 设置多进程启动方式
    final_results = main()  # 接收返回的结果
    print("训练完成，最终结果：", final_results)