import random
import matplotlib
import os
# 修改后端设置
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from utils.get_offload_dict_dml import GetFlow
from utils.sampling_dml import mnist_iid, mnist_noniid_dirichlet
from utils.options import args_parser
from models.Update_dml import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
from multiprocessing import Pool
import torch.multiprocessing as mp  # 导入多进程模块，用于并行训练
from datetime import datetime

# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")




def get_size_in_mb(state_dict_list):
    total_size = 0
    for state_dict in state_dict_list:
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.element_size() * tensor.numel()
    return total_size / (1024 ** 2)  # Bytes -> MB

def train_client(args, dataset_train, dict_users, client_dataset_sizes, idx, net_glob, worker_capacity):
    # 记录客户端训练开始时间
    client_start_time = time.time()

    # 检查客户端数据大小是否为0
    if client_dataset_sizes[idx] == 0:
        return None, None, 0, idx

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                        client_data_size=client_dataset_sizes[idx])
    w, loss = local.train(net=copy.deepcopy(net_glob))

    # 记录客户端训练结束时间
    client_end_time = time.time()
    # 计算客户端训练时间
    client_elapsed_time = client_end_time - client_start_time

    # 手动释放 CUDA 缓存
    torch.cuda.empty_cache()

    return w, loss, client_elapsed_time, idx


def main():
    args = args_parser()
    epoch_value = args.epochs
    p_value = args.p
    ai = args.ai
    alpha = args.alpha
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
    # 存储每个 Ai 下的全局轮次训练时间
    global_round_times = []

    # 使用Ai_actdata列表元素的个数，进行限定循环轮数
    for round_idx, (Ai, training_sizes) in enumerate(Ai_actdata):
        # load dataset and split users
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            if args.iid:
                print("iid")
                print(f"args.iid:{args.iid}")
                # 将 round_idx 传递给 mnist_iid 函数
                # =========关键修改：传递Ai_actdata和total_size参数==========
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx,offloading_data,
                          Ai_actdata=Ai_actdata, total_size=total_size)

            else:
                print("non iid")
                print(f"args.iid:{args.iid}")
                # Dirichlet分布实现Non-IID  alpha控制 non-IID 程度
                # =========关键修改：传递Ai_actdata和total_size参数==========
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, offloading_data,alpha=alpha,
                          Ai_actdata=Ai_actdata, total_size=total_size)
                # visualize_dirichlet_distribution(dataset_train, dict_users,10,args.num_users)

        img_size = dataset_train[0][0].shape

        # build model
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

        # copy weights
        w_glob = net_glob.state_dict()

        if args.all_clients:
            # print("Aggregation over all clients")
            w_locals = [w_glob for i in range(args.num_users)]

        client_dataset_sizes = {}
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)  # 转换为兆
        label_size = 4 / (1024 * 1024)  # 其标签为整数，整数的字节大小是固定的（通常为 4 字节）
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))



        # 全局训练轮次
        accuracies_per_round = []
        times_per_round = []
        for epoch in range(args.epochs):
            start_time = time.time()  # 记录开始时间
            w_locals = []
            loss_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            with Pool(processes=len(idxs_users)) as pool:
                results = pool.starmap(train_client, [
                    (args, dataset_train, dict_users, client_dataset_sizes, idx, net_glob, worker_capacity) for idx in
                    idxs_users])

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

            # 输出 w_locals 的总大小
            size_mb = get_size_in_mb(w_locals)
            print(f"Total memory used by w_locals: {size_mb:.2f} MB")


            # 更新全局模型
            net_glob.load_state_dict(w_glob)

            # 找出处理时间最长的客户端，作为本轮全局训练的时间
            max_client_time = max(client_processing_times.values()) if client_processing_times else 0

            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算训练时间
            # 使用调整后的最大客户端处理时间作为全局训练时间
            adjusted_global_time = max_client_time

            # 测试全局模型
            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            accuracies_per_round.append(acc_test)
            times_per_round.append(adjusted_global_time)

            print(
                f"Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, Global training time: {adjusted_global_time}")
            # 手动释放 CUDA 缓存
            torch.cuda.empty_cache()

        # 存储每个 Ai 下的全局轮次准确率
        global_round_accuracies.append(accuracies_per_round)
        # 存储每个 Ai 下的全局轮次训练时间
        global_round_times.append(times_per_round)
        # 存储测试集准确率
        test_accuracies.append(accuracies_per_round[-1])

    # 构造保存路径
    save_dir = 'results/DML'
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建
    # 构造文件名（注意添加 save_dir 前缀）
    filename = os.path.join(save_dir,f'Ai_{Ai}_P_{p_value}_epoch_{epoch_value}_is_iid_{args.iid}_local_alpha_{alpha}_Final_Acc_{accuracies_per_round[-1]:.4f}_{formatted_time}.pkl')

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

