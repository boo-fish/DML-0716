import random
import matplotlib
import os
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
from dml_确定节点流量分配 import GetFlow
from utils.sampling_dml import mnist_iid, mnist_noniid_dirichlet
from utils.options import args_parser
from models.Update_dml import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
from multiprocessing import Pool
import torch.multiprocessing as mp


def calculate_model_size(net):
    """计算模型参数总大小（单位：兆字节 MB）"""
    total_params = 0
    for param in net.parameters():
        # 每个参数的元素数量 × 每个元素的字节数（32位浮点数=4字节）
        total_params += param.numel() * 4
    # 转换为兆字节（1 MB = 1024×1024 字节）
    return total_params / (1024 * 1024)


def train_client(args, dataset_train, dict_users, client_dataset_sizes, idx, net_glob, worker_capacity):
    client_start_time = time.time()

    if client_dataset_sizes[idx] == 0:
        return None, None, 0, idx

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],
                        client_data_size=client_dataset_sizes[idx])
    w, loss = local.train(net=copy.deepcopy(net_glob))

    client_end_time = time.time()
    client_elapsed_time = client_end_time - client_start_time

    torch.cuda.empty_cache()

    return w, loss, client_elapsed_time, idx


def main():
    args = args_parser()
    epoch_value = args.epochs
    p_value = args.p
    ai = args.ai
    alpha = args.alpha
    offloading_data = GetFlow(p=p_value, ai=ai)
    print("main中的卸载字典", offloading_data)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    with open('Ai_actdata.pkl', 'rb') as f:
        Ai_actdata = pickle.load(f)

    test_accuracies = []
    global_round_accuracies = []
    global_round_times = []
    # 新增：存储每轮模型同步的参数量（MB）
    global_round_model_sizes = []

    for round_idx, (Ai, training_sizes) in enumerate(Ai_actdata):
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

            if args.iid:
                print("iid")
                dict_users = mnist_iid(dataset_train, args.num_users, round_idx, offloading_data)
            else:
                print("non iid")
                dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, offloading_data,
                                                    alpha=alpha)

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

        # 新增：计算初始模型大小（每轮同步的参数量相同，只需计算一次）
        model_size_mb = calculate_model_size(net_glob)
        print(f"模型参数量：{model_size_mb:.2f} MB（每轮同步此大小）")

        w_glob = net_glob.state_dict()

        if args.all_clients:
            w_locals = [w_glob for i in range(args.num_users)]

        client_dataset_sizes = {}
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
        label_size = 4 / (1024 * 1024)
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))

        with open('worker_capacity.pkl', 'rb') as f:
            worker_capacity = pickle.load(f)

        accuracies_per_round = []
        times_per_round = []
        # 新增：记录本轮各全局轮次的模型同步量
        model_sizes_per_round = []

        for epoch in range(args.epochs):
            start_time = time.time()
            w_locals = []
            loss_locals = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            with Pool(processes=len(idxs_users)) as pool:
                results = pool.starmap(train_client, [
                    (args, dataset_train, dict_users, client_dataset_sizes, idx, net_glob, worker_capacity) for idx in
                    idxs_users])

            client_processing_times = {}
            for w, loss, elapsed_time, client_id in results:
                if w is not None and loss is not None:
                    w_locals.append(copy.deepcopy(w))
                    loss_locals.append(copy.deepcopy(loss))
                    adjusted_time = elapsed_time / worker_capacity[client_id]
                    client_processing_times[client_id] = adjusted_time

            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)

            max_client_time = max(client_processing_times.values()) if client_processing_times else 0
            end_time = time.time()
            adjusted_global_time = max_client_time

            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            accuracies_per_round.append(acc_test)
            times_per_round.append(adjusted_global_time)
            # 新增：记录本轮模型同步量（与初始模型大小相同）
            model_sizes_per_round.append(model_size_mb)

            print(
                f"Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, "
                f"Global training time: {adjusted_global_time:.2f}s, "
                f"Model sync size: {model_size_mb:.2f} MB"  # 新增：打印同步量
            )
            torch.cuda.empty_cache()

        global_round_accuracies.append(accuracies_per_round)
        global_round_times.append(times_per_round)
        # 新增：保存本轮所有全局轮次的模型同步量
        global_round_model_sizes.append(model_sizes_per_round)
        test_accuracies.append(accuracies_per_round[-1])

    save_dir = 'saving/temp'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir,
                            f'Ai_{Ai}_P_{p_value}_epoch_{epoch_value}_is_iid_{args.iid}_DML_alpha_{alpha}_canshu.pkl')
    data_to_save = {
        'ais': [x[0] for x in Ai_actdata],
        'test_accuracies': test_accuracies,
        'global_round_accuracies': global_round_accuracies,
        'global_round_times': global_round_times,
        'global_round_model_sizes': global_round_model_sizes  # 新增：保存模型同步量数据
    }
    with open(filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print("成功保存数据pkl文件（含模型同步量）")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()