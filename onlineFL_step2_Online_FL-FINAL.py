import random
from venv import logger
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
from utils.sampling_dml import mnist_iid, mnist_noniid_dirichlet, mnist_iid_by_training_sizes
from utils.options import args_parser
from models.Update_dml import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, VGG11
from models.Fed import FedAvg
from models.test import test_img
import math
import pickle
from multiprocessing import Pool
import torch.multiprocessing as mp  # 导入多进程模块，用于并行训练
from datetime import datetime
from onlineFL_step1_全参数固定的仿真模板代码_用于获取每一time的实际训练量_FINAL import get_real_flow_mb_in_t_time

# 获取当前时间
now = datetime.now()

# 按所需格式转换为字符串
formatted_time = now.strftime("%m-%d-%H-%M-%S")



def get_size_in_mb(state_dict_list):
    total_size = 0
    for state_dict in state_dict_list:
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor):
                total_size += tensor.element_size() * tensor.numel()
    return total_size / (1024 ** 2)  # Bytes -> MB


def get_gradient_size_in_mb(global_w, local_w_list):
    """
    计算客户端梯度的总内存大小（单位：MB）
    :param global_w: 全局模型初始权重（state_dict）
    :param local_w_list: 各客户端本地训练后的权重列表（list of state_dict）
    :return: 所有客户端梯度的总内存大小（MB）
    """
    total_gradient_size = 0
    for local_w in local_w_list:
        if local_w is None:
            continue
        # 遍历每一层参数，计算梯度（本地权重 - 全局初始权重）
        for key in global_w.keys():
            global_tensor = global_w[key].cpu().contiguous()  # 转移到CPU并连续存储
            local_tensor = local_w[key].cpu().contiguous()
            gradient_tensor = local_tensor - global_tensor  # 得到梯度张量

            # 计算该梯度张量的字节大小
            element_size = gradient_tensor.element_size()  # 每个元素的字节数
            num_elements = gradient_tensor.numel()  # 元素总数
            total_gradient_size += element_size * num_elements

    return total_gradient_size / (1024 ** 2)  # 转换为MB

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


    torch.cuda.empty_cache()


    args = args_parser()
    # epoch_value = args.epochs
    p_value = args.p
    ai = args.ai
    alpha = args.alpha
    total_size = args.total_mb
    total_slots = args.total_slots
    T = args.T

    ### 1.先准备好每个时隙的工作节点和实际训练量 [slots.4]
    time_slot_results = get_real_flow_mb_in_t_time(total_slots, T)
    print(f"step1 获取到{len(time_slot_results)}个时隙的工作节点以及实际训练量")


    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 设定数据集对应的目标准确率
    if args.dataset == 'mnist':
        target_accuracy = 92.0  # mnist数据集目标准确率92%
        target_epoch = 50
    elif args.dataset == 'cifar10':
        target_accuracy = 55.0  # cifar10数据集目标准确率55%
        target_epoch = 100
    else:
        target_accuracy = 0.0  # 未知数据集默认值
        print(f"警告：未识别的数据集 {args.dataset}，未设置目标准确率")

    print(f"目标准确率：{target_accuracy}，目标epoch：{target_epoch}")


    # 存储测试集准确率
    test_accuracies = []
    # 存储每个 Ai 下的全局轮次准确率
    global_round_accuracies = []
    # 存储每个 Ai 下的全局轮次训练时间
    global_round_times = []
    
    ####  实验设定数
    round_idx = 0
    
    
    if args.iid:
        print("iid")
    else:
        print("non-iid")
    
    ### 2先加载数据集（在每一个time时隙中再进行客户端数据划分）
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=False, transform=trans_mnist)
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar10', train=False, download=False, transform=trans_cifar)

        

    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape


    ### 3模型构建（训练前加载一次即可）
    if args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)

    elif args.model == 'vgg11' and args.dataset == 'cifar10':
        net_glob = VGG11(args=args).to(args.device)

    elif args.model == 'vgg11' and args.dataset == 'mnist':
        net_glob = VGG11(args=args).to(args.device)

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


    ### 不选K个设备的全聚合，暂时用不到
    # if args.all_clients:
    #     # print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]

    
    



    # 4开始进行全局训练
    accuracies_per_round = []
    times_per_round = []
    
    for epoch in range(args.total_slots):
        start_time = time.time()  # 记录开始时间
        
        
        
        
        ## 5每个time时隙中进行客户端数据划分（time时隙内操作）

        selected_idx_in_time = time_slot_results[epoch][1] # [K]
        offloading_data = time_slot_results[epoch][2]  # int
        worker_capacity = time_slot_results[epoch][3]  # [n=50]
        Ai_actdata = time_slot_results[epoch][4]  # [n=50]
        training_data = time_slot_results[epoch][4]  # [n=50]
        Ai_actdata = [[2,Ai_actdata]]  # 添加一个Ai值，用于后续计算
        
 

        # print("main中Ai的len", len(Ai_actdata))
        # print(f"main中Ai:{Ai_actdata}")
        # print(f"main中worker_capacity:{worker_capacity}")
        # print(f"main中selected_idx_in_time:{selected_idx_in_time}")
        # print(f"main中training_data:{training_data}")
        
        if args.iid:
            # 将 round_idx 传递给 mnist_iid 函数
            dict_users = mnist_iid_by_training_sizes(dataset_train, args.num_users, round_idx, offloading_data,
                                    Ai_actdata=Ai_actdata, total_size=total_size)

        else:
            # Dirichlet分布实现Non-IID  alpha控制 non-IID 程度
            dict_users = mnist_noniid_dirichlet(dataset_train, args.num_users, round_idx, offloading_data,
                                                alpha=alpha,
                                                Ai_actdata=Ai_actdata, total_size=total_size)
        
        
        
        ### 6计算每个设备上在该分到的数据大小（time时隙内操作）
        client_dataset_sizes = {}
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)  # 转换为兆
        label_size = 4 / (1024 * 1024)  # 其标签为整数，整数的字节大小是固定的（通常为 4 字节）
        for client_idx, client_idxs in dict_users.items():
            client_dataset_sizes[client_idx] = math.floor(len(client_idxs) * (sample_size + label_size))
        
        
        
        w_locals = []
        loss_locals = []
  
        ### 7传递onlineFL选中的当前time的工作节点
        idxs_users = selected_idx_in_time
        # print(f"Round {epoch + 1}, 工作节点：{idxs_users}")

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
        gradient_mb = get_gradient_size_in_mb(w_glob,w_locals)
        print(f"  w_locals: {size_mb:.2f} MB")
        print(f"  gradient: {gradient_mb:.2f} MB")


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

        # 监测测试集准确率，达到目标则提前停止训练
        if acc_test > target_accuracy and epoch >= target_epoch:
            print(f"Epoch{epoch}  测试集准确率 {acc_test:.2f} 达到目标准确率 {target_accuracy}，提前终止训练！")

            # 立即保存当前轮次结果（替代原有的仅保存最后一轮）
            accuracies_per_round.append(acc_test)
            times_per_round.append(adjusted_global_time)

            # 手动释放CUDA缓存
            torch.cuda.empty_cache()

            # 跳出epoch循环，不再继续训练
            break


        accuracies_per_round.append(acc_test)
        times_per_round.append(adjusted_global_time)

        print(
            f"Round {epoch + 1}, Test Accuracy: {acc_test}, Loss: {loss_test}, Global training time: {adjusted_global_time}\n")
        # 手动释放 CUDA 缓存
        torch.cuda.empty_cache()

    # 存储每个 Ai 下的全局轮次准确率
    global_round_accuracies.append(accuracies_per_round)
    # 存储每个 Ai 下的全局轮次训练时间
    global_round_times.append(times_per_round)
    # 存储测试集准确率
    test_accuracies.append(accuracies_per_round[-1])

    # 构造保存路径
    save_dir = f'results/online_{T}'
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在则创建
    # 构造文件名（注意添加 save_dir 前缀）
    
    if args.iid:
        type = 'IID'
    else:
        type = 'NOIID'
    filename = os.path.join(save_dir, f'{args.dataset}_{args.model}_{type}_P_{p_value}_Alpha_{alpha}_Epochs_{total_slots}_Final_Acc_{accuracies_per_round[-1]:.4f}_{formatted_time}.pkl')

    # 保存数据
    data_to_save = {
        'ais': 2,
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

