import random
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torchvision import datasets, transforms
client_data_size = None
from utils.options import args_parser
import math
import pickle

def get_client_dataset_sizes():
    # 设置相同的随机数种子
    np.random.seed(42)
    random.seed(42)
    # parse args
    args = args_parser()

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)

    else:
        exit('Error: unrecognized dataset')

    # 打印训练集总元素的大小和所占用的存储空间大小
    total_elements = len(dataset_train)
    # 估算占用空间大小
    sample, label = dataset_train[0]
    sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)  # 转换为兆
    label_size = 4 / (1024 * 1024)  # 其标签为整数，整数的字节大小是固定的（通常为 4 字节）
    total_size = math.floor((sample_size + label_size) * total_elements)
    print(f"单个元素的大小: {sample_size + label_size}")
    print(f"训练集总元素数量: {total_elements}")
    print(f"训练集占用存储空间大小: {total_size} 兆")

    # 保存total_size到文件
    with open('total_size.pkl', 'wb') as f:
        pickle.dump(total_size, f)

    return 0
