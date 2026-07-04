# import random
# import matplotlib
# matplotlib.use('Agg')
# import numpy as np
# from torchvision import datasets, transforms
# client_data_size = None
# # from utils.options import args_parser
# import math
# import pickle

# def get_client_dataset_sizes(data=None):
#     # 设置相同的随机数种子
#     np.random.seed(42)
#     random.seed(42)
#     # parse args
#     # args = args_parser()

#     # load dataset and split users
#     if data == 'mnist':
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)

#     elif data == 'cifar10':
#         trans_cifar = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=False, transform=trans_cifar)

#     # 打印训练集总元素的大小和所占用的存储空间大小
#     total_elements = len(dataset_train)
#     # 估算占用空间大小
#     sample, label = dataset_train[0]
#     sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)  # 转换为兆
#     label_size = 4 / (1024 * 1024)  # 其标签为整数，整数的字节大小是固定的（通常为 4 字节）
#     total_size = math.floor((sample_size + label_size) * total_elements)
#     print(f"单个元素的大小: {sample_size + label_size}")
#     print(f"训练集总元素数量: {total_elements}")
#     print(f"训练集占用存储空间大小: {total_size} 兆")

#     # 保存total_size到文件
#     with open('total_size.pkl', 'wb') as f:
#         pickle.dump(total_size, f)

#     return 0

# get_client_dataset_sizes(data='cifar10')
# get_client_dataset_sizes(data='mnist')


import random
import numpy as np
from torchvision import datasets, transforms
import math
import pickle

def get_client_dataset_sizes(data=None, download=False):
    """
    获取数据集大小
    :param data: 数据集名称 (mnist/cifar10/cifar100)
    :param download: 是否下载数据集（False 则仅估算，True 则实际下载后计算）
    :return: 0
    """
    # 设置相同的随机数种子
    np.random.seed(42)
    random.seed(42)

    # 预定义数据集规格（用于不下载时估算）
    dataset_specs = {
        'mnist': {'size': (28, 28, 1), 'train_samples': 60000, 'test_samples': 10000},
        'cifar10': {'size': (32, 32, 3), 'train_samples': 50000, 'test_samples': 10000},
        'cifar100': {'size': (32, 32, 3), 'train_samples': 50000, 'test_samples': 10000}
    }

    if data not in dataset_specs:
        print(f"不支持的数据集: {data}，支持的数据集有: {list(dataset_specs.keys())}")
        return -1

    # 1. 先通过规格估算大小（无需下载）
    specs = dataset_specs[data]
    # 计算单张图片的字节数 (32*32*3=3072 字节 for cifar)
    pixel_count = specs['size'][0] * specs['size'][1] * specs['size'][2]
    sample_size_mb = (pixel_count * 1) / (1024 * 1024)  # 每个像素1字节（8位）
    label_size_mb = 4 / (1024 * 1024)  # 标签4字节
    single_element_size_mb = sample_size_mb + label_size_mb
    train_total_size_mb = math.floor(single_element_size_mb * specs['train_samples'])
    
    print(f"\n=== {data.upper()} 估算大小（未下载）===")
    print(f"单张图片像素数: {pixel_count}")
    print(f"单个元素(图片+标签)大小: {single_element_size_mb:.6f} MB")
    print(f"训练集总样本数: {specs['train_samples']}")
    print(f"训练集估算总大小: {train_total_size_mb} MB")

    # 2. 如果选择下载，则实际加载数据集并计算
    if download:
        print(f"\n=== {data.upper()} 实际大小（已下载）===")
        if data == 'mnist':
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans)
        elif data == 'cifar10':
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans)
        elif data == 'cifar100':
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans)
        
        # 实际计算大小
        total_elements = len(dataset_train)
        sample, label = dataset_train[0]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
        label_size = 4 / (1024 * 1024)
        total_size = math.floor((sample_size + label_size) * total_elements)
        
        print(f"单个元素的实际大小: {sample_size + label_size:.6f} MB")
        print(f"训练集总元素数量: {total_elements}")
        print(f"训练集实际占用存储空间大小: {total_size} MB")

        

    return 0

# 仅估算CIFAR100大小（不下载）
get_client_dataset_sizes(data='cifar10', download=False)

# 如需验证，可取消下面注释（会实际下载数据集）
# get_client_dataset_sizes(data='cifar100', download=True)

# 也可以查看MNIST和CIFAR10的估算大小
get_client_dataset_sizes(data='mnist', download=False)
get_client_dataset_sizes(data='cifar100', download=False)