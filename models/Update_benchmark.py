#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import sys
import math


class DataSizeCalculator:
    def __init__(self, sample_size, label_size):
        self.sample_size = sample_size
        self.label_size = label_size

    def calculate_total_size(self, client_idxs):
        return math.floor(len(client_idxs) * (self.sample_size + self.label_size))


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, client_data_size=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # 计算训练数据占用的存储空间
        sample = dataset[0][0]
        label = dataset[0][1]
        sample_size = sample.element_size() * sample.nelement() / (1024 * 1024)
        label_size = 4 / (1024 * 1024)
        calculator = DataSizeCalculator(sample_size, label_size)
        total_size = calculator.calculate_total_size(idxs)

        # if client_data_size is not None and total_size > client_data_size:
        #     # 根据容量计算需要保留的数据个数
        #     num_keep = int(client_data_size / (sample_size + label_size))
        #     idxs = idxs[:num_keep]

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    # 在models/Update_benchmark.py的train方法中添加
    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr,
                                    momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                # 反向传播
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                # 定期清理内存
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # 返回前清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)