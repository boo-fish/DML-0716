#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import math
from torch.optim.lr_scheduler import StepLR # 新增：导入学习率调度器


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

        if client_data_size is not None and total_size > client_data_size:
            # 根据容量计算需要保留的数据个数
            num_keep = int(client_data_size / (sample_size + label_size))
            idxs = idxs[:num_keep]

        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        # 根据实际训练轮数调整step_size，确保学习率能够正常衰减
        # if self.args.local_ep <= 5:
        #     step_size = 2  # 如果总轮数≤5，每2个epoch衰减一次
        # elif self.args.local_ep <= 10:
        #     step_size = 3  # 如果总轮数≤10，每3个epoch衰减一次
        # else:
        #     step_size = 5  # 默认每5个epoch衰减一次

        # ========== 关键修改1：初始化学习率调度器（新增） ==========
        # 策略1：StepLR（每step_size个epoch，学习率乘以gamma）
        # scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)  # 每5个epoch衰减为1/10

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            # ========== 关键修改2：每个epoch后更新学习率（新增） ==========
            # scheduler.step()


        # print(f"此时的学习率为: {optimizer.param_groups[0]['lr']}")


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)