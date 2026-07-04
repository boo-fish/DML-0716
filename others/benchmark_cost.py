import numpy as np

class LocalTraining:
    def __init__(self, env, Ai):
        self.env = env
        self.Ai = Ai

    def run_local_training(self):
        total_throughput = 0
        total_cost = 0
        total_sync_cost = 0
        total_training_cost = 0
        total_tidu = 0
        # 初始化参数服务器已使用的容量
        server_used_capacity = [0] * self.env['num_servers']

        for i in range(self.env['num_workers']):
            # 每个工作节点实际可训练的数据量，受限于工作节点容量
            # trainable_data = min(self.env['data_arrival'][i], self.env['worker_capacity'][i])
            trainable_data = self.Ai
            # 计算训练成本
            training_cost = trainable_data * self.env['training_cost'][i]
            total_training_cost += training_cost
            # 随机选择三个参数服务器
            connected_servers = np.random.choice(self.env['num_servers'], 1, replace=True)

            # 从这三个参数服务器中选择链路成本最低的
            min_cost_server = min(connected_servers, key=lambda s: self.env['transmission_cost_i_k'][i][s])

            # 判断该服务器是否有足够的容量
            scaled_trainable_data = trainable_data * self.env['alpha']
            total_tidu += scaled_trainable_data
            if server_used_capacity[min_cost_server] + scaled_trainable_data > self.env['server_capacity'][min_cost_server]:
                continue  # 如果没有足够容量，跳过此工作节点

            # 工人节点数据缩放后传输到参数服务器，计算传输成本
            transmission_cost = scaled_trainable_data * self.env['transmission_cost_i_k'][i][min_cost_server]

            # 更新参数服务器已使用的容量
            server_used_capacity[min_cost_server] += scaled_trainable_data

            # 参数服务器数据缩放后同步到虚拟终端节点，计算同步成本
            scaled_sync_data = scaled_trainable_data
            sync_cost = scaled_sync_data * self.env['sync_cost'][min_cost_server]

            total_cost += training_cost + transmission_cost + sync_cost

            total_sync_cost = total_sync_cost + sync_cost
            # 新的吞吐量计算方式
            throughput = scaled_trainable_data / self.env['alpha']
            total_throughput += throughput

        # 计算成本效率
        cost_efficiency = total_throughput / total_cost if total_cost != 0 else float('inf')
        return total_throughput, total_cost, cost_efficiency,total_training_cost,total_tidu,total_sync_cost