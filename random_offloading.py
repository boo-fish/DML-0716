import numpy as np


class RandomOffloading:
    def __init__(self, env, Ai):
        self.env = env
        self.Ai = Ai

    def run_random_offloading(self):
        total_throughput = 0
        total_cost = 0

        # 初始化每个工作节点实际处理的数据量
        processed_data = self.env['data_arrival'].copy()

        # 进行随机卸载
        for i in range(self.env['num_workers']):
            while processed_data[i] > self.env['worker_capacity'][i]:
                excess_data = processed_data[i] - self.env['worker_capacity'][i]
                # 找到与该节点相连的相邻节点
                neighbors = [j for j in range(self.env['num_workers']) if (i, j) in self.env['connections']]
                available_neighbors = [n for n in neighbors if self.env['worker_capacity'][n] - processed_data[n] > 0]
                if not available_neighbors:
                    # 如果没有可用的相邻节点，退出循环
                    break
                while excess_data > 0 and available_neighbors:
                    # 随机选择一个相邻节点进行卸载
                    target_worker = np.random.choice(available_neighbors)
                    # 考虑链路容量限制
                    offload_data = min(excess_data, self.env['link_capacity'][i][target_worker])
                    # 目标节点剩余容量
                    target_remaining_capacity = self.env['worker_capacity'][target_worker] - processed_data[
                        target_worker]
                    offload_data = min(offload_data, target_remaining_capacity)

                    if offload_data > 0:
                        processed_data[i] -= offload_data
                        processed_data[target_worker] += offload_data
                        # 计算卸载的传输成本
                        total_cost += offload_data * self.env['transmission_cost'][i][target_worker]
                        excess_data -= offload_data

                    # 更新可用邻居节点
                    available_neighbors = [n for n in neighbors if
                                           self.env['worker_capacity'][n] - processed_data[n] > 0]
                    if not available_neighbors:
                        # 如果没有可用的相邻节点，退出内层循环
                        break

        # 初始化参数服务器已使用的容量
        server_used_capacity = [0] * self.env['num_servers']

        # 计算训练和同步成本
        for i in range(self.env['num_workers']):
            trainable_data = min(processed_data[i], self.env['worker_capacity'][i])

            # 计算训练成本
            training_cost = trainable_data * self.env['training_cost'][i]

            # 随机选择三个参数服务器
            connected_servers = np.random.choice(self.env['num_servers'], 1, replace=True)
            # 从这三个参数服务器中选择链路成本最低的
            min_cost_server = min(connected_servers, key=lambda s: self.env['transmission_cost_i_k'][i][s])

            # 工人节点数据缩放后传输到参数服务器
            scaled_trainable_data = trainable_data * self.env['alpha']

            # 判断参数服务器是否有足够的容量
            if server_used_capacity[min_cost_server] + scaled_trainable_data <= self.env['server_capacity'][
                min_cost_server]:
                # 有足够容量，全部接收
                received_data = scaled_trainable_data
                server_used_capacity[min_cost_server] += received_data
            else:
                # 容量不足，只接收剩余容量部分，其余舍弃
                received_data = self.env['server_capacity'][min_cost_server] - server_used_capacity[min_cost_server]
                server_used_capacity[min_cost_server] = self.env['server_capacity'][min_cost_server]

            # 计算传输成本
            transmission_cost = received_data * self.env['transmission_cost_i_k'][i][min_cost_server]

            # 参数服务器数据同步到虚拟终端节点，计算同步成本
            scaled_sync_data = received_data
            sync_cost = scaled_sync_data * self.env['sync_cost'][min_cost_server]

            total_cost += training_cost + transmission_cost
            # 新的吞吐量计算方式
            throughput = received_data / self.env['alpha']
            total_throughput += throughput

        # 计算成本效率
        cost_efficiency = total_throughput / total_cost if total_cost != 0 else float('inf')
        return total_throughput, total_cost, cost_efficiency
