from utils.options import args_parser
import numpy as np


# 获取命令行参数
args = args_parser()
num_clients = args.num_users

# # # 设置随机种子
# RANDOM_SEED = 43
# np.random.seed(RANDOM_SEED)

# 导入 NetworkEnvironment 类
class NetworkEnvironment:
    def __init__(self, num_workers, num_servers, num_clients,link_capacity_range, worker_capacity_range, server_capacity_range,
                 data_arrival_range, alpha):
        self.num_workers = num_workers
        self.num_servers = num_servers
        self.num_clients = args.num_users
        self.link_capacity = np.random.randint(*link_capacity_range, (num_workers, num_workers))
        self.link_capacity_i_k = self._generate_link_capacity_i_k(link_capacity_range, num_workers, num_servers)
        self.worker_capacity = self._generate_worker_capacity(worker_capacity_range, num_workers)
        self.server_capacity = self._generate_server_capacity(server_capacity_range, num_servers)
        self.client_capacity = self._generate_client_capacity(worker_capacity_range, num_clients)
        self.transmission_cost = self._generate_transmission_cost(num_workers)#最小平均值种子44257
        self.transmission_cost_i_k = self._generate_transmission_cost_i_k(num_workers, num_servers)
        self.training_cost = self._generate_training_cost(num_workers)
        self.sync_cost = self._generate_sync_cost(num_servers)
        self.data_arrival = self._generate_data_arrival(data_arrival_range, num_workers)
        self.alpha = alpha



        # 确保工作节点之间的连接比例是 0.3
        num_connections = int(num_workers * (num_workers - 1) * 0.15)
        connections = []
        while len(connections) < num_connections:
            i, j = np.random.choice(num_workers, 2, replace=False)
            if (i, j) not in connections and (j, i) not in connections:
                connections.append((i, j))
                connections.append((j, i))
        self.connections = connections

        num_connections_fl = int(num_workers * (num_workers - 1) * 0.15)
        connections_fl = []
        while len(connections_fl) < num_connections_fl:
            i, j = np.random.choice(num_workers, 2, replace=False)
            if (i, j) not in connections_fl and (j, i) not in connections_fl:
                connections_fl.append((i, j))
                connections_fl.append((j, i))
        self.connections_fl = connections_fl

        # 记录每个工人节点 result 与哪些参数服务器相连
        self.worker_server_connections = []
        for i in range(self.num_workers):
            connected_servers = np.random.choice(self.num_servers, 1, replace=True)
            self.worker_server_connections.append(connected_servers)

    def _generate_link_capacity(self, link_capacity_range, num_workers):
        np.random.seed(4)
        return np.random.randint(*link_capacity_range, (num_workers, num_workers))

    def _generate_link_capacity_i_k(self, link_capacity_range, num_workers, num_servers):
        np.random.seed(4)
        return np.random.randint(*link_capacity_range, (num_workers, num_servers))

    def _generate_worker_capacity(self, worker_capacity_range, num_workers):
        # 固定的左端点值和右端点值
        left_value = worker_capacity_range[0]
        right_value = worker_capacity_range[1]

        # 设置随机种子
        np.random.seed(4)

        # 生成中间 num_workers-2 个随机数
        middle_values = np.random.randint(*worker_capacity_range, num_workers - 2)

        # 组合结果：左端点值 + 中间随机值 + 右端点值
        return np.array([left_value] + list(middle_values) + [right_value])

    def _generate_client_capacity(self, worker_capacity_range, num_clients):
        np.random.seed(4)
        return np.random.randint(*worker_capacity_range, num_clients)

    def _generate_server_capacity(self, server_capacity_range, num_servers):
        np.random.seed(4)
        return np.random.randint(*server_capacity_range, num_servers)

    def _generate_transmission_cost(self, num_workers):

        np.random.seed(4)
        return np.random.randint(10, 30, (num_workers, num_workers))

    def _generate_transmission_cost_i_k(self, num_workers, num_servers):
        np.random.seed(4)
        return np.random.randint(10, 30, (num_workers, num_servers))

    def _generate_training_cost(self, num_workers):
        np.random.seed(4)
        # 原为50，150
        return np.random.randint(50, 150, num_workers)

    def _generate_sync_cost(self, num_servers):
        np.random.seed(4)
        return np.random.randint(30, 120, num_servers)

    def _generate_data_arrival(self, data_arrival_range, num_workers):
        np.random.seed(4)
        return np.random.randint(*data_arrival_range, num_workers)


    def _generate_connections(self, num_workers):
        num_connections = int(num_workers * (num_workers - 1) * 0.15)
        connections = []
        while len(connections) < num_connections:
            i, j = np.random.choice(num_workers, 2, replace=False)
            if (i, j) not in connections and (j, i) not in connections:
                connections.append((i, j))
                connections.append((j, i))
        return connections

    def _generate_connections_fl(self, num_workers):
        num_connections_fl = int(num_workers * (num_workers - 1) * 0.15)
        connections_fl = []
        while len(connections_fl) < num_connections_fl:
            i, j = np.random.choice(num_workers, 2, replace=False)
            if (i, j) not in connections_fl and (j, i) not in connections_fl:
                connections_fl.append((i, j))
                connections_fl.append((j, i))
        return connections_fl

    def _generate_worker_server_connections(self, num_workers, num_servers):
        worker_server_connections = []
        for i in range(num_workers):
            connected_servers = np.random.choice(num_servers, 1, replace=True)
            worker_server_connections.append(connected_servers)
        return worker_server_connections

    def get_worker_capacity_ranges(self):
        return self.worker_capacity_ranges

    def get_network(self):
        return {
            'link_capacity': self.link_capacity,
            'link_capacity_i_k': self.link_capacity_i_k,
            'worker_capacity': self.worker_capacity,
            'server_capacity': self.server_capacity,
            'client_capacity': self.client_capacity,
            'transmission_cost': self.transmission_cost,
            'transmission_cost_i_k': self.transmission_cost_i_k,
            'training_cost': self.training_cost,
            'sync_cost': self.sync_cost,
            'data_arrival': self.data_arrival,
            'connections': self.connections,
            'connections_fl': self.connections_fl,
            'alpha': self.alpha,
            'num_workers': self.num_workers,
            'num_servers': self.num_servers,
            'worker_server_connections': self.worker_server_connections
        }

    def get_worker_capacity_ranges(self):
        return self.worker_capacity

    def set_client_capacity_ranges(self, capacity_ranges):
        np.random.seed(4)
        self.client_capacity = capacity_ranges

