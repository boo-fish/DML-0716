from network_environment import NetworkEnvironment
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import matplotlib
import pickle

matplotlib.use('Agg')


class GraphBuilder:
    def __init__(self, environment, Ai):
        self.env = environment
        self.Ai = Ai
        self.graph = None
        self.Ai_actdata = []
        self.offloading_dict = {}  # 存储工人节点卸载信息

    def build_graph(self, is_feasible=True):
        G = nx.DiGraph()

        flow = float(self.Ai * self.env['num_workers'])

        G.add_node('source', demand=-flow)
        G.add_node('sink', demand=flow)

        for i in range(self.env['num_workers']):
            if is_feasible:
                G.add_edge("source", f"worker_{i}", capacity=self.env['data_arrival'][i], weight=0, flow=0)
            else:
                G.add_edge("source", f"worker_{i}", capacity=self.env['data_arrival'][i], weight=-500, flow=0)

        for i, j in self.env['connections']:
            G.add_edge(
                f"worker_{i}", f"worker_{j}",
                capacity=self.env['link_capacity'][i][j],
                weight=self.env['transmission_cost'][i][j],
                flow=0
            )

        for i in range(self.env['num_workers']):
            G.add_edge(
                f"worker_{i}", f"worker_{i}_result",
                capacity=self.env['worker_capacity'][i],
                weight=self.env['training_cost'][i],
                flow=0
            )

            for server in self.env['worker_server_connections'][i]:
                G.add_edge(
                    f"worker_{i}_result", f"server_{server}",
                    capacity=int(self.env['link_capacity_i_k'][i][server] / self.env['alpha']),
                    weight=int(self.env['alpha'] * self.env['transmission_cost_i_k'][i][server]),
                    flow=0
                )

        for k in range(self.env['num_servers']):
            G.add_edge(f"server_{k}", "sink", capacity=int(self.env['server_capacity'][k] / self.env['alpha']),
                       weight=int(self.env['alpha'] * self.env['sync_cost'][k]),
                       flow=0)
        return G

    def solve(self, print_training_data=False):
        self.graph = self.build_graph(is_feasible=True)
        try:
            flowCost, flow_dict = nx.network_simplex(self.graph)

            if not flow_dict:
                print("未找到可行流。尝试使用不可行算法。")
                return self.run_algorithm(print_training_data=True)

            total_throughput = sum(flow_dict["source"][f"worker_{i}"] for i in range(self.env['num_workers']))

            total_cost = nx.cost_of_flow(self.graph, flow_dict)
            # print("库求解的flow_dict:",flow_dict)


            # 计算从参数服务器到汇点的成本
            server_to_sink_cost = 0
            for u, v, data in self.graph.edges(data=True):
                if u.startswith("server_") and v == "sink":
                    flow = flow_dict[u][v]
                    cost_per_unit = data['weight']
                    server_to_sink_cost += flow * cost_per_unit
            # total_cost = total_cost - server_to_sink_cost

            cost_efficiency = total_throughput / total_cost if total_cost != 0 else float('inf')

            # 计算工人节点的训练成本总和
            worker_training_cost = 0
            for i in range(self.env['num_workers']):
                u = f"worker_{i}"
                v = f"worker_{i}_result"
                if u in flow_dict and v in flow_dict[u]:
                    flow = flow_dict[u][v]
                    cost_per_unit = self.env['training_cost'][i]
                    worker_training_cost += flow * cost_per_unit

            if print_training_data:
                # 打印每个工人节点最终训练的数据量
                self.print_worker_training_data(flow_dict)

            training_sizes = self.get_worker_training_sizes(flow_dict)
            self.Ai_actdata.append([self.Ai, training_sizes])
            self.save_Ai_actdata()

            # 生成并存储卸载字典
            self.offloading_dict = self.get_worker_offloading_dict(flow_dict)

            return total_throughput, total_cost, cost_efficiency, worker_training_cost

        except nx.NetworkXUnfeasible as e:
            print(f"可行情况抛出异常，失败。具体错误信息: {str(e)}")
            # 当抛出异常时，调用 run_algorithm 方法并返回结果
            return self.run_algorithm(print_training_data=True)

    def build_residual_graph(self, G):
        residual_G = nx.DiGraph()
        for u, v, data in G.edges(data=True):
            if data['capacity'] - data.get('flow', 0) > 0:
                residual_G.add_edge(u, v, capacity=data['capacity'] - data.get('flow', 0), weight=data['weight'])
            if data.get('flow', 0) > 0:
                residual_G.add_edge(v, u, capacity=data.get('flow', 0), weight=-data['weight'])
        return residual_G

    def bellman_ford(self, residual_G, source, sink):
        distance = {node: float('inf') for node in residual_G.nodes()}
        distance[source] = 0
        parent = {node: None for node in residual_G.nodes()}

        for _ in range(len(residual_G.nodes()) - 1):
            for u, v, data in residual_G.edges(data=True):
                if distance[u] + data['weight'] < distance[v] and data['capacity'] > 0:
                    distance[v] = distance[u] + data['weight']
                    parent[v] = u

        for u, v, data in residual_G.edges(data=True):
            if distance[u] + data['weight'] < distance[v] and data['capacity'] > 0:
                return None

        path = []
        current = sink
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

        if path[0] != source:
            return None
        return path

    def calculate_path_capacity(self, residual_G, path):
        min_capacity = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_capacity = min(min_capacity, residual_G[u][v]['capacity'])
        return min_capacity

    def augment_flow(self, G, residual_G, path, capacity):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in G.edges():
                if 'flow' not in G[u][v]:
                    G[u][v]['flow'] = 0
                G[u][v]['flow'] += capacity
            else:
                G[v][u]['flow'] -= capacity
        return G

    def is_valid_worker_result(self, u):
        pattern = r'^worker_([0-9]|[1-4][0-9])_result$'
        match = re.match(pattern, u)
        return bool(match)

    def run_algorithm(self, print_training_data=False):
        G = self.build_graph(is_feasible=False)
        source = 'source'
        sink = 'sink'
        iteration = 0
        while True:
            residual_G = self.build_residual_graph(G)
            path = self.bellman_ford(residual_G, source, sink)
            if path is None or len(path) == 0:
                break
            capacity = self.calculate_path_capacity(residual_G, path)
            if capacity == 0:
                break
            G = self.augment_flow(G, residual_G, path, capacity)
            iteration += 1

        total_cost = 0
        total_throughput = 0
        flow_dict = {u: {v: data['flow'] for v, data in G[u].items() if 'flow' in data} for u in G.nodes()}
        for u, v, data in G.edges(data=True):
            if 'flow' in data and data['flow'] > 0:
                total_cost += data['weight'] * data['flow']
                if u == source:
                    # 减去负权成本才是实际成本
                    total_cost -= data['weight'] * data['flow']

        # 计算从参数服务器到汇点的成本
        server_to_sink_cost = 0
        for u, v, data in G.edges(data=True):
            if u.startswith("server_") and v == "sink" and 'flow' in data and data['flow'] > 0:
                flow = data['flow']
                cost_per_unit = data['weight']
                server_to_sink_cost += flow * cost_per_unit

        # total_cost = total_cost - server_to_sink_cost


        # 计算从 worker_i 到 worker_i_result 的流量总和作为不可行情况下的吞吐量
        for i in range(self.env['num_workers']):
            u = f"worker_{i}"
            v = f"worker_{i}_result"
            if (u, v) in G.edges() and 'flow' in G[u][v]:
                total_throughput += G[u][v]['flow']

        cost_efficiency = total_throughput / total_cost if total_cost != 0 else float('inf')

        # 计算工人节点的训练成本总和
        worker_training_cost = 0
        for i in range(self.env['num_workers']):
            u = f"worker_{i}"
            v = f"worker_{i}_result"
            if (u, v) in G.edges() and 'flow' in G[u][v]:
                flow = G[u][v]['flow']
                cost_per_unit = self.env['training_cost'][i]
                worker_training_cost += flow * cost_per_unit


        if print_training_data:
            # 打印每个工人节点最终训练的数据量
            self.print_worker_training_data(flow_dict)

        training_sizes = self.get_worker_training_sizes(flow_dict)
        self.Ai_actdata.append([self.Ai, training_sizes])
        self.save_Ai_actdata()

        # 生成并存储卸载字典
        self.offloading_dict = self.get_worker_offloading_dict(flow_dict)

        return total_throughput, total_cost, cost_efficiency, worker_training_cost  # 返回路径选择字典

    def print_offloading_info(self, flow_dict):
        for i in range(self.env['num_workers']):
            from_worker = f"worker_{i}"
            offloading_info = []
            for v, flow in flow_dict.get(from_worker, {}).items():
                if v.startswith("worker_") and v != from_worker:
                    to_worker = v
                    offloading_info.append((to_worker, flow))
            if offloading_info:
                print(f"工作节点 {i} 将数据卸载到了以下节点:")
                for to_worker, flow in offloading_info:
                    # 提取目标工作节点编号
                    to_worker_num = to_worker.split("_")[1]
                    print(f"  - 工作节点 {to_worker_num}: {flow}")

    def get_path_selection(self, flow_dict):
        path_selection = {}
        for i in range(self.env['num_workers']):
            from_worker = f"worker_{i}"
            offloading_info = []
            for v, flow in flow_dict.get(from_worker, {}).items():
                if v.startswith("worker_") and v != from_worker:
                    to_worker = v
                    offloading_info.append((to_worker, flow))
            if offloading_info:
                path_selection[i] = offloading_info
        return path_selection

    def print_worker_training_data(self, flow_dict):
        training_data = {}
        for i in range(self.env['num_workers']):
            worker_node = f"worker_{i}"
            result_node = f"worker_{i}_result"
            if result_node in flow_dict.get(worker_node, {}):
                training_data[i] = flow_dict[worker_node][result_node]
                print(f"工人节点 {i} 最终训练的数据量: {training_data[i]} 兆")
            else:
                training_data[i] = 0
                print(f"工人节点 {i} 最终训练的数据量: 0 兆")

        # 将训练数据量保存到文件中
        with open('worker_training_data.pkl', 'wb') as f:
            pickle.dump(training_data, f)

        return training_data

    def get_worker_training_sizes(self, flow_dict):
        training_sizes = []
        for i in range(self.env['num_workers']):
            worker_node = f"worker_{i}"
            result_node = f"worker_{i}_result"
            if result_node in flow_dict.get(worker_node, {}):
                training_data = flow_dict[worker_node][result_node]
                training_sizes.append(training_data)
            else:
                training_sizes.append(0)
        return training_sizes

    def save_Ai_actdata(self):
        try:
            with open('Ai_actdata.pkl', 'rb') as f:
                existing_data = pickle.load(f)
                self.Ai_actdata = existing_data + self.Ai_actdata
        except (FileNotFoundError, EOFError):
            pass

        with open('Ai_actdata.pkl', 'wb') as f:
            pickle.dump(self.Ai_actdata, f)

    def get_worker_offloading_dict(self, flow_dict=None):
        """
        提取工人节点之间的数据卸载情况，返回字典供其他函数使用。
        字典格式：{源工人节点ID: {目标工人节点ID: 卸载流量, ...}, ...}
        """
        offloading_dict = {}
        # 如果未传入flow_dict，使用当前图的流量数据
        if flow_dict is None and self.graph is not None:
            flow_dict = {u: {v: data['flow'] for v, data in self.graph[u].items() if 'flow' in data} for u in
                         self.graph.nodes()}

        if flow_dict is None:
            return offloading_dict  # 无流量数据时返回空字典

        # 遍历所有节点的流量数据，筛选worker之间的卸载
        for source_node, targets in flow_dict.items():
            # 匹配源节点是否为工人节点（格式：worker_数字）
            if not re.match(r'^worker_\d+$', source_node):
                continue
            source_id = int(source_node.split('_')[1])  # 提取源工人节点ID（如worker_0 → 0）
            offloading_dict[source_id] = {}

            for target_node, flow in targets.items():
                # 匹配目标节点是否为其他工人节点（排除自身和非worker节点）
                if re.match(r'^worker_\d+$', target_node) and target_node != source_node and flow > 0:
                    target_id = int(target_node.split('_')[1])  # 提取目标工人节点ID
                    offloading_dict[source_id][target_id] = flow  # 存储非零流量的卸载数据

        # 移除无卸载行为的工人节点（值为空字典的键）
        return {k: v for k, v in offloading_dict.items() if v}