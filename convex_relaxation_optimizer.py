# convex_relaxation_optimizer.py
import numpy as np
import cvxpy as cp
import time
import matplotlib.pyplot as plt


class ConvexRelaxationOptimizer:
    def __init__(self, env):
        """
        初始化凸松弛优化器

        Args:
            env: 网络环境字典
        """
        self.env = env
        self.num_workers = env['num_workers']
        self.num_servers = env['num_servers']
        self.alpha = env['alpha']

    def solve_feasible_case(self, Ai):
        """
        求解可行情况下的凸松弛优化问题

        Args:
            Ai: 数据到达量，形状为(num_workers,)

        Returns:
            dict: 包含吞吐量、成本、成本效率和求解状态的字典
        """
        num_workers = self.num_workers
        num_servers = self.num_servers
        alpha = self.alpha

        # 定义变量
        d = cp.Variable((num_workers, num_workers), nonneg=True)
        f = cp.Variable(num_workers, nonneg=True)
        d_server = cp.Variable((num_workers, num_servers), nonneg=True)

        # 目标函数：最小化总成本
        transmission_cost_workers = cp.sum(cp.multiply(d, self.env['transmission_cost']))
        training_cost = cp.sum(cp.multiply(f, self.env['training_cost']))
        transmission_cost_server = cp.sum(cp.multiply(d_server, self.env['transmission_cost_i_k']))
        sync_cost = cp.sum(cp.multiply(cp.sum(d_server, axis=0), self.env['sync_cost']))

        total_cost = transmission_cost_workers + training_cost + transmission_cost_server + sync_cost

        # 约束条件
        constraints = []

        # 工人容量约束
        for i in range(num_workers):
            constraints.append(f[i] <= self.env['worker_capacity'][i])

        # 参数服务器容量约束
        for k in range(num_servers):
            constraints.append(cp.sum(d_server[:, k]) <= self.env['server_capacity'][k])

        # 链路容量约束（只对存在的连接）
        connection_matrix = np.zeros((num_workers, num_workers))
        for conn in self.env['connections']:
            i, j = conn
            if i < num_workers and j < num_workers:
                connection_matrix[i, j] = 1

        for i in range(num_workers):
            for j in range(num_workers):
                if connection_matrix[i, j] == 1 and i != j:
                    constraints.append(d[i, j] + d[j, i] <= self.env['link_capacity'][i, j])
                else:
                    constraints.append(d[i, j] == 0)

        # 数据流平衡约束
        for i in range(num_workers):
            inflow = Ai[i] + cp.sum(d[:, i])
            outflow = f[i] + cp.sum(d[i, :]) + cp.sum(d_server[i, :])
            constraints.append(inflow == outflow)

        # 模型同步约束
        for i in range(num_workers):
            constraints.append(alpha * f[i] == cp.sum(d_server[i, :]))

        # 工人到参数服务器的链路容量约束
        for i in range(num_workers):
            for k in range(num_servers):
                constraints.append(d_server[i, k] <= self.env['link_capacity_i_k'][i, k] / alpha)

        # 求解问题
        problem = cp.Problem(cp.Minimize(total_cost), constraints)

        try:
            start_time = time.time()
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
            solve_time = time.time() - start_time

            if problem.status in ["optimal", "optimal_inaccurate"]:
                # 计算吞吐量
                if d_server.value is not None:
                    total_throughput = np.sum(d_server.value) / alpha
                else:
                    total_throughput = 0

                # 计算总成本
                if d.value is not None and f.value is not None and d_server.value is not None:
                    total_cost_value = (
                            np.sum(d.value * self.env['transmission_cost']) +
                            np.sum(f.value * self.env['training_cost']) +
                            np.sum(d_server.value * self.env['transmission_cost_i_k']) +
                            np.sum(np.sum(d_server.value, axis=0) * self.env['sync_cost'])
                    )
                else:
                    total_cost_value = float('inf')

                # 计算成本效率
                if total_cost_value > 0:
                    cost_efficiency = total_throughput / total_cost_value
                else:
                    cost_efficiency = float('inf') if total_throughput > 0 else 0

                return {
                    'throughput': total_throughput,
                    'total_cost': total_cost_value,
                    'cost_efficiency': cost_efficiency,
                    'status': problem.status,
                    'solve_time': solve_time,
                    'feasible': True
                }
            else:
                return {
                    'throughput': 0,
                    'total_cost': float('inf'),
                    'cost_efficiency': 0,
                    'status': problem.status,
                    'solve_time': 0,
                    'feasible': False,
                    'error': f"Solver status: {problem.status}"
                }

        except Exception as e:
            return {
                'throughput': 0,
                'total_cost': float('inf'),
                'cost_efficiency': 0,
                'status': 'error',
                'solve_time': 0,
                'feasible': False,
                'error': str(e)
            }

    def solve_infeasible_case(self, Ai, data_importance=None):
        """
        求解不可行情况下的凸松弛优化问题（带数据重要性感知）

        Args:
            Ai: 数据到达量，形状为(num_workers,)
            data_importance: 数据重要性权重，形状为(num_workers,)

        Returns:
            dict: 包含吞吐量、成本、成本效率和数据接纳率的字典
        """
        num_workers = self.num_workers
        num_servers = self.num_servers
        alpha = self.alpha

        # 默认数据重要性
        if data_importance is None:
            data_importance = np.ones(num_workers)

        # 定义变量
        d = cp.Variable((num_workers, num_workers), nonneg=True)
        f = cp.Variable(num_workers, nonneg=True)
        d_server = cp.Variable((num_workers, num_servers), nonneg=True)
        a = cp.Variable(num_workers, nonneg=True)  # 接纳的数据量

        # 目标函数：最大化效用 = 数据接纳收益 - 总成本
        data_admission_profit = cp.sum(cp.multiply(a, data_importance))

        # 各项成本
        transmission_cost_workers = cp.sum(cp.multiply(d, self.env['transmission_cost']))
        training_cost = cp.sum(cp.multiply(f, self.env['training_cost']))
        transmission_cost_server = cp.sum(cp.multiply(d_server, self.env['transmission_cost_i_k']))
        sync_cost = cp.sum(cp.multiply(cp.sum(d_server, axis=0), self.env['sync_cost']))

        total_cost = transmission_cost_workers + training_cost + transmission_cost_server + sync_cost

        # 最大化效用等价于最小化负效用
        objective = cp.Minimize(total_cost - data_admission_profit)

        # 约束条件
        constraints = []

        # 数据接纳量约束
        for i in range(num_workers):
            constraints.append(a[i] <= Ai[i])

        # 其他约束与可行情况相同
        for i in range(num_workers):
            constraints.append(f[i] <= self.env['worker_capacity'][i])

        for k in range(num_servers):
            constraints.append(cp.sum(d_server[:, k]) <= self.env['server_capacity'][k])

        # 链路容量约束
        connection_matrix = np.zeros((num_workers, num_workers))
        for conn in self.env['connections']:
            i, j = conn
            if i < num_workers and j < num_workers:
                connection_matrix[i, j] = 1

        for i in range(num_workers):
            for j in range(num_workers):
                if connection_matrix[i, j] == 1 and i != j:
                    constraints.append(d[i, j] + d[j, i] <= self.env['link_capacity'][i, j])
                else:
                    constraints.append(d[i, j] == 0)

        # 修改的数据流平衡约束
        for i in range(num_workers):
            inflow = a[i] + cp.sum(d[:, i])
            outflow = f[i] + cp.sum(d[i, :]) + cp.sum(d_server[i, :])
            constraints.append(inflow == outflow)

        # 模型同步约束
        for i in range(num_workers):
            constraints.append(alpha * f[i] == cp.sum(d_server[i, :]))

        # 工人到参数服务器的链路容量约束
        for i in range(num_workers):
            for k in range(num_servers):
                constraints.append(d_server[i, k] <= self.env['link_capacity_i_k'][i, k] / alpha)

        # 求解问题
        problem = cp.Problem(objective, constraints)

        try:
            start_time = time.time()
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
            solve_time = time.time() - start_time

            if problem.status in ["optimal", "optimal_inaccurate"]:
                # 计算吞吐量
                if d_server.value is not None:
                    total_throughput = np.sum(d_server.value) / alpha
                else:
                    total_throughput = 0

                # 计算总成本（只计算成本部分）
                if d.value is not None and f.value is not None and d_server.value is not None:
                    total_cost_value = (
                            np.sum(d.value * self.env['transmission_cost']) +
                            np.sum(f.value * self.env['training_cost']) +
                            np.sum(d_server.value * self.env['transmission_cost_i_k']) +
                            np.sum(np.sum(d_server.value, axis=0) * self.env['sync_cost'])
                    )
                else:
                    total_cost_value = float('inf')

                # 计算成本效率
                if total_cost_value > 0:
                    cost_efficiency = total_throughput / total_cost_value
                else:
                    cost_efficiency = float('inf') if total_throughput > 0 else 0

                # 计算数据接纳率
                if a.value is not None and np.sum(Ai) > 0:
                    admitted_data_ratio = np.sum(a.value) / np.sum(Ai)
                else:
                    admitted_data_ratio = 0

                return {
                    'throughput': total_throughput,
                    'total_cost': total_cost_value,
                    'cost_efficiency': cost_efficiency,
                    'admission_ratio': admitted_data_ratio,
                    'status': problem.status,
                    'solve_time': solve_time,
                    'feasible': False
                }
            else:
                return {
                    'throughput': 0,
                    'total_cost': float('inf'),
                    'cost_efficiency': 0,
                    'admission_ratio': 0,
                    'status': problem.status,
                    'solve_time': 0,
                    'feasible': False,
                    'error': f"Solver status: {problem.status}"
                }

        except Exception as e:
            return {
                'throughput': 0,
                'total_cost': float('inf'),
                'cost_efficiency': 0,
                'admission_ratio': 0,
                'status': 'error',
                'solve_time': 0,
                'feasible': False,
                'error': str(e)
            }

    def auto_detect_case(self, Ai, data_importance=None):
        """
        自动检测是可行情况还是不可行情况，并选择合适的求解方法

        Args:
            Ai: 数据到达量
            data_importance: 数据重要性权重

        Returns:
            dict: 优化结果
        """
        # 先尝试可行情况求解
        feasible_result = self.solve_feasible_case(Ai)

        # 如果可行情况求解成功，返回结果
        if feasible_result['status'] in ["optimal", "optimal_inaccurate"]:
            return feasible_result
        else:
            # 如果可行情况失败，尝试不可行情况
            print(f"可行情况求解失败({feasible_result['status']})，尝试不可行情况...")
            return self.solve_infeasible_case(Ai, data_importance)