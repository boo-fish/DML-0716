import numpy as np
import pickle
from graph_builder import GraphBuilder
from network_environment import NetworkEnvironment
from utils.options import args_parser
from total_size import get_client_dataset_sizes
from benchmark_cost import LocalTraining


def convert_numpy_int_to_python(d):
    """递归地将字典中的所有NumPy整数转换为Python原生整数"""
    if isinstance(d, dict):
        return {k: convert_numpy_int_to_python(v) for k, v in d.items()}
    elif isinstance(d, (np.int32, np.int64, np.float32, np.float64)):
        return d.item()  # 将NumPy标量转换为Python标量
    elif isinstance(d, list):
        return [convert_numpy_int_to_python(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(convert_numpy_int_to_python(v) for v in d)
    else:
        return d


def GetFlow(p=None, ai=None):
    try:
        # 打开文件（如果不存在则创建）
        with open('Ai_actdata.pkl', 'r+b') as f:
            # 清空文件内容
            f.truncate(0)

        # 设置相同的随机数种子
        np.random.seed(423)

        args = args_parser()
        num_servers = 10

        # Ai 范围为 2 到 24 中的偶数
        Ai_range = np.arange(ai, ai + 1)

        # 倍数p
        # Ai_range = np.arange(2, 7)
        # alpha = 0.4
        alpha = 0.007
        num_simulations = 1
        num_workers = args.num_users
        num_clients = args.num_users

        get_client_dataset_sizes()

        for Ai in Ai_range:

            print(f"Ai: {Ai}")
            # 工人节点的容量是5，45
            for sim in range(num_simulations):
                env = NetworkEnvironment(num_workers, num_servers, num_clients, (50, 150),
                                         (5, 5 * p), (30, 60),
                                         (Ai, Ai + 1), alpha).get_network()

                # 保存工人节点容量到文件
                with open('worker_capacity.pkl', 'wb') as f:
                    pickle.dump(env['worker_capacity'], f)

                # 打印每个工人节点的初始容量
                print(f"工人节点数量: {num_workers}, 模拟次数: {sim}")
                for i in range(num_workers):
                    print(f"工作节点 {i} 初始容量: {env['worker_capacity'][i]}")

                # Proposed方法
                solver = GraphBuilder(env, Ai)
                throughput_feasible, cost_feasible, cost_efficiency_feasible, worker_training_cost = solver.solve(print_training_data=True)
                print(f"worker_training_cost: {worker_training_cost}")
                print(f"total_cost: {cost_feasible}")
                offloading_data = solver.offloading_dict

                # 将NumPy整数转换为Python整数
                offloading_data = convert_numpy_int_to_python(offloading_data)
                print("工人节点数据卸载情况", offloading_data)

                total_offloaded_data = sum(sum(targets.values()) for targets in offloading_data.values())
                print("总卸载数据量：", total_offloaded_data)

                # # Local Training方法
                local_training = LocalTraining(env, Ai)
                _, local_cost, _,training_cost,total_tidu,total_sync_cost = local_training.run_local_training()
                print(f"benchmark_training_cost: {training_cost}")
                print(f"benchmark_total_sync_cost: {total_sync_cost}")
                print(f"benchmark_total_cost: {local_cost}")
                print(f"benchmark_total_tidu: {total_tidu}")

        return offloading_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # 确保在发生异常时返回None，而不是让程序崩溃



if __name__ == "__main__":
    p=3
    ai=6
    GetFlow(p=p, ai=ai)