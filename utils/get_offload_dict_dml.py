import numpy as np
import pickle
from utils.graph_builder import GraphBuilder
from utils.network_environment import NetworkEnvironment
from utils.options import args_parser
# from utils.get_total_MB_of_dataset import get_client_dataset_sizes
from others.benchmark_cost import LocalTraining


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


def GetFlow(p=None, ai=None, dataset=None):
    try:


        # 设置相同的随机数种子
        np.random.seed(423)

        args = args_parser()
        num_servers = 10
        Ai_range = np.arange(ai, ai + 1)
        if dataset == 'mnist':
            alpha = 0.4
            left = 5
            print("alpha和左区间设置为:", alpha,left)
        elif dataset == 'cifar10':
            alpha = 0.007
            left = 15
            print("alpha和左区间设置为:", alpha, left)
        elif dataset == 'cifar100':
            alpha = 0.007
            left = 15
            print("alpha和左区间设置为:", alpha, left)
        else:
            raise NotImplementedError

        num_simulations = 1
        num_workers = args.num_users
        num_clients = args.num_users

        # get_client_dataset_sizes()

        # 修改：初始化Ai_actdata列表（内存中维护）
        ai_actdata = []
        worker_capacity = None  # 初始化变量
        training_data = None  # 初始化变量


        for Ai in Ai_range:

            print(f"Ai: {Ai}")
            # 工人节点的容量是5，45
            for sim in range(num_simulations):
                env = NetworkEnvironment(num_workers, num_servers, num_clients, (50, 150),
                                         (left, left * p), (30, 60),
                                         (Ai, Ai + 1), alpha).get_network()


                # 移除：不再保存worker_capacity到pkl
                # with open('worker_capacity.pkl', 'wb') as f:
                #     pickle.dump(env['worker_capacity'], f)
                worker_capacity = env['worker_capacity']  # 内存中存储

                # 打印每个工人节点的初始容量
                print(f"工人节点数量: {num_workers}, 模拟次数: {sim}")
                for i in range(num_workers):
                    print(f"工作节点 {i} 初始容量: {env['worker_capacity'][i]}")


                # Proposed方法
                solver = GraphBuilder(env, Ai)
                # 修改：接收完整返回值
                throughput_feasible, cost_feasible, cost_efficiency_feasible, worker_training_cost, ai_actdata_sub, training_data = solver.solve(print_training_data=True)
                # =========修复：打印ai_actdata_sub长度，确认是1==========
                # print(f"[DEBUG] ai_actdata_sub长度: {len(ai_actdata_sub)}")  # 应输出1
                ai_actdata.extend(ai_actdata_sub)  # 聚合Ai_actdata

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

        # 修改：返回4个值（原offloading_data + 新增的worker_capacity/ai_actdata/training_data）
        return offloading_data, worker_capacity, ai_actdata, training_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None  # 异常时返回全None



# if __name__ == "__main__":
    # p=3
    # ai=6
    # GetFlow(p=p, ai=ai)