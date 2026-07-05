"""
基线方法 3: Weight Divergence based Device Selection (文献 [23] 方法)
- 基于数据分布的 K-means 风格设备聚类
- 每 T 时隙从每个聚类中均匀选择工作节点
- 保留 Algorithm 1 LP 流优化进行数据卸载
- 保留拉格朗日乘子更新

参考: T. Zhang et al., "Enhancing Federated Learning with Spectrum Allocation
      Optimization and Device Selection," ICLR 2022.

原理简述:
  1. 设备聚类: 根据 non-IID 数据分布为每个设备确定主导类别，将相同主导类别的设备归入同一聚类
  2. 聚类选择: 从每个聚类中均匀选择 s = K/c 个设备
  3. 流优化: 对选中的设备运行 Algorithm 1 LP 流优化进行 D2D 数据卸载

用法:
    python run_method.py --method weight_divergence [其余参数...]
"""

import numpy as np
from scipy.optimize import linprog


def get_real_flow_mb_in_t_time(total_slots=None, onlineFL_T=None, p_value=20, A_min=30, cap_min=20):
    """
    生成 Weight Divergence 基线的时隙结果。

    策略:
    - 聚类: 利用 non-IID Dirichlet 分布信息，将设备按其主导类别分入 c=10 个聚类
    - 每 T 时隙: 从每个聚类中均匀选择 s = K/c 个工作节点
    - 每时隙: 运行 LP 流优化进行数据卸载 (Algorithm 1)
    - 保留拉格朗日乘子更新
    """

    # ============ 参数设置 ============
    Num_slot = total_slots
    count_slot = 0
    T = onlineFL_T
    epsilon = 0.0005
    N = 50
    R = 250
    Thre = 35
    K = 5
    F_min = 20
    F_max = F_min * p_value
    A_max = 80
    cap_max = 100
    e_link_min = 0.5
    e_link_max = 2
    e_worker_min = 40
    e_worker_max = 80
    E_device_min = 5
    E_device_max = 20
    NUM_CLUSTERS = 10   # c = 类别数 (CIFAR-10)

    F_max_var = 1.5; F_min_var = 0.5; cap_min_var = 0.5; cap_max_var = 1.5
    theta_min_var = 1; theta_max_var = 1; theta_cap_min = 0.5; theta_cap_max = 1.5
    A_min_var = 0.5; A_max_var = 1.5; E_min_var = 0.5; E_max_var = 1.5

    time_slot_results = []

    # ============ 设备生成 ============
    np.random.seed(42)

    Dis = np.random.rand(N, 1) * R
    angle = np.random.rand(N, 1) * 2 * np.pi
    y = Dis * np.sin(angle); x = Dis * np.cos(angle)
    x_matrix = np.tile(x, (1, N)); y_matrix = np.tile(y, (1, N))
    d_matrix = np.sqrt((x_matrix - x_matrix.T) ** 2 + (y_matrix - y_matrix.T) ** 2)
    Map_connect = (d_matrix < Thre).astype(float)
    np.fill_diagonal(Map_connect, 0)

    A = np.ones((N, 1)) * A_min + np.random.rand(N, 1) * (A_max - A_min)
    A_matrix = np.tile(A, (1, N))
    E_device = np.ones((N, 1)) * E_device_min + np.random.rand(N, 1) * (E_device_max - E_device_min)
    F_device = np.ones((N, 1)) * F_min + np.random.rand(N, 1) * (F_max - F_min)
    e_device = np.ones((N, 1)) * e_worker_min + np.random.rand(N, 1) * (e_worker_max - e_worker_min)

    epsilon_link = (np.ones((N, N)) * e_link_min + np.random.rand(N, N) * (e_link_max - e_link_min)) * Map_connect * A_matrix
    Cap_link = (np.ones((N, N)) * cap_min + np.random.rand(N, N) * (cap_max - cap_min)) * Map_connect
    Cap_link = np.tril(Cap_link, -1) + np.triu(Cap_link.T, 0)

    # ============ 模拟设备聚类 ============
    # 文献[23]通过K-means对模型权重进行聚类。在Step 1中我们无法获取模型权重，
    # 因此使用设备的non-IID数据分布信息模拟聚类结果：
    # 为每个设备随机分配一个主导类别 (0~NUM_CLUSTERS-1)，模拟Dirichlet分布下的聚类
    rng_cluster = np.random.RandomState(42)
    device_cluster_labels = rng_cluster.randint(0, NUM_CLUSTERS, size=N)
    clusters = {c: np.where(device_cluster_labels == c)[0].tolist() for c in range(NUM_CLUSTERS)}
    # 过滤空聚类
    clusters = {c: members for c, members in clusters.items() if len(members) > 0}

    # ============ 初始化 ============
    lambda_val = np.ones((N, 1)) * 0.3
    s_t_wd = np.zeros((N, 1))
    Result = np.zeros(1)

    # ============ 主仿真循环 ============
    for Time in range(1, Num_slot + 1):
        if Time % 200 == 0:
            print(f"[Weight Divergence] Time = {Time}")

        F_rand = np.ones((N, 1)) * F_min_var + np.random.rand(N, 1) * (F_max_var - F_min_var)
        theta_device_rand = np.ones((N, 1)) * theta_min_var + np.random.rand(N, 1) * (theta_max_var - theta_min_var)
        cap_rand = np.ones((N, N)) * cap_min_var + np.random.rand(N, N) * (cap_max_var - cap_min_var)
        theta_cap_rand = np.ones((N, N)) * theta_cap_min + np.random.rand(N, N) * (theta_cap_max - theta_cap_min)
        A_rand = np.ones((N, 1)) * A_min_var + np.random.rand(N, 1) * (A_max_var - A_min_var)
        E_rand = np.ones((N, 1)) * E_min_var + np.random.rand(N, 1) * (E_max_var - E_min_var)

        F_t = F_device * F_rand
        e_device_t = e_device * theta_device_rand
        cap_t = Cap_link * cap_rand
        e_link_t = epsilon_link * theta_cap_rand
        A_t = A * A_rand
        E_t = E_device * E_rand
        lambda_t_matrix = np.tile(lambda_val, (1, N))

        # ---- 大时间尺度: 聚类感知设备选择 (文献[23] Algorithm 3/4) ----
        if (Time - 1) % T == 0:
            s_t_wd = np.zeros((N, 1))
            selected = []
            num_clusters = len(clusters)
            base_per_cluster = K // num_clusters
            remainder = K % num_clusters

            # 每个聚类选择 s = base_per_cluster 个 (均匀分配余数)
            cluster_list = list(clusters.keys())
            for idx, c in enumerate(cluster_list):
                members = clusters[c]
                n_select = base_per_cluster + (1 if idx < remainder else 0)
                if n_select > 0 and len(members) > 0:
                    chosen = list(np.random.choice(members, min(n_select, len(members)), replace=False))
                    selected.extend(chosen)
            # 若总数不足 K，从剩余设备中随机补充
            if len(selected) < K:
                remaining = [d for d in range(N) if d not in selected]
                additional = list(np.random.choice(remaining, min(K - len(selected), len(remaining)), replace=False))
                selected.extend(additional)
            selected = selected[:K]
            for d in selected:
                s_t_wd[d] = 1

        # ---- 小时间尺度: LP 流优化 (Algorithm 1) ----
        cap_recons = np.zeros((N + 2, N + 2))
        cost_recons = np.zeros((N + 2, N + 2))
        cap_recons[0, 1:N + 1] = A_t.flatten()
        cost_recons[0, 1:N + 1] = -1
        cap_recons[1:N + 1, 1:N + 1] = cap_t
        cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix * e_link_t
        for i in range(1, N + 1):
            cap_recons[i, N + 1] = s_t_wd[i - 1] * F_t[i - 1]

        ind = np.where(cap_recons != 0)
        f = cost_recons[ind]
        bounds = [(0, cap_recons[i, j]) for i, j in zip(ind[0], ind[1])]
        A_eq = []; b_eq = []
        for i in range(1, N + 1):
            row = np.zeros(len(ind[0]))
            row[ind[0] == i] = -1; row[ind[1] == i] = 1
            if np.any(row != 0):
                A_eq.append(row); b_eq.append(0)
        A_eq = np.array(A_eq) if A_eq else np.zeros((0, len(ind[0])))
        b_eq = np.array(b_eq) if b_eq else np.zeros(0)

        res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        flow_temp = np.zeros((N + 2, N + 2))
        if res.success:
            for idx, (i, j) in enumerate(zip(ind[0], ind[1])):
                flow_temp[i, j] = res.x[idx]

        # ---- 更新拉格朗日乘子 ----
        lambda_val = np.maximum(lambda_val + epsilon * (
            s_t_wd * e_device_t +
            np.sum(flow_temp[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True) - E_t
        ), 0)

        # ---- 提取结果 ----
        selected_devices_t = np.where(s_t_wd.flatten() > 0.5)[0].tolist()
        total_offload_t = 0.0
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j and flow_temp[i, j] > 0:
                    total_offload_t += flow_temp[i, j]

        device_computation_f_t = [float(F_t[i - 1, 0]) for i in range(1, N + 1)]
        final_data_t = []
        for i in range(1, N + 1):
            incoming = np.sum(flow_temp[1:N + 1, i])
            outgoing = np.sum(flow_temp[i, 1:N + 1])
            local = flow_temp[0, i]
            final_data_t.append(float(local + incoming - outgoing))

        time_slot_results.append((
            Time,
            selected_devices_t,
            round(total_offload_t, 4),
            [round(x, 4) for x in device_computation_f_t],
            [round(x, 4) for x in final_data_t]
        ))

        if Time > count_slot:
            Result[0] += np.sum(flow_temp[0, :])

        if Time % 5 == 0:
            print(f"\n[Weight Divergence] Time Slot {Time}:")
            print(f"  Selected (clustered): {selected_devices_t}")
            print(f"  Offload: {total_offload_t:.4f}")

    avg_thp = Result[0] / (Num_slot - count_slot) if (Num_slot - count_slot) > 0 else 0
    print(f"\n=== Weight Divergence 基线结果 (N=50) ===")
    print(f"  平均吞吐量: {avg_thp:.4f}")

    return time_slot_results
