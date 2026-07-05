"""
基线方法 2: Random Orchestration（随机编排）
- 随机工作者选择（每 T 个时隙随机选 K 个，替代 BPSO/MILP 优化）
- 保留数据卸载（Algorithm 1 的 LP 流优化仍在运行）
- 保留拉格朗日乘子更新
- 对应论文中的 "Random Orchestration" 基线

用法：在主脚本 onlineFL_step2_Online_FL-FINAL.py 中替换 import 即可:
    from baseline_random_orchestration import get_real_flow_mb_in_t_time as get_real_flow_mb_in_t_time_rand
"""

import numpy as np
from scipy.optimize import linprog


def get_real_flow_mb_in_t_time(total_slots=None, onlineFL_T=None, p_value=20, A_min=30, cap_min=20):
    """
    生成 Random Orchestration 基线的时隙结果。

    策略：
    - 每 T 个时隙：随机选择 K 个工作节点（替代 BPSO/MILP 优化）
    - 每个时隙：运行 LP 流优化进行数据卸载（Algorithm 1 的核心）
    - 保留拉格朗日乘子更新机制

    返回格式与 get_real_flow_mb_in_t_time 兼容：
    [(Time, selected_devices, total_offload, device_computation_f, final_data_amounts), ...]
    """

    # ============ 参数设置（与原 Step 1 脚本保持一致）============
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

    F_max_var = 1.5
    F_min_var = 0.5
    cap_min_var = 0.5
    cap_max_var = 1.5
    theta_min_var = 1
    theta_max_var = 1
    theta_cap_min = 0.5
    theta_cap_max = 1.5
    A_min_var = 0.5
    A_max_var = 1.5
    E_min_var = 0.5
    E_max_var = 1.5

    time_slot_results = []

    # ============ 设备生成 ============
    np.random.seed(42)

    Dis = np.random.rand(N, 1) * R
    angle = np.random.rand(N, 1) * 2 * np.pi
    y = Dis * np.sin(angle)
    x = Dis * np.cos(angle)
    x_matrix = np.tile(x, (1, N))
    y_matrix = np.tile(y, (1, N))
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

    # ============ 初始化 ============
    lambda_val = np.ones((N, 1)) * 0.3
    # s_t_random: 随机选择的工作节点 (N x 1 二进制向量)
    s_t_random = np.zeros((N, 1))
    # 初始随机选择 K 个
    init_indices = np.random.choice(N, K, replace=False)
    s_t_random[init_indices] = 1

    Result = np.zeros(1)

    # ============ 主仿真循环 ============
    for Time in range(1, Num_slot + 1):
        if Time % 200 == 0:
            print(f"[Random Orchestration] Time = {Time}")

        # 生成随机变化
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

        # ---- 大时间尺度：每 T 时隙随机选择 K 个工作节点 ----
        if (Time - 1) % T == 0:
            s_t_random = np.zeros((N, 1))
            random_indices = np.random.choice(N, K, replace=False)
            s_t_random[random_indices] = 1

        # ---- 小时间尺度：LP 流优化进行数据卸载（Algorithm 1）----
        # 构建残差图（与原脚本一致：源节点0 → 设备1..N → 虚拟汇N+1）
        cap_recons = np.zeros((N + 2, N + 2))
        cost_recons = np.zeros((N + 2, N + 2))

        # 源 → 设备的边：容量 = A_i(t)，成本 = -1（鼓励接纳）
        cap_recons[0, 1:N + 1] = A_t.flatten()
        cost_recons[0, 1:N + 1] = -1

        # 设备间 D2D 边：容量 = C_ij(t)，成本 = λ_i(t) × e_ij(t)
        cap_recons[1:N + 1, 1:N + 1] = cap_t
        cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix * e_link_t

        # 设备 → 虚拟汇：容量 = s_i × F_i(t)
        for i in range(1, N + 1):
            cap_recons[i, N + 1] = s_t_random[i - 1] * F_t[i - 1]

        # 构建线性规划
        ind = np.where(cap_recons != 0)
        f = cost_recons[ind]
        bounds = [(0, cap_recons[i, j]) for i, j in zip(ind[0], ind[1])]

        # 流守恒约束
        A_eq = []
        b_eq = []
        for i in range(1, N + 1):
            row = np.zeros(len(ind[0]))
            row[ind[0] == i] = -1
            row[ind[1] == i] = 1
            if np.any(row != 0):
                A_eq.append(row)
                b_eq.append(0)
        A_eq = np.array(A_eq) if A_eq else np.zeros((0, len(ind[0])))
        b_eq = np.array(b_eq) if b_eq else np.zeros(0)

        flow_temp = None
        res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            flow_temp = np.zeros((N + 2, N + 2))
            for idx, (i, j) in enumerate(zip(ind[0], ind[1])):
                flow_temp[i, j] = res.x[idx]
        else:
            # LP 失败时不更新，使用空流
            print(f"[Random Orchestration] LP failed at Time={Time}")
            flow_temp = np.zeros((N + 2, N + 2))

        # ---- 更新拉格朗日乘子 ----
        lambda_val = np.maximum(lambda_val + epsilon * (
            s_t_random * e_device_t +
            np.sum(flow_temp[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True) - E_t
        ), 0)

        # ---- 提取结果 ----
        selected_devices_t = np.where(s_t_random.flatten() > 0.5)[0].tolist()

        # 计算卸载总量（设备间传输）
        total_offload_t = 0.0
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                if i != j and flow_temp[i, j] > 0:
                    total_offload_t += flow_temp[i, j]

        # 设备计算能力
        device_computation_f_t = [float(F_t[i - 1, 0]) for i in range(1, N + 1)]

        # 最终每设备数据量
        final_data_t = []
        for i in range(1, N + 1):
            incoming = np.sum(flow_temp[1:N + 1, i])
            outgoing = np.sum(flow_temp[i, 1:N + 1])
            local = flow_temp[0, i]
            final_data_t.append(float(local + incoming - outgoing))

        # ---- 存储结果 ----
        time_slot_results.append((
            Time,
            selected_devices_t,
            round(total_offload_t, 4),
            [round(x, 4) for x in device_computation_f_t],
            [round(x, 4) for x in final_data_t]
        ))

        # ---- 统计 ----
        if Time > count_slot:
            Result[0] += np.sum(flow_temp[0, :])

        if Time % 5 == 0:
            print(f"\n[Random Orchestration] Time Slot {Time}:")
            print(f"  Selected (random): {selected_devices_t}")
            print(f"  Offload: {total_offload_t:.4f}")
            print(f"  Final Data (sample): {[round(x,2) for x in final_data_t[:5]]}")

    # ============ 显示结果 ============
    avg_thp = Result[0] / (Num_slot - count_slot) if (Num_slot - count_slot) > 0 else 0
    print(f"\n=== Random Orchestration 基线结果 (N=50) ===")
    print(f"  平均吞吐量: {avg_thp:.4f}")

    return time_slot_results
