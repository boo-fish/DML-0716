"""
基线方法 1: Local Training（本地训练）
- 随机工作者选择（基于能量可用性）
- 无数据卸载（每台设备仅使用自身数据训练）
- 对应论文中的 "Local Training" 基线

用法：在主脚本 onlineFL_step2_Online_FL-FINAL.py 中替换 import 即可:
    from baseline_local_training import get_real_flow_mb_in_t_time as get_real_flow_mb_in_t_time_local
"""

import numpy as np
from scipy.optimize import linprog


def get_real_flow_mb_in_t_time(total_slots=None, onlineFL_T=None, p_value=20, A_min=30, cap_min=20):
    """
    生成 Local Training 基线的时隙结果。

    策略：
    - 每个时隙从能量充足的设备中随机选择 K 个工作节点
    - 无 D2D 数据卸载，每个工作节点仅使用自身的本地数据
    - 每个工作节点的训练数据量 = min(A_i(t), F_i(t))

    返回格式与 get_real_flow_mb_in_t_time 兼容：
    [(Time, selected_devices, total_offload, device_computation_f, final_data_amounts), ...]
    """

    # ============ 参数设置（与原 Step 1 脚本保持一致）============
    Num_slot = total_slots
    count_slot = 0
    T = onlineFL_T
    T1 = 20
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

    # 随机变化参数
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

    # ============ 设备生成（种子固定，与原脚本一致）============
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
    E_device = np.ones((N, 1)) * E_device_min + np.random.rand(N, 1) * (E_device_max - E_device_min)
    F_device = np.ones((N, 1)) * F_min + np.random.rand(N, 1) * (F_max - F_min)
    e_device = np.ones((N, 1)) * e_worker_min + np.random.rand(N, 1) * (e_worker_max - e_worker_min)

    # ============ 初始化能量状态 ============
    E_local = np.zeros((N, 1))

    # ============ 收集统计指标 ============
    Result = np.zeros(1)  # 仅用于本地训练的吞吐量统计

    # ============ 主仿真循环 ============
    for Time in range(1, Num_slot + 1):
        if Time % 200 == 0:
            print(f"[Local Training] Time = {Time}")

        # 生成随机变化
        F_rand = np.ones((N, 1)) * F_min_var + np.random.rand(N, 1) * (F_max_var - F_min_var)
        A_rand = np.ones((N, 1)) * A_min_var + np.random.rand(N, 1) * (A_max_var - A_min_var)
        E_rand = np.ones((N, 1)) * E_min_var + np.random.rand(N, 1) * (E_max_var - E_min_var)

        F_t = F_device * F_rand
        A_t = A * A_rand
        E_t = E_device * E_rand

        # ---- Local Training 逻辑 ----
        # 累积能量
        E_local = E_local + E_t

        # 从能量充足的设备中随机选择 min(K, 可用数) 个工作节点
        eligible_indices = np.where(E_local.flatten() > e_device.flatten())[0]
        if len(eligible_indices) == 0:
            # 若没有设备有足够能量，使用所有设备
            eligible_indices = np.arange(N)

        num_selected = min(K, len(eligible_indices))
        selected_devices = list(np.random.choice(eligible_indices, num_selected, replace=False))

        # 计算每个被选中设备的训练数据量: min(A_i, F_i)
        final_data_t = [0.0] * N
        for idx in selected_devices:
            trainable = min(A_t[idx, 0], F_t[idx, 0])
            final_data_t[idx] = float(trainable)

        # 扣除能量
        for idx in selected_devices:
            E_local[idx] = E_local[idx] - e_device[idx]

        # 设备计算能力列表
        device_computation_f_t = [float(F_t[i, 0]) for i in range(N)]

        # 无卸载
        total_offload_t = 0.0

        # ---- 存储结果 ----
        time_slot_results.append((
            Time,
            selected_devices,
            round(total_offload_t, 4),
            [round(x, 4) for x in device_computation_f_t],
            [round(x, 4) for x in final_data_t]
        ))

        # ---- 统计 ----
        if Time > count_slot:
            thp_local = np.sum(np.minimum(A_t[selected_devices].flatten(),
                                          F_t[selected_devices].flatten()))
            Result[0] += thp_local

        if Time % 5 == 0:
            print(f"\n[Local Training] Time Slot {Time}:")
            print(f"  Selected: {selected_devices}")
            print(f"  Offload: 0 (无卸载)")
            print(f"  Final Data (sample): {[round(x,2) for x in final_data_t[:5]]}")

    # ============ 显示结果 ============
    avg_throughput = Result[0] / (Num_slot - count_slot) if (Num_slot - count_slot) > 0 else 0
    print(f"\n=== Local Training 基线结果 (N=50) ===")
    print(f"  平均吞吐量: {avg_throughput:.4f}")

    return time_slot_results
