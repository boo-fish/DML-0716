import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
import os


def get_real_flow_mb_in_t_time(total_slots=None, onlineFL_T=None,p_value=20):

    # Basic Parameters Setting
    Num_slot = total_slots
    count_slot = 0
    T = onlineFL_T
    T1 = 20
    epsilon = 0.0005
    N = 50  # Fixed at 50
    R = 250
    Thre = 35
    K = 5
    F_min = 20
    # F_max = 20
    F_max = F_min * p_value  # F_max is determined by F_min and p_value 
    # A_min = 1
    # A_max = 3
    A_min = 20
    A_max = 80
    cap_min = 20
    cap_max = 100
    e_link_min = 0.5  # Link Cost
    e_link_max = 2
    e_worker_min = 40  # Worker cost
    e_worker_max = 80
    E_device_min = 5  # energy arrival
    E_device_max = 20

    # Variation parameters
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

    # Initialize result arrays
    Result = np.zeros(4)  # [Proposed_T, Local, Static, Proposed_T1]
    Result_K = np.zeros(4)  # [s_t_count, local_count, K, s_t1_count]

    # Initialize detailed results storage
    time_slot_results = []  # Store results for each time slot
    
    # ================== 新增：初始化前100时隙成本存储 ==================
    cost_records = []  # 每个元素: (时隙, Worker成本, 传输成本, 显性核心总成本)
    # ====================================================================

    # Device Generation
    np.random.seed(42)

    # Generate device positions and network topology
    Dis = np.random.rand(N, 1) * R
    angle = np.random.rand(N, 1) * 2 * np.pi
    y = Dis * np.sin(angle)
    x = Dis * np.cos(angle)

    x_matrix = np.tile(x, (1, N))
    y_matrix = np.tile(y, (1, N))
    d_matrix = np.sqrt((x_matrix - x_matrix.T) ** 2 + (y_matrix - y_matrix.T) ** 2)

    Map_connect = (d_matrix < Thre).astype(float)
    np.fill_diagonal(Map_connect, 0)

    # Device characteristics
    A = np.ones((N, 1)) * A_min + np.random.rand(N, 1) * (A_max - A_min)
    A_matrix = np.tile(A, (1, N))
    E_device = np.ones((N, 1)) * E_device_min + np.random.rand(N, 1) * (E_device_max - E_device_min)
    F_device = np.ones((N, 1)) * F_min + np.random.rand(N, 1) * (F_max - F_min)
    e_device = np.ones((N, 1)) * e_worker_min + np.random.rand(N, 1) * (e_worker_max - e_worker_min)

    # Network characteristics
    epsilon_link = (np.ones((N, N)) * e_link_min + np.random.rand(N, N) * (e_link_max - e_link_min)) * Map_connect * A_matrix
    Cap_link = (np.ones((N, N)) * cap_min + np.random.rand(N, N) * (cap_max - cap_min)) * Map_connect
    Cap_link = np.tril(Cap_link, -1) + np.triu(Cap_link.T, 0)

    # Energy constraint adjustment
    temp_ind = (A > F_device).flatten()
    e_device_mod = e_device.copy()
    e_device_mod[temp_ind] = np.inf

    # Initialize optimization variables
    lambda_val = np.ones((N, 1)) * 0.3
    lambda1 = np.ones((N, 1)) * 0.3
    E_local = np.zeros((N, 1))
    s_t = np.zeros((N, 1))
    s_t[:K] = 1
    s_t1 = s_t.copy()
    s_static_ori = s_t.copy()
    s_static_ori_ind = list(range(K))
    lambda_static = np.ones((N, 1)) * 0.3
    E_static = np.zeros((N, 1))

    # Main simulation loop
    for Time in range(1, Num_slot + 1):
        if Time % 200 == 0:
            print(f"Time = {Time}")
        
        # Generate random variations
        F_rand = np.ones((N, 1)) * F_min_var + np.random.rand(N, 1) * (F_max_var - F_min_var)
        theta_device_rand = np.ones((N, 1)) * theta_min_var + np.random.rand(N, 1) * (theta_max_var - theta_min_var)
        cap_rand = np.ones((N, N)) * cap_min_var + np.random.rand(N, N) * (cap_max_var - cap_min_var)
        theta_cap_rand = np.ones((N, N)) * theta_cap_min + np.random.rand(N, N) * (theta_cap_max - theta_cap_min)
        A_rand = np.ones((N, 1)) * A_min_var + np.random.rand(N, 1) * (A_max_var - A_min_var)
        E_rand = np.ones((N, 1)) * E_min_var + np.random.rand(N, 1) * (E_max_var - E_min_var)
        
        # Apply variations
        F_t = F_device * F_rand
        e_device_t = e_device * theta_device_rand
        cap_t = Cap_link * cap_rand
        e_link_t = epsilon_link * theta_cap_rand
        A_t = A * A_rand
        E_t = E_device * E_rand
        
        # Initialize metrics for current time slot
        selected_devices_t = []
        total_offload_t = 0
        device_computation_f_t = []  # Changed from dict to list
        final_data_t = []
        
        # Proposed Approach (T period)
        lambda_t_matrix = np.tile(lambda_val, (1, N))
        
        if (Time - 1) % T == 0:
            # MILP formulation for device selection
            cap_recons = np.zeros((N + 1, N + 1))
            cost_recons = np.zeros((N + 1, N + 1))
            cap_recons[0, 1:N + 1] = A_t.flatten()
            cost_recons[0, 1:N + 1] = -1
            cap_recons[1:N + 1, 1:N + 1] = cap_t
            cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix * e_link_t
            
            ind = np.where(cap_recons != 0)
            M = len(ind[0])
            
            # Objective function
            f_flow = cost_recons[ind]
            f_s = (lambda_val.flatten() * e_device_t.flatten())
            c = np.concatenate([f_flow, f_s])
            
            # Integrality constraints
            integrality = np.concatenate([np.zeros(M), np.full(N, 2)])  # 2 for binary
            
            # Bounds
            lb = np.concatenate([np.zeros(M), np.zeros(N)])
            ub = np.concatenate([cap_recons[ind], np.ones(N)])
            bounds = Bounds(lb, ub)
            
            # Equality constraints
            A_eq_rows = []
            b_eq_rows = []
            
            # Flow conservation constraints
            for i in range(1, N + 1):
                row = np.zeros(M + N)
                for idx in range(M):
                    src, dst = ind[0][idx], ind[1][idx]
                    if dst == i:
                        row[idx] = 1
                    elif src == i:
                        row[idx] = -1
                row[M + (i - 1)] = -F_t[i - 1, 0]
                A_eq_rows.append(row)
                b_eq_rows.append(0)
            
            # Sum(s) == K constraint
            row = np.zeros(M + N)
            row[M:] = 1
            A_eq_rows.append(row)
            b_eq_rows.append(K)
            
            A_eq = np.array(A_eq_rows)
            b_eq = np.array(b_eq_rows)
            constraints = LinearConstraint(A_eq, b_eq, b_eq)
            
            # Solve MILP
            res_milp = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
            
            if res_milp.success:
                flow_temp = np.zeros((N + 1, N + 1))
                for idx in range(M):
                    i, j = ind[0][idx], ind[1][idx]
                    flow_temp[i, j] = res_milp.x[idx]
                
                s_t_vals = res_milp.x[M:M+N]
                
                # K-value compensation strategy
                s_t_rounded = np.round(s_t_vals)
                if np.sum(s_t_rounded) == K:
                    s_t = s_t_rounded
                else:
                    # Fallback to top-K selection
                    s_t = np.zeros_like(s_t_vals)
                    topk_indices = np.argsort(s_t_vals)[-K:]
                    s_t[topk_indices] = 1
                
                s_t = s_t.reshape(-1, 1)
            else:
                print(f"Proposed T MILP failed at Time={Time}")
        else:
            # Flow optimization with fixed device selection
            cap_recons = np.zeros((N + 2, N + 2))
            cost_recons = np.zeros((N + 2, N + 2))
            cap_recons[0, 1:N + 1] = A_t.flatten()
            cost_recons[0, 1:N + 1] = -1
            cap_recons[1:N + 1, 1:N + 1] = cap_t
            cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix * e_link_t
            for i in range(1, N + 1):
                cap_recons[i, N + 1] = s_t[i - 1] * F_t[i - 1]
            
            ind = np.where(cap_recons != 0)
            f = cost_recons[ind]
            bounds = [(0, cap_recons[i, j]) for i, j in zip(ind[0], ind[1])]
            
            # Flow conservation constraints
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
            
            res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                flow_temp = np.zeros((N + 2, N + 2))
                for idx, (i, j) in enumerate(zip(ind[0], ind[1])):
                    flow_temp[i, j] = res.x[idx]
        
        # Update Lagrange multipliers
        if 'flow_temp' in locals():
            lambda_val = np.maximum(lambda_val + epsilon * (s_t * e_device_t +
                                                                np.sum(flow_temp[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True) - E_t), 0)
            
            # Collect metrics for Proposed T method
            selected_devices_t = np.where(s_t.flatten() > 0.5)[0].tolist()
            
            # Calculate total offloading data (device-to-device transfers)
            total_offload_t = 0
            for i in range(1, N+1):
                for j in range(1, N+1):
                    if i != j and flow_temp[i, j] > 0:
                        total_offload_t += flow_temp[i, j]
            
            # Device computation capabilities (F) - 改为list
            device_computation_f_t = [F_t[i-1, 0] for i in range(1, N+1)]
            
            # Final data amounts for each device - 改为list
            final_data_t = []
            for i in range(1, N+1):
                # Final data = incoming flow - outgoing flow + local processing
                incoming_flow = np.sum(flow_temp[1:N+1, i])  # Flow from other devices to i
                outgoing_flow = np.sum(flow_temp[i, 1:N+1])  # Flow from i to other devices
                local_processing = flow_temp[0, i]  # Flow from source to i
                final_data_amount = local_processing + incoming_flow - outgoing_flow
                final_data_t.append(final_data_amount)
        
        # ================== 新增：计算并保存前100时隙的显性核心成本 ==================
        if Time <= 100 and 'flow_temp' in locals():
            # 1. Worker节点运行与模型全生命周期成本 (θ_i(t)s_i(t))
            worker_cost = np.sum(s_t * e_device_t)
            
            # 2. 多跳数据卸载传输成本 (∑d_ij(t)θ_ij(t))
            # 注意：flow_temp[1:N+1, 1:N+1] 对应设备间的流量矩阵
            transmission_cost = np.sum(flow_temp[1:N+1, 1:N+1] * e_link_t)
            
            # 3. 显性核心总成本
            total_explicit_cost = worker_cost + transmission_cost
            
            # 保存到记录列表
            cost_records.append({
                'time_slot': Time,
                'worker_cost': round(worker_cost, 4),
                'transmission_cost': round(transmission_cost, 4),
                'total_explicit_cost': round(total_explicit_cost, 4)
            })
            
            # 显式调试输出
            print(f"\n[调试] 时隙 {Time:3d} 显性核心成本详情:")
            print(f"  ├─ Worker节点运行与模型成本: {worker_cost:10.4f}")
            print(f"  ├─ 多跳数据卸载传输成本:   {transmission_cost:10.4f}")
            print(f"  └─ 显性核心总成本:         {total_explicit_cost:10.4f}")
        # ================================================================================
        
        # Proposed Approach T1 (similar logic as T but with different period)
        lambda_t_matrix1 = np.tile(lambda1, (1, N))
        
        if (Time - 1) % T1 == 0:
            # MILP formulation for T1 period
            cap_recons = np.zeros((N + 1, N + 1))
            cost_recons = np.zeros((N + 1, N + 1))
            cap_recons[0, 1:N + 1] = A_t.flatten()
            cost_recons[0, 1:N + 1] = -1
            cap_recons[1:N + 1, 1:N + 1] = cap_t
            cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix1 * e_link_t
            
            ind = np.where(cap_recons != 0)
            M = len(ind[0])
            
            f_flow = cost_recons[ind]
            f_s = (lambda1.flatten() * e_device_t.flatten())
            c = np.concatenate([f_flow, f_s])
            integrality = np.concatenate([np.zeros(M), np.full(N, 2)])
            lb = np.concatenate([np.zeros(M), np.zeros(N)])
            ub = np.concatenate([cap_recons[ind], np.ones(N)])
            bounds = Bounds(lb, ub)
            
            A_eq_rows = []
            b_eq_rows = []
            for i in range(1, N + 1):
                row = np.zeros(M + N)
                for idx in range(M):
                    src, dst = ind[0][idx], ind[1][idx]
                    if dst == i:
                        row[idx] = 1
                    elif src == i:
                        row[idx] = -1
                row[M + (i - 1)] = -F_t[i - 1, 0]
                A_eq_rows.append(row)
                b_eq_rows.append(0)
            
            row = np.zeros(M + N)
            row[M:] = 1
            A_eq_rows.append(row)
            b_eq_rows.append(K)
            
            A_eq = np.array(A_eq_rows)
            b_eq = np.array(b_eq_rows)
            constraints = LinearConstraint(A_eq, b_eq, b_eq)
            
            res_milp = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
            
            if res_milp.success:
                flow_temp1 = np.zeros((N + 1, N + 1))
                for idx in range(M):
                    i, j = ind[0][idx], ind[1][idx]
                    flow_temp1[i, j] = res_milp.x[idx]
                
                s_t1_vals = res_milp.x[M:M + N]
                
                # K-value compensation for T1
                s_t1_rounded = np.round(s_t1_vals)
                if np.sum(s_t1_rounded) == K:
                    s_t1 = s_t1_rounded
                else:
                    s_t1 = np.zeros_like(s_t1_vals)
                    topk_indices = np.argsort(s_t1_vals)[-K:]
                    s_t1[topk_indices] = 1
                
                s_t1 = s_t1.reshape(-1, 1)
            else:
                print(f"Proposed T1 MILP failed at Time={Time}")
        else:
            # Flow optimization for T1 period
            cap_recons = np.zeros((N + 2, N + 2))
            cost_recons = np.zeros((N + 2, N + 2))
            cap_recons[0, 1:N + 1] = A_t.flatten()
            cost_recons[0, 1:N + 1] = -1
            cap_recons[1:N + 1, 1:N + 1] = cap_t
            cost_recons[1:N + 1, 1:N + 1] = lambda_t_matrix1 * e_link_t
            for i in range(1, N + 1):
                cap_recons[i, N + 1] = s_t1[i - 1] * F_t[i - 1]
            
            ind = np.where(cap_recons != 0)
            f = cost_recons[ind]
            bounds = [(0, cap_recons[i, j]) for i, j in zip(ind[0], ind[1])]
            
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
            
            res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                flow_temp1 = np.zeros((N + 2, N + 2))
                for idx, (i, j) in enumerate(zip(ind[0], ind[1])):
                    flow_temp1[i, j] = res.x[idx]
        
        # Update Lagrange multipliers for T1
        if 'flow_temp1' in locals():
            lambda1 = np.maximum(lambda1 + epsilon * (s_t1 * e_device_t +
                                                        np.sum(flow_temp1[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True) - E_t), 0)
        
        # Static Orchestration (baseline)
        lambda_static_matrix = np.tile(lambda_static, (1, N))
        E_static = E_static + E_t
        s_static = s_static_ori * (E_static > e_device_t)
        
        # Rotate static selection periodically
        if Time % T == 0:
            s_static_ori_ind = [(x + K) % N for x in s_static_ori_ind]
            s_static_ori = np.zeros((N, 1))
            for idx in s_static_ori_ind:
                s_static_ori[idx] = 1
        
        # Static flow optimization
        cap_recons = np.zeros((N + 2, N + 2))
        cost_recons = np.zeros((N + 2, N + 2))
        cap_recons[0, 1:N + 1] = A_t.flatten()
        cost_recons[0, 1:N + 1] = -1
        cap_recons[1:N + 1, 1:N + 1] = cap_t
        cost_recons[1:N + 1, 1:N + 1] = lambda_static_matrix * e_link_t
        for i in range(1, N + 1):
            cap_recons[i, N + 1] = s_static[i - 1] * F_t[i - 1]
        
        ind = np.where(cap_recons != 0)
        f = cost_recons[ind]
        bounds = [(0, cap_recons[i, j]) for i, j in zip(ind[0], ind[1])]
        
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
        
        res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            flow_temp_static = np.zeros((N + 2, N + 2))
            for idx, (i, j) in enumerate(zip(ind[0], ind[1])):
                flow_temp_static[i, j] = res.x[idx]
            E_static = E_static - s_static * e_device_t + np.sum(flow_temp_static[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True)
            lambda_static = np.maximum(lambda_static + epsilon * (s_static * e_device_t +
                                                                    np.sum(flow_temp_static[1:N + 1, 1:N + 1] * e_link_t, axis=1, keepdims=True) - E_t), 0)
        
        # Local Execution (baseline)
        E_local = E_local + E_t
        ind_local = np.where(E_local > e_device_t)[0]
        if len(ind_local) > K:
            ind_temp_perm = np.random.choice(len(ind_local), K, replace=False)
            ind_local = ind_local[ind_temp_perm]
        thp_local = np.sum(np.minimum(A_t[ind_local].flatten(), F_t[ind_local].flatten()))
        E_local[ind_local] = E_local[ind_local] - e_device_t[ind_local]
        
        # Store time slot results as separate parameters (四舍五入保留4位小数)
        time_slot_results.append((
            Time,                                               # time_slot
            selected_devices_t,                                 # selected_devices
            round(total_offload_t, 4),                          # total_offload (四舍五入)
            [round(x, 4) for x in device_computation_f_t],      # device_computation_f (每个元素四舍五入)
            [round(x, 4) for x in final_data_t]                 # final_data_amounts (每个元素四舍五入)
        ))
        
        # Print current time slot results (optional)
        if Time % 5 == 0:
            print(f"\nTime Slot {Time} Results:")
            print(f"  Selected Devices: {selected_devices_t}")
            print(f"  Total Offload Data: {total_offload_t:.4f}")
            print(f"  Sample Device Computation F: {device_computation_f_t[:5]}")
            print(f"  Sample Final Data Amounts: {final_data_t[:5]}")
        
        # Collect results (after warm-up period)
        if Time > count_slot:
            if 'flow_temp' in locals():
                Result[0] += np.sum(flow_temp[0, :])
            Result[1] += thp_local
            if 'flow_temp_static' in locals():
                Result[2] += np.sum(flow_temp_static[0, :])
            if 'flow_temp1' in locals():
                Result[3] += np.sum(flow_temp1[0, :])
            
            Result_K[0] += np.sum(s_t)
            Result_K[1] += len(ind_local)
            Result_K[2] += K
            Result_K[3] += np.sum(s_t1)

    # Calculate average throughput
    avg_throughput = Result / (Num_slot - count_slot)

    # Display results
    print("\n=== 仿真结果 (N=50) ===")
    print(f"平均吞吐量:")
    print(f"  提出的方法 (T={T}): {avg_throughput[0]:.4f}")
    print(f"  本地执行: {avg_throughput[1]:.4f}")
    print(f"  静态编排: {avg_throughput[2]:.4f}")
    print(f"  提出的方法 (T1={T1}): {avg_throughput[3]:.4f}")
    
    # ================== 新增：输出前100时隙成本汇总 ==================
    print("\n=== 前100时隙显性核心成本汇总 ===")
    print(f"{'时隙':<6} {'Worker成本':<12} {'传输成本':<12} {'总成本':<12}")
    print("-" * 46)
    for record in cost_records:
        print(f"{record['time_slot']:<6} "
              f"{record['worker_cost']:<12.4f} "
              f"{record['transmission_cost']:<12.4f} "
              f"{record['total_explicit_cost']:<12.4f}")
    # ====================================================================

    # ================== 保存100时隙成本为csv和npy ==================
    save_cost_dir = './绘图文件/每轮OnlineFL成本'
    os.makedirs(save_cost_dir, exist_ok=True)

    # 构建成本数组: (100, 4) -> [time_slot, worker_cost, transmission_cost, total_explicit_cost]
    cost_array = np.array([
        [r['time_slot'], r['worker_cost'], r['transmission_cost'], r['total_explicit_cost']]
        for r in cost_records
    ])

    # 保存为npy
    npy_path = os.path.join(save_cost_dir, f'cost_T{T}_p{p_value}.npy')
    np.save(npy_path, cost_array)
    print(f"\n成本数据已保存(npy): {npy_path}")

    # 保存为csv
    csv_path = os.path.join(save_cost_dir, f'cost_T{T}_p{p_value}.csv')
    np.savetxt(csv_path, cost_array, delimiter=',',
               header='time_slot,worker_cost,transmission_cost,total_explicit_cost',
               fmt='%d,%.4f,%.4f,%.4f')
    print(f"成本数据已保存(csv): {csv_path}")
    # ====================================================================

    return time_slot_results, cost_records  # 修改返回值，同时返回成本记录


# 示例调用
if __name__ == "__main__":
    time_slot_results, cost_records = get_real_flow_mb_in_t_time(total_slots=100, onlineFL_T=10,p_value=5)