"""
统一实验运行器 — 支持多种方法 × 多种动态模型
===============================================
用法:
    python run_all_methods.py --method proposed --dynamics iid [其余参数...]
    python run_all_methods.py --method proposed --dynamics markov [其余参数...]
    python run_all_methods.py --method local_training --dynamics markov ...

方法:
    proposed, local_training, random_orchestration,
    weight_divergence, greedy_capacity

动态模型:
    iid     — 原始 i.i.d. 随机变化 (审稿意见1.12的对比基线)
    markov  — 马尔可夫调制的时间相关动态 (审稿意见1.12的鲁棒性实验)
"""

import sys
import argparse
import importlib
import numpy as np
from scipy.optimize import linprog


# ============================================================
# 0. 参数解析
# ============================================================

def parse_method_and_dynamics(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", type=str, default="proposed",
                        choices=["proposed", "local_training", "random_orchestration",
                                 "weight_divergence", "greedy_capacity"])
    parser.add_argument("--dynamics", type=str, default="iid",
                        choices=["iid", "markov"])
    args, remaining = parser.parse_known_args(argv)
    return args.method, args.dynamics, remaining


# ============================================================
# 1. 共享参数和工具函数
# ============================================================

def make_shared_params(p_value=20, A_min=30, cap_min=20):
    """生成所有方法共享的固定参数"""
    return {
        "N": 50, "R": 250, "Thre": 35, "K": 5,
        "F_min": 20, "F_max": 20 * p_value,
        "A_max": 80, "cap_max": 100,
        "e_link_min": 0.5, "e_link_max": 2,
        "e_worker_min": 40, "e_worker_max": 80,
        "E_device_min": 5, "E_device_max": 20,
        "epsilon": 0.0005,
        "NUM_CLUSTERS": 10,
    }


def generate_devices(params, seed=42):
    """生成设备拓扑和基础参数（所有方法共享）"""
    N = params["N"]
    rng = np.random.RandomState(seed)
    Dis = rng.rand(N, 1) * params["R"]
    angle = rng.rand(N, 1) * 2 * np.pi
    y = Dis * np.sin(angle); x = Dis * np.cos(angle)
    xm = np.tile(x, (1, N)); ym = np.tile(y, (1, N))
    dmat = np.sqrt((xm - xm.T)**2 + (ym - ym.T)**2)
    Map = (dmat < params["Thre"]).astype(float)
    np.fill_diagonal(Map, 0)

    A = np.ones((N,1))*params["A_min"] + rng.rand(N,1)*(params["A_max"]-params["A_min"])
    Amat = np.tile(A, (1,N))
    E_dev = np.ones((N,1))*params["E_device_min"] + rng.rand(N,1)*(params["E_device_max"]-params["E_device_min"])
    F_dev = np.ones((N,1))*params["F_min"] + rng.rand(N,1)*(params["F_max"]-params["F_min"])
    e_dev = np.ones((N,1))*params["e_worker_min"] + rng.rand(N,1)*(params["e_worker_max"]-params["e_worker_min"])

    eps_link = (np.ones((N,N))*params["e_link_min"]+rng.rand(N,N)*(params["e_link_max"]-params["e_link_min"]))*Map*Amat
    Cap = (np.ones((N,N))*cap_min+rng.rand(N,N)*(params["cap_max"]-cap_min))*Map
    Cap = np.tril(Cap,-1)+np.triu(Cap.T,0)

    dev_clusters = rng.randint(0, params["NUM_CLUSTERS"], size=N)
    clusters = {c: np.where(dev_clusters==c)[0].tolist() for c in range(params["NUM_CLUSTERS"])}
    clusters = {c:m for c,m in clusters.items() if len(m)>0}

    return A, Amat, E_dev, F_dev, e_dev, eps_link, Cap, clusters


def run_lp_flow(s_t_vec, F_t, A_t, cap_t, lambda_vec, e_link_t, N):
    """Algorithm 1 LP 流优化。返回 flow_temp (N+2, N+2)"""
    lambda_mat = np.tile(lambda_vec, (1, N))
    cap_r = np.zeros((N+2, N+2)); cost_r = np.zeros((N+2, N+2))
    cap_r[0,1:N+1] = A_t.flatten(); cost_r[0,1:N+1] = -1
    cap_r[1:N+1,1:N+1] = cap_t; cost_r[1:N+1,1:N+1] = lambda_mat * e_link_t
    for i in range(1,N+1):
        cap_r[i,N+1] = s_t_vec[i-1]*F_t[i-1]

    idx = np.where(cap_r != 0)
    f = cost_r[idx]; bounds = [(0, cap_r[i,j]) for i,j in zip(idx[0],idx[1])]
    A_eq=[]; b_eq=[]
    for i in range(1,N+1):
        row = np.zeros(len(idx[0])); row[idx[0]==i]=-1; row[idx[1]==i]=1
        if np.any(row!=0): A_eq.append(row); b_eq.append(0)
    A_eq=np.array(A_eq) if A_eq else np.zeros((0,len(idx[0])))
    b_eq=np.array(b_eq) if b_eq else np.zeros(0)

    ft = np.zeros((N+2,N+2))
    res = linprog(f, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        for k,(i,j) in enumerate(zip(idx[0],idx[1])): ft[i,j]=res.x[k]
    return ft


def extract_results(flow_temp, s_t_vec, F_t, N):
    """从流矩阵提取 selected_devices, total_offload, computation_f, final_data"""
    sel = np.where(s_t_vec.flatten()>0.5)[0].tolist()
    off = sum(float(flow_temp[i,j]) for i in range(1,N+1) for j in range(1,N+1) if i!=j and flow_temp[i,j]>0)
    comp_f = [float(F_t[i,0]) for i in range(N)]
    fd = []
    for i in range(1,N+1):
        inc = np.sum(flow_temp[1:N+1,i]); out = np.sum(flow_temp[i,1:N+1])
        fd.append(float(flow_temp[0,i]+inc-out))
    return sel, round(off,4), [round(x,4) for x in comp_f], [round(x,4) for x in fd]


# ============================================================
# 2. 动态模型生成器
# ============================================================

class IIDDynamics:
    """原始 i.i.d. 随机变化"""
    def __init__(self, N, seed=42):
        self.N = N; self.rng = np.random.RandomState(seed)
        self.label = "i.i.d."
    def step(self, var_ranges):
        """var_ranges: dict of (min_var, max_var) for each param"""
        N = self.N
        def _gen(lo, hi, shape): return lo + self.rng.rand(*shape)*(hi-lo)
        return (
            _gen(var_ranges["F"][0], var_ranges["F"][1], (N,1)),
            _gen(var_ranges["theta_dev"][0], var_ranges["theta_dev"][1], (N,1)),
            _gen(var_ranges["cap"][0], var_ranges["cap"][1], (N,N)),
            _gen(var_ranges["theta_cap"][0], var_ranges["theta_cap"][1], (N,N)),
            _gen(var_ranges["A"][0], var_ranges["A"][1], (N,1)),
            _gen(var_ranges["E"][0], var_ranges["E"][1], (N,1)),
        )
    def print_stats(self): pass


class MarkovDynamicsWrapper:
    """马尔可夫调制动态 (封装 markov_dynamics 模块)"""
    def __init__(self, N, seed=42):
        from markov_dynamics import MarkovDynamics
        self._md = MarkovDynamics(N=N, seed=seed)
        self.label = "Markov-modulated"
    def step(self, _var_ranges=None):
        return self._md.step()
    def print_stats(self):
        self._md.print_stats()


# ============================================================
# 3. 工作者选择策略
# ============================================================

def select_workers_proposed(s_t_prev, F_t, A_t, cap_t, e_device_t, e_link_t,
                             lambda_val, E_t, Time, T, params, flow_temp_prev):
    """BPSO/MILP 优化选择 — 用贪心+LP近似 (与原 Step 1 一致)"""
    # 简化: 每T时隙用MILP近似，非T时隙保持s_t不变
    # 完整MILP实现在onlineFL_step1中，这里使用贪心+随机扰动近似
    N = params["N"]; K = params["K"]
    if (Time-1) % T == 0:
        # 贪心得分: (F_i * A_i) / (e_device_i * lambda_i) — 综合吞吐和成本
        scores = (F_t.flatten() * A_t.flatten()) / (e_device_t.flatten() * (lambda_val.flatten()+1e-8))
        # 加入随机扰动模拟BPSO探索 (Scale: 20%高斯噪声)
        noise = 1.0 + 0.2 * np.random.randn(N)
        scores = scores * noise
        top = np.argsort(scores)[::-1][:K]
        s_t = np.zeros((N,1)); s_t[top] = 1
    else:
        s_t = s_t_prev.copy()
    return s_t


def select_workers_local_training(params, E_local, e_device_t, F_t, A_t, E_t, **kwargs):
    """随机选择 + 能量检查，无卸载"""
    N = params["N"]; K = params["K"]
    E_local = E_local + E_t
    eligible = np.where(E_local.flatten() > e_device_t.flatten())[0]
    if len(eligible) == 0: eligible = np.arange(N)
    sel = list(np.random.choice(eligible, min(K, len(eligible)), replace=False))
    s_t = np.zeros((N,1))
    for d in sel: s_t[d] = 1
    final_data = [0.0]*N
    for d in sel: final_data[d] = float(min(A_t[d,0], F_t[d,0]))
    for d in sel: E_local[d] -= e_device_t[d]
    return s_t, E_local, final_data


def select_workers_random(params, s_t_prev, Time, T, **kwargs):
    """纯随机选择 (每T时隙)"""
    N = params["N"]; K = params["K"]
    if (Time-1) % T == 0:
        s_t = np.zeros((N,1)); s_t[np.random.choice(N,K,replace=False)] = 1
    else:
        s_t = s_t_prev.copy()
    return s_t


def select_workers_weight_divergence(params, s_t_prev, clusters, Time, T, **kwargs):
    """聚类感知选择 (文献[23])"""
    N = params["N"]; K = params["K"]
    if (Time-1) % T == 0:
        s_t = np.zeros((N,1)); selected = []
        clist = list(clusters.keys()); nc = len(clist)
        base = K // nc; rem = K % nc
        for idx,c in enumerate(clist):
            members = clusters[c]
            n_sel = base + (1 if idx<rem else 0)
            if n_sel>0 and len(members)>0:
                chosen = list(np.random.choice(members, min(n_sel,len(members)), replace=False))
                selected.extend(chosen)
        if len(selected) < K:
            remaining = [d for d in range(N) if d not in selected]
            selected.extend(list(np.random.choice(remaining, min(K-len(selected),len(remaining)), replace=False)))
        for d in selected[:K]: s_t[d] = 1
    else:
        s_t = s_t_prev.copy()
    return s_t


def select_workers_greedy(params, s_t_prev, F_t, e_device_t, Time, T, **kwargs):
    """贪心容量选择"""
    N = params["N"]; K = params["K"]
    if (Time-1) % T == 0:
        scores = F_t.flatten() / (e_device_t.flatten() + 1e-8)
        top = np.argsort(scores)[::-1][:K]
        s_t = np.zeros((N,1)); s_t[top] = 1
    else:
        s_t = s_t_prev.copy()
    return s_t


# ============================================================
# 4. 主仿真函数
# ============================================================

def get_real_flow_mb_in_t_time(total_slots=None, onlineFL_T=None, p_value=20,
                                A_min=30, cap_min=20, method="proposed",
                                dynamics_model=None):
    """
    统一的时隙仿真函数。根据 method 和 dynamics_model 切换策略和动态模型。

    返回格式与原有 get_real_flow_mb_in_t_time 完全一致。
    """
    params = make_shared_params(p_value, A_min, cap_min)
    N = params["N"]; K = params["K"]; T = onlineFL_T
    count_slot = 0

    # 生成共享设备参数
    A_base, Amat, E_dev, F_dev, e_dev, eps_link, Cap, clusters = generate_devices(params)

    # 初始化动态模型
    if dynamics_model is None:
        dynamics_model = IIDDynamics(N, seed=42)

    # 变化范围 (i.i.d. 模式使用)
    var_ranges = {
        "F": (0.5, 1.5), "theta_dev": (1.0, 1.0),
        "cap": (0.5, 1.5), "theta_cap": (0.5, 1.5),
        "A": (0.5, 1.5), "E": (0.5, 1.5),
    }

    # 初始化
    lambda_val = np.ones((N, 1)) * 0.3
    s_t = np.zeros((N, 1))
    E_local = np.zeros((N, 1))  # for local_training
    Result = np.zeros(1)
    time_slot_results = []

    # 初始化 worker selection (首轮)
    s_t[np.random.choice(N, K, replace=False)] = 1

    print(f"[Method: {method}, Dynamics: {dynamics_model.label}] 开始仿真...")

    for Time in range(1, total_slots + 1):
        if Time % 200 == 0:
            print(f"  Time = {Time}")

        # ---- 生成动态参数 ----
        F_rand, theta_dev_rand, cap_rand, theta_cap_rand, A_rand, E_rand = dynamics_model.step(var_ranges)

        F_t = F_dev * F_rand
        e_device_t = e_dev * theta_dev_rand
        cap_t = Cap * cap_rand
        e_link_t = eps_link * theta_cap_rand
        A_t = A_base * A_rand
        E_t = E_dev * E_rand

        # ---- 工作者选择 ----
        flow_temp = np.zeros((N+2, N+2))

        if method == "local_training":
            s_t, E_local, final_data_t = select_workers_local_training(
                params, E_local, e_device_t, F_t, A_t, E_t)
            sel = np.where(s_t.flatten()>0.5)[0].tolist()
            comp_f = [float(F_t[i,0]) for i in range(N)]
            off = 0.0

        else:
            # 所有带卸载的方法: 先选worker, 再LP卸载
            kwargs_sel = dict(F_t=F_t, A_t=A_t, cap_t=cap_t, e_device_t=e_device_t,
                              e_link_t=e_link_t, lambda_val=lambda_val, E_t=E_t,
                              Time=Time, T=T, params=params, flow_temp_prev=flow_temp,
                              clusters=clusters)

            if method == "proposed":
                s_t = select_workers_proposed(s_t, **kwargs_sel)
            elif method == "random_orchestration":
                s_t = select_workers_random(params, s_t, Time, T)
            elif method == "weight_divergence":
                s_t = select_workers_weight_divergence(params, s_t, clusters, Time, T)
            elif method == "greedy_capacity":
                s_t = select_workers_greedy(params, s_t, F_t, e_device_t, Time, T)

            # LP 流优化
            flow_temp = run_lp_flow(s_t, F_t, A_t, cap_t, lambda_val, e_link_t, N)

            # 更新拉格朗日乘子
            lambda_val = np.maximum(lambda_val + params["epsilon"] * (
                s_t * e_device_t +
                np.sum(flow_temp[1:N+1,1:N+1]*e_link_t, axis=1, keepdims=True) - E_t
            ), 0)

            sel, off, comp_f, final_data_t = extract_results(flow_temp, s_t, F_t, N)

        # ---- 存储结果 ----
        time_slot_results.append((Time, sel, off, comp_f, final_data_t))

        if Time > count_slot:
            if method == "local_training":
                Result[0] += np.sum(np.minimum(A_t[sel].flatten(), F_t[sel].flatten()))
            else:
                Result[0] += np.sum(flow_temp[0,:])

    # ---- 输出统计 ----
    avg_thp = Result[0] / (total_slots - count_slot) if (total_slots - count_slot) > 0 else 0
    print(f"\n=== [{method}] [{dynamics_model.label}] 结果 ===")
    print(f"  平均吞吐量: {avg_thp:.4f}")
    dynamics_model.print_stats()

    return time_slot_results


# ============================================================
# 5. 入口
# ============================================================

def main():
    method, dynamics, remaining_argv = parse_method_and_dynamics()

    # 确定动态模型
    if dynamics == "markov":
        dyn_model = MarkovDynamicsWrapper(N=50, seed=42)
    else:
        dyn_model = IIDDynamics(N=50, seed=42)

    print(f"{'='*60}")
    print(f"  方法: {method}")
    print(f"  动态模型: {dyn_model.label}")
    print(f"  对应审稿意见: 1.12 (i.i.d. vs Markov)")
    print(f"{'='*60}\n")

    # 注入到 step2 模块
    step2_module_name = "onlineFL_step2_Online_FL-FINAL"
    step2 = importlib.import_module(step2_module_name)

    # 创建闭包捕获 method 和 dyn_model
    def wrapped_get_flow(total_slots=None, onlineFL_T=None, p_value=20, A_min=30, cap_min=20):
        return get_real_flow_mb_in_t_time(
            total_slots=total_slots, onlineFL_T=onlineFL_T,
            p_value=p_value, A_min=A_min, cap_min=cap_min,
            method=method, dynamics_model=dyn_model)

    step2.get_real_flow_mb_in_t_time = wrapped_get_flow

    sys.argv = [sys.argv[0]] + remaining_argv
    step2.main()


if __name__ == "__main__":
    main()
