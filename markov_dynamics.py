"""
Markov-Modulated Network Dynamics Module
========================================
针对审稿意见 1.12，提供具有时间相关性的网络动态参数生成。

原 Step 1 使用 i.i.d. 随机变量生成每时隙的网络参数变化。
本模块使用两状态马尔可夫链替代 i.i.d.，体现真实IoT场景中的时间相关性：

  - 无线信道 (cap, link cost):  阴影效应 → 高时间相关性 (p_stay=0.90)
  - 能量收集 (E):              昼夜模式 → 非常高时间相关性 (p_stay=0.95)
  - 数据到达 (A):              IoT周期上报 → 中等时间相关性 (p_stay=0.85)
  - 设备计算 (F, e_device):    相对稳定 → 中等时间相关性 (p_stay=0.85)

原理:
  两状态马尔可夫链: state ∈ {LOW=0, HIGH=1}
  状态转移矩阵: [[p_stay, 1-p_stay], [1-p_stay, p_stay]]
  LOW 状态: 参数×low_multiplier (较差条件)
  HIGH 状态: 参数×high_multiplier (较好条件)

用法:
    from markov_dynamics import MarkovDynamics

    md = MarkovDynamics(N=50, seed=42)
    for t in range(total_slots):
        F_rand, theta_device_rand, cap_rand, theta_cap_rand, A_rand, E_rand = md.step()
        # 与原始代码完全一致的用法
        F_t = F_device * F_rand
        ...
"""

import numpy as np


class MarkovDynamics:
    """
    马尔可夫调制的网络动态参数生成器。

    Parameters
    ----------
    N : int
        设备数量 (默认 50)
    seed : int
        随机种子 (默认 42)
    """

    # 默认马尔可夫停留概率（时间相关性强度）
    # 值越接近 1.0，时间相关性越强
    DEFAULT_P_STAY = {
        "channel": 0.90,    # 无线信道: 阴影效应影响，高相关性
        "energy":  0.95,    # 能量收集: 昼夜模式，非常高相关性
        "data":    0.85,    # 数据到达: IoT周期性，中等相关性
        "device":  0.85,    # 设备计算: 相对稳定，中等相关性
    }

    # 各状态的乘子范围
    # LOW 状态: [min_mult, 1.0) — 低于基线
    # HIGH 状态: [1.0, max_mult] — 高于基线
    DEFAULT_MULTIPLIERS = {
        "channel": {"low": (0.4, 0.8), "high": (1.2, 1.6)},  # 信道波动大
        "energy":  {"low": (0.3, 0.7), "high": (1.3, 1.8)},  # 能量波动大(昼夜)
        "data":    {"low": (0.5, 0.9), "high": (1.1, 1.5)},  # 数据波动中等
        "device":  {"low": (0.6, 0.9), "high": (1.1, 1.4)},  # 计算能力波动小
    }

    def __init__(self, N=50, seed=42, p_stay=None, multipliers=None):
        self.N = N
        self.rng = np.random.RandomState(seed)

        # 停留概率
        self.p_stay = p_stay if p_stay is not None else self.DEFAULT_P_STAY.copy()

        # 乘子范围
        self.multipliers = multipliers if multipliers is not None else self.DEFAULT_MULTIPLIERS.copy()

        # 为每个参数组初始化马尔可夫状态
        # state=1 表示 HIGH, state=0 表示 LOW
        # 初始状态随机
        self._state_channel = self.rng.randint(0, 2)    # cap + link_cost
        self._state_energy  = self.rng.randint(0, 2)    # E
        self._state_data    = self.rng.randint(0, 2)    # A
        self._state_device  = self.rng.randint(0, 2)    # F + e_device

        # 统计信息
        self.state_history = {
            "channel": [self._state_channel],
            "energy":  [self._state_energy],
            "data":    [self._state_data],
            "device":  [self._state_device],
        }

    def _transition(self, state, p_stay):
        """单步状态转移"""
        if self.rng.rand() < p_stay:
            return state  # 保持
        else:
            return 1 - state  # 翻转

    def _generate_multiplier(self, state, low_range, high_range, shape):
        """根据状态生成乘子矩阵"""
        if state == 0:  # LOW
            lo, hi = low_range
        else:  # HIGH
            lo, hi = high_range
        return lo + self.rng.rand(*shape) * (hi - lo)

    def step(self):
        """
        生成一个时隙的网络动态参数变化。

        Returns
        -------
        F_rand : ndarray (N, 1)
            设备计算能力变化因子
        theta_device_rand : ndarray (N, 1)
            设备能耗成本变化因子
        cap_rand : ndarray (N, N)
            链路容量变化因子
        theta_cap_rand : ndarray (N, N)
            链路传输成本变化因子
        A_rand : ndarray (N, 1)
            数据到达率变化因子
        E_rand : ndarray (N, 1)
            能量收集变化因子
        """
        N = self.N

        # ---- 1. 无线信道 (cap + link cost) ----
        self._state_channel = self._transition(self._state_channel, self.p_stay["channel"])
        mc = self.multipliers["channel"]
        cap_mult = self._generate_multiplier(self._state_channel, mc["low"], mc["high"], (N, N))
        link_cost_mult = self._generate_multiplier(self._state_channel, mc["low"], mc["high"], (N, N))

        # ---- 2. 能量收集 (E) ----
        self._state_energy = self._transition(self._state_energy, self.p_stay["energy"])
        me = self.multipliers["energy"]
        energy_mult = self._generate_multiplier(self._state_energy, me["low"], me["high"], (N, 1))

        # ---- 3. 数据到达 (A) ----
        self._state_data = self._transition(self._state_data, self.p_stay["data"])
        md = self.multipliers["data"]
        data_mult = self._generate_multiplier(self._state_data, md["low"], md["high"], (N, 1))

        # ---- 4. 设备计算 (F + device cost) ----
        self._state_device = self._transition(self._state_device, self.p_stay["device"])
        mdev = self.multipliers["device"]
        F_mult = self._generate_multiplier(self._state_device, mdev["low"], mdev["high"], (N, 1))
        device_cost_mult = self._generate_multiplier(self._state_device, mdev["low"], mdev["high"], (N, 1))

        # ---- 记录状态 ----
        self.state_history["channel"].append(self._state_channel)
        self.state_history["energy"].append(self._state_energy)
        self.state_history["data"].append(self._state_data)
        self.state_history["device"].append(self._state_device)

        return F_mult, device_cost_mult, cap_mult, link_cost_mult, data_mult, energy_mult

    def get_state_stats(self):
        """返回各状态的时间占比"""
        stats = {}
        for key, history in self.state_history.items():
            arr = np.array(history)
            stats[key] = {
                "high_ratio": float(np.mean(arr == 1)),
                "low_ratio": float(np.mean(arr == 0)),
                "transitions": int(np.sum(np.diff(arr) != 0)),
                "total_steps": len(history),
            }
        return stats

    def print_stats(self):
        """打印状态统计"""
        stats = self.get_state_stats()
        print("\n=== Markov Dynamics 状态统计 ===")
        for key, s in stats.items():
            print(f"  {key}: HIGH={s['high_ratio']:.1%}, LOW={s['low_ratio']:.1%}, "
                  f"切换次数={s['transitions']}/{s['total_steps']}")
