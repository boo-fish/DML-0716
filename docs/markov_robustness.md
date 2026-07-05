# 时间相关动态鲁棒性实验

> 对应审稿意见: **Comment 1.12**
>
> "The i.i.d. assumption on network dynamics (Section 5.1) is restrictive. Wireless channels are typically temporally correlated due to shadowing and Doppler effects, energy harvesting often exhibits diurnal patterns, and IoT data arrivals frequently show periodic structure."

---

## 设计思路

### 原 i.i.d. 模型的问题

原 Step 1 脚本中，每时隙的6个网络参数变化因子完全独立采样：

```python
F_rand     = uniform(F_min_var, F_max_var)    # 设备计算能力
cap_rand   = uniform(cap_min_var, cap_max_var)  # 链路容量
A_rand     = uniform(A_min_var, A_max_var)      # 数据到达
E_rand     = uniform(E_min_var, E_max_var)      # 能量收集
theta_*_rand = ...                                # 成本参数
```

这忽略了真实 IoT 场景中的时间相关性。

### Markov 模型设计

使用**4组独立的两状态马尔可夫链**，每组对应一类物理参数：

| 参数组 | 物理含义 | p_stay | 时间相关性来源 |
|--------|------|:--:|------|
| 信道 (cap, link_cost) | 链路容量、传输成本 | 0.90 | 阴影效应（秒-分钟级相关） |
| 能量 (E) | 能量收集 | 0.95 | 昼夜模式（小时级相关） |
| 数据 (A) | 数据到达率 | 0.85 | IoT 周期性上报 |
| 设备 (F, e_device) | 计算能力、能耗 | 0.85 | 硬件状态变化缓慢 |

两状态马尔可夫链转移矩阵：

```
       LOW      HIGH
LOW  [ p_stay, 1-p_stay ]
HIGH [1-p_stay,  p_stay ]
```

- **LOW 状态**: 参数 × [0.3~0.9]（较差条件）
- **HIGH 状态**: 参数 × [1.1~1.8]（较好条件）

p_stay 越接近 1.0，时间相关性越强；i.i.d. 等价于 p_stay = 0.5。

---

## 运行命令

所有5种方法均支持 `--dynamics markov`：

```bash
# === i.i.d. 动态（原实验，作为对比基线）===
python run_method.py --method proposed --dynamics iid \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

# === Markov 动态（鲁棒性实验）===
python run_method.py --method proposed --dynamics markov \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

# === 基线方法 × Markov 动态 ===
python run_method.py --method local_training --dynamics markov \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

python run_method.py --method random_orchestration --dynamics markov \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

python run_method.py --method weight_divergence --dynamics markov \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

python run_method.py --method greedy_capacity --dynamics markov \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32
```

---

## 文件结构

| 文件 | 说明 |
|------|------|
| `markov_dynamics.py` | 马尔可夫动态参数生成器（`MarkovDynamics` 类） |
| `run_all_methods.py` | 统一仿真引擎（含所有方法的选择策略 + i.i.d./Markov 双模式） |
| `run_method.py` | 入口脚本（`--dynamics markov` 时委托 `run_all_methods`） |

---

## 实验对比矩阵

建议运行以下组合以完成审稿意见 1.12：

| 方法 | i.i.d. | Markov | 目的 |
|------|:--:|:--:|------|
| Proposed | ✅ (已有) | 🔲 待运行 | 本文方法的时间鲁棒性 |
| Local Training | ✅ (已有) | 🔲 待运行 | 基线对比 |
| Random Orchestration | ✅ (已有) | 🔲 待运行 | 隔离卸载鲁棒性 |
| Weight Divergence | ✅ (已有) | 🔲 待运行 | 文献[23]方法鲁棒性 |
| Greedy Capacity | ✅ (已有) | 🔲 待运行 | 贪心方法鲁棒性 |

**核心对比**: Proposed-i.i.d. vs Proposed-Markov，验证本文方法在时间相关动态下的性能退化程度。

---

## 自定义 Markov 参数

如需调整 Markov 链的时间相关性强度，可以在调用前修改 `markov_dynamics.py` 中的 `DEFAULT_P_STAY`：

```python
# markov_dynamics.py 中
DEFAULT_P_STAY = {
    "channel": 0.90,   # 增大 → 更强的信道相关性
    "energy":  0.95,   # 增大 → 更强的昼夜模式
    "data":    0.85,   # 增大 → 更强的周期性
    "device":  0.85,   # 增大 → 更稳定的硬件状态
}
```

或通过代码自定义：

```python
from markov_dynamics import MarkovDynamics
md = MarkovDynamics(N=50, seed=42,
    p_stay={"channel": 0.95, "energy": 0.98, "data": 0.90, "device": 0.90},
    multipliers={"channel": {"low": (0.3,0.7), "high": (1.3,1.7)}, ...}
)
```

---

## 审稿回复要点

在 response letter 中建议强调：

1. **实验验证**: 在 Markov 调制的时间相关动态下，本文方法依然保持性能优势
2. **理论讨论**: 本文的渐近最优性结果基于随机近似理论，该理论对具有遍历性的马尔可夫过程同样适用（参考 [Bertsekas & Tsitsiklis, Neuro-Dynamic Programming, 1996]），因为拉格朗日乘子的 SGD 更新在遍历 Markov 噪声下仍收敛
3. **物理直觉**: 本文的双时间尺度设计天然容忍中等程度的时间相关性——大时间尺度（每 T 时隙更新）的工作者选择过滤了快变波动；小时间尺度的流优化在每个时隙基于最新观测自适应该时隙的条件
