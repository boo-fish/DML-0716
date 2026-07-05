# 基线方法说明文档

本文实现了两个基线方法，用于与所提出的双时间尺度在线编排方法进行对比实验。所有方法通过 `run_method.py` 统一入口运行。

---

## 运行命令

```bash
# 本文方法 (Proposed)
python run_method.py --method proposed \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

# 基线1: Local Training（随机选择 + 无卸载）
python run_method.py --method local_training \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32

# 基线2: Random Orchestration（随机选择 + 卸载）
python run_method.py --method random_orchestration \
    --dataset cifar10 --num_classes 10 --num_channels 3 \
    --model vgg11 --total_slots 160 --T 10 --gpu 0 \
    --p 20 --alpha 1.0 --total_mb 586 \
    --lr 0.001 --local_bs 32
```

`--method` 以外的所有参数均为原 `args_parser()` 支持的参数，完整透传。

---

## 基线方法对比

| 维度 | Proposed (本文) | Baseline 1: Local Training | Baseline 2: Random Orchestration |
|------|:--:|:--:|:--:|
| 工作者选择 | BPSO 优化（每 T 时隙） | 随机（基于能量可用性） | **随机**（每 T 时隙） |
| D2D 数据卸载 | Algorithm 1 流优化 | **无卸载** | Algorithm 1 流优化 |
| 拉格朗日乘子更新 | SGD 更新 | 无 | SGD 更新 |
| 能量约束 | 长期约束（拉格朗日） | 即时能量检查 | 长期约束（拉格朗日） |
| 每设备训练数据 | 通过卸载路由的 final_data | Ai 原始本地数据 | 通过卸载路由的 final_data |
| 对应文件 | `onlineFL_step1_..._FINAL.py` | `baseline_local_training.py` | `baseline_random_orchestration.py` |

---

## Baseline 1: Local Training

### 策略

- **工作者选择**: 每个时隙从能量充足的设备中随机选择 K 个工作节点
- **数据获取**: 无 D2D 卸载，每个工作节点仅使用自身的本地数据到达量（A_i）
- **训练数据量**: min(A_i(t), F_i(t)) — 受限于数据到达率和设备计算能力
- **能量模型**: 即时能量检查（E_i(t) > e_device），无长期能量约束

### 核心逻辑

```
每个时隙 t:
  1. 累积能量: E_i += E_i^arrival(t)
  2. 筛选可工作设备: E_i > e_device
  3. 随机选择 K 个
  4. 每设备训练数据 = min(A_i(t), F_i(t))
  5. 扣除能量: E_i -= e_device (选中设备)
  6. 无 D2D 卸载
```

### 对应审稿意见

- Comment 1.4: 该基线只包含随机选择 + 无卸载，缺乏优化的客户端选择

### 使用方式

```bash
python run_method.py --method local_training [其余参数...]
```

如需直接在代码中调用：

```python
from baseline_local_training import get_real_flow_mb_in_t_time

time_slot_results = get_real_flow_mb_in_t_time(
    total_slots=total_slots, onlineFL_T=T,
    p_value=p_value, A_min=A_min, cap_min=cap_min
)
```

---

## Baseline 2: Random Orchestration

### 策略

- **工作者选择**: 每 T 个时隙**随机**选择 K 个工作节点（替代 BPSO/MILP 优化）
- **数据卸载**: 保留 Algorithm 1 的 LP 流优化，在随机选择的工作节点间进行 D2D 数据卸载
- **训练数据量**: 通过卸载路由后的 final_data（与 Proposed 方法相同的数据流优化，但工作者是随机选的）
- **拉格朗日乘子**: 保留 SGD 更新，维持长期能量约束

### 核心逻辑

```
每个时隙 t:
  1. 每 T 时隙: 随机选择 K 个工作节点（替代 MILP）
  2. 构建残差图（源→设备→工作节点→虚拟汇）
  3. LP 求解最小成本流（Algorithm 1）
  4. 更新拉格朗日乘子 λ_i
  5. 计算卸载量和最终数据分配
```

### 与 Proposed 方法的区别

| 步骤 | Proposed | Random Orchestration |
|------|----------|---------------------|
| 大时间尺度工作者选择 | MILP 优化 s(t) | 随机选择 s(t) |
| 小时间尺度流优化 | LP（相同） | LP（相同） |
| 拉格朗日乘子更新 | SGD（相同） | SGD（相同） |

### 对应审稿意见

- Comment 1.4: 该基线包含卸载优化但缺乏优化的客户端选择，可用于分离"卸载"和"选择"各自对性能的贡献

### 使用方式

```python
# 在 onlineFL_step2_Online_FL-FINAL.py 中：
from baseline_random_orchestration import get_real_flow_mb_in_t_time as get_real_flow_mb_in_t_time_rand

# 调用方式与原函数完全一致
time_slot_results = get_real_flow_mb_in_t_time_rand(
    total_slots=total_slots, onlineFL_T=T,
    p_value=p_value, A_min=A_min, cap_min=cap_min
)
```

---

## 输出格式

两个基线脚本的 `get_real_flow_mb_in_t_time` 函数输出格式与原 Step 1 脚本完全一致：

```python
time_slot_results = [
    (Time,                    # int: 时隙编号
     selected_devices,        # list[int]: 选中的工作节点索引
     total_offload,           # float: D2D 卸载总量 (Baseline 1 恒为 0)
     device_computation_f,    # list[float]: 每设备计算能力 F_i(t), 长度 N
     final_data_amounts       # list[float]: 每设备最终训练数据量, 长度 N
    ),
    ...
]
```

因此可直接作为 `onlineFL_step2_Online_FL-FINAL.py` 中 `get_real_flow_mb_in_t_time` 的替代品使用。

---

## 参数说明

两个基线脚本均接受与原始函数相同的参数：

| 参数 | 含义 | 默认值 |
|------|------|:--:|
| `total_slots` | FL 训练总时隙数 | 由 main 传入 |
| `onlineFL_T` | 大时间尺度周期 | 由 main 传入 (10 或 20) |
| `p_value` | F_max / F_min 比值 | 20 |
| `A_min` | 设备数据到达率最小值 | 30 |
| `cap_min` | 链路容量最小值 | 20 |

内部固定参数（N=50, K=5, ε=0.0005 等）与原 Step 1 脚本保持一致。
