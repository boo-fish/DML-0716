# 项目进度记录

## 2026年07月05日

**核心目的**: 实现两个基线方法脚本，为后续实验对比做准备

- ✅ 创建 `baseline_local_training.py`：Local Training 基线（随机工作者选择 + 无数据卸载）
- ✅ 创建 `baseline_random_orchestration.py`：Random Orchestration 基线（随机工作者选择 + Algorithm 1 卸载）
- ✅ 创建 `docs/baseline_methods.md`：基线方法说明文档（对比表、核心逻辑、使用方式）
- ✅ 创建 `baseline_weight_divergence.py`：基线3 Weight Divergence（文献[23] 聚类感知设备选择 + LP卸载）
- ✅ 创建 `baseline_greedy_capacity.py`：基线4 Greedy Capacity（贪心容量打分选择 + LP卸载）
- ✅ 更新 `run_method.py`：注册 weight_divergence / greedy_capacity 两种新方法
- ✅ 更新 `docs/baseline_methods.md`：补充基线3、4的说明，更新对比表和运行命令
- ✅ 现有共5种方法 × 2种动态模型 = 10种组合
- ✅ 审稿意见1.12: 创建 `markov_dynamics.py`（4组两状态Markov链，可配p_stay和乘子范围）
- ✅ 创建 `run_all_methods.py`：统一仿真引擎（含所有方法选择策略 + i.i.d./Markov双模式）
- ✅ 更新 `run_method.py`：新增 `--dynamics iid/markov` 参数
- ✅ 创建 `docs/markov_robustness.md`：Markov鲁棒性实验说明（含运行命令、参数说明、回复建议）
- ✅ 两个基线脚本输出格式与原 Step 1 脚本完全兼容，可直接替换 import 使用

**数据信息**:
- Baseline 1 (Local Training): 随机选K个 + 每设备数据 = min(A_i, F_i) + 无卸载
- Baseline 2 (Random Orchestration): 随机选K个(每T时隙) + LP流优化卸载 + 拉格朗日乘子
- 参数与原脚本一致: N=50, K=5, T=10/20, ε=0.0005

**下一步计划**:
- ⏳ 启动 Step 1：1.7 确保FL训练收敛（增加轮次、审查超参数）
- ⏳ 撰写回复信

---

## 2026年07月04日

**核心目的**: 整理FGCS期刊审稿意见，按工作量维度分类，为修订工作提供优先级指引

- ✅ 创建 `docs/review_comments.md`：29条审稿意见逐条整理（英文原文 + 中文翻译 + 汇总表）
- ✅ 创建 `docs/review_classification.md`：按6个维度分类标记（🧪实验/📊图表/📝正文/📐理论/💬讨论/🔧小修）
- ✅ 工作量统计：实验7条（高）、图表8条（中）、正文9条（中）、理论2条（中）、讨论4条（低-中）、小修12条（低）
- ✅ 策略建议：优先处理实验和理论，因为图表和正文依赖前两者结果

**数据信息**:
- 稿件编号: FGCS-D-26-02280
- 审稿人数量: 2位
- Reviewer #1: 14条主要 + 14条次要 = 28条
- Reviewer #2: 1条
- 修订截止日期: 2026年8月14日
- 需运行新实验: 7条（1.4/1.7/1.8/1.9/1.11/1.12 + 部分1.6）
- 需新增/修改图表: 8条

- ✅ 创建 `docs/experiment_revision_order.md`：实验修订顺序（6步，含依赖关系和预估时间线）

**下一步计划**:
- ⏳ 启动 Step 1：1.7 确保FL训练收敛（增加轮次、审查超参数）
- ⏳ 制定逐条回复策略和回复信大纲
- ⏳ 撰写回复信（response letter）
- ⏳ 准备修订稿
