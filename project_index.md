# 项目索引

> 稿件: Optimal Online Orchestration of Scalable Federated Learning for Massive Internet-of-Things (FGCS-D-26-02280)

## 文档类 (docs/)

- [docs/review_comments.md](docs/review_comments.md) — FGCS期刊审稿意见整理（英文原文 + 中文翻译，含汇总表）
- [docs/review_classification.md](docs/review_classification.md) — 审稿意见分类标记（按6个工作量维度标记）
- [docs/experiment_revision_order.md](docs/experiment_revision_order.md) — 实验修订顺序（6步，含依赖关系和预估时间线）
- [docs/baseline_methods.md](docs/baseline_methods.md) — 基线方法说明文档（4个基线 + Proposed）
- [docs/markov_robustness.md](docs/markov_robustness.md) — Markov鲁棒性实验说明（审稿意见1.12）
- [docs/progress.md](docs/progress.md) — 项目进度记录

## 运行入口

- [run_method.py](run_method.py) — Python 方法切换包装器（--method + --dynamics iid/markov）
- [run_all_methods.py](run_all_methods.py) — 统一仿真引擎（支持全部5种方法 × 2种动态模型）

## 主脚本

- [onlineFL_step1_全参数固定的仿真模板代码_用于获取每一time的实际训练量_FINAL.py](onlineFL_step1_全参数固定的仿真模板代码_用于获取每一time的实际训练量_FINAL.py) — Step 1: 全参数固定仿真，获取每time实际训练量（含 Proposed + Static + Local 基线）
- [onlineFL_step2_Online_FL-FINAL.py](onlineFL_step2_Online_FL-FINAL.py) — Step 2: 在线FL编排主程序
- [baseline_local_training.py](baseline_local_training.py) — 基线1: Local Training
- [baseline_random_orchestration.py](baseline_random_orchestration.py) — 基线2: Random Orchestration
- [baseline_weight_divergence.py](baseline_weight_divergence.py) — 基线3: Weight Divergence (文献[23])
- [baseline_greedy_capacity.py](baseline_greedy_capacity.py) — 基线4: Greedy Capacity
- [markov_dynamics.py](markov_dynamics.py) — Markov调制动态参数生成器（审稿意见1.12）

## 模型层 (models/)

- [models/Fed.py](models/Fed.py) — 联邦学习聚合/训练逻辑
- [models/Nets.py](models/Nets.py) — 神经网络模型定义（VGG11等）
- [models/Update.py](models/Update.py) — 本地更新逻辑
- [models/Update_benchmark.py](models/Update_benchmark.py) — 基线方法本地更新
- [models/Update_dml.py](models/Update_dml.py) — DML（本文方法）本地更新
- [models/test.py](models/test.py) — 模型测试脚本
- [models/\_\_init\_\_.py](models/__init__.py) — 包初始化

## 工具层 (utils/)

- [utils/options.py](utils/options.py) — 命令行参数解析
- [utils/network_environment.py](utils/network_environment.py) — 网络环境模拟（信道、能量等）
- [utils/graph_builder.py](utils/graph_builder.py) — 图构建器
- [utils/get_info_for_CADIB.py](utils/get_info_for_CADIB.py) — CADIB方法信息提取
- [utils/get_offload_dict_dml.py](utils/get_offload_dict_dml.py) — DML方法卸载字典
- [utils/get_total_MB_of_dataset.py](utils/get_total_MB_of_dataset.py) — 数据集大小计算
- [utils/sampling_CADIB.py](utils/sampling_CADIB.py) — CADIB采样
- [utils/sampling_dml.py](utils/sampling_dml.py) — DML采样
- [utils/sampling_benchmark.py](utils/sampling_benchmark.py) — 基线采样
- [utils/计算cifar10数据集的大小.py](utils/计算cifar10数据集的大小.py) — CIFAR-10数据集大小统计
- [utils/\_\_init\_\_.py](utils/__init__.py) — 包初始化

## 辅助脚本 (others/)

- [others/benchmark_cost.py](others/benchmark_cost.py) — 基线方法成本计算
- [others/local_training.py](others/local_training.py) — 本地训练脚本
- [others/view_noiid分布.py](others/view_noiid分布.py) — 查看non-IID数据分布
- [others/提取Ai_actdata文件中的实际训练数据总和.py](others/提取Ai_actdata文件中的实际训练数据总和.py) — 从actdata提取训练数据总和
- [others/读取pkl文件内容.py](others/读取pkl文件内容.py) — pkl文件内容读取
- [others/test.py](others/test.py) — 辅助测试脚本

## 配置

- [.vscode/settings.json](.vscode/settings.json) — VSCode工作区配置
