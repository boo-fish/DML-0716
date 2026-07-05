"""
实验运行入口 — 支持不同方法切换 × 不同动态模型

用法:
    python run_method.py --method proposed [--dynamics iid] [其余参数...]
    python run_method.py --method proposed --dynamics markov [其余参数...]
    python run_method.py --method local_training --dynamics markov ...

方法 (--method):
    proposed, local_training, random_orchestration,
    weight_divergence, greedy_capacity

动态模型 (--dynamics):
    iid     — 原始 i.i.d. 随机变化 (默认)
    markov  — 马尔可夫调制时间相关动态 (审稿意见 1.12)
"""

import sys
import argparse
import importlib


METHOD_MAP = {
    "proposed": "onlineFL_step1_全参数固定的仿真模板代码_用于获取每一time的实际训练量_FINAL",
    "local_training": "baseline_local_training",
    "random_orchestration": "baseline_random_orchestration",
    "weight_divergence": "baseline_weight_divergence",
    "greedy_capacity": "baseline_greedy_capacity",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", type=str, default="proposed",
                        choices=list(METHOD_MAP.keys()),
                        help="选择运行方法")
    parser.add_argument("--dynamics", type=str, default="iid",
                        choices=["iid", "markov"],
                        help="动态模型: iid (默认) / markov (时间相关)")
    args, remaining = parser.parse_known_args(argv)
    return args.method, args.dynamics, remaining


def main():
    method_name, dynamics, remaining_argv = parse_args()

    print(f"{'='*60}")
    print(f"  方法: {method_name}")
    print(f"  动态模型: {dynamics}")
    print(f"{'='*60}\n")

    if dynamics == "markov":
        # 使用 run_all_methods 中的统一仿真引擎
        import run_all_methods as ram
        from run_all_methods import MarkovDynamicsWrapper
        dyn_model = MarkovDynamicsWrapper(N=50, seed=42)

        def wrapped_get_flow(total_slots=None, onlineFL_T=None, p_value=20, A_min=30, cap_min=20):
            return ram.get_real_flow_mb_in_t_time(
                total_slots=total_slots, onlineFL_T=onlineFL_T,
                p_value=p_value, A_min=A_min, cap_min=cap_min,
                method=method_name, dynamics_model=dyn_model)

        step2_module_name = "onlineFL_step2_Online_FL-FINAL"
        step2 = importlib.import_module(step2_module_name)
        step2.get_real_flow_mb_in_t_time = wrapped_get_flow
    else:
        # i.i.d. 模式：使用原有各方法独立脚本
        module_name = METHOD_MAP[method_name]
        step1_module = importlib.import_module(module_name)
        get_flow = step1_module.get_real_flow_mb_in_t_time

        step2_module_name = "onlineFL_step2_Online_FL-FINAL"
        step2 = importlib.import_module(step2_module_name)
        step2.get_real_flow_mb_in_t_time = get_flow

    sys.argv = [sys.argv[0]] + remaining_argv
    step2.main()


if __name__ == "__main__":
    main()
