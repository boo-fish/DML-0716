"""
实验运行入口 — 支持不同方法切换

用法:
    python run_method.py --method proposed [其他参数...]
    python run_method.py --method local_training [其他参数...]
    python run_method.py --method random_orchestration [其他参数...]

方法说明:
    proposed           — 本文方法 (BPSO优化选择 + Algorithm 1卸载)
    local_training     — 基线1: 随机选择 + 无卸载
    random_orchestration — 基线2: 随机选择 + Algorithm 1卸载
"""

import sys
import argparse
import importlib


METHOD_MAP = {
    "proposed": "onlineFL_step1_全参数固定的仿真模板代码_用于获取每一time的实际训练量_FINAL",
    "local_training": "baseline_local_training",
    "random_orchestration": "baseline_random_orchestration",
}


def parse_method(argv=None):
    """从命令行参数中提取 --method，返回 (method_name, 剩余参数列表)"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", type=str, default="proposed",
                        choices=list(METHOD_MAP.keys()),
                        help="选择运行方法")
    args, remaining = parser.parse_known_args(argv)
    return args.method, remaining


def main():
    # 1. 提取 --method 并保持其余参数原样传递给原 main
    method_name, remaining_argv = parse_method()

    print(f"{'='*60}")
    print(f"  方法: {method_name}")
    print(f"  模块: {METHOD_MAP[method_name]}")
    print(f"{'='*60}\n")

    # 2. 动态导入对应的 get_real_flow_mb_in_t_time
    module_name = METHOD_MAP[method_name]
    step1_module = importlib.import_module(module_name)
    get_flow = step1_module.get_real_flow_mb_in_t_time

    # 3. 动态导入 step2 模块（文件名含连字符，不能用 import 语句）
    step2_module_name = "onlineFL_step2_Online_FL-FINAL"
    step2 = importlib.import_module(step2_module_name)
    step2.get_real_flow_mb_in_t_time = get_flow

    # 4. 将剩余参数写回 sys.argv 供 args_parser 解析
    sys.argv = [sys.argv[0]] + remaining_argv

    # 5. 运行原 main
    step2.main()


if __name__ == "__main__":
    main()
