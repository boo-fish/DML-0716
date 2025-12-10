# test_convex_relaxation.py
import numpy as np
import matplotlib.pyplot as plt
from network_environment import NetworkEnvironment
from convex_relaxation_optimizer import ConvexRelaxationOptimizer
from utils.options import args_parser


def generate_data_arrival_scenarios(num_workers, base_arrival, arrival_range=(0, 40), num_scenarios=21):
    """
    生成不同数据到达量的场景

    Args:
        num_workers: 工人数量
        base_arrival: 基础到达模式
        arrival_range: 数据到达量范围
        num_scenarios: 场景数量

    Returns:
        list: 不同数据到达量的列表
    """
    scenarios = []
    arrival_levels = np.linspace(arrival_range[0], arrival_range[1], num_scenarios)

    for level in arrival_levels:
        if level == 0:
            Ai = np.zeros(num_workers)
        else:
            # 保持相对比例，缩放基础到达模式
            Ai = base_arrival * level / np.mean(base_arrival)
        scenarios.append((level, Ai))

    return scenarios


def run_convex_relaxation_experiments():
    """
    运行凸松弛优化实验，测试不同数据到达量下的性能
    """
    # 获取命令行参数
    args = args_parser()

    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 创建网络环境
    num_workers = 50
    num_servers = 10
    num_clients = args.num_users
    link_capacity_range = (50, 150)
    worker_capacity_range = (5, 45)
    server_capacity_range = (30, 60)
    data_arrival_range = (10, 30)  # 基础数据到达范围
    # alpha = args.alpha
    alpha = 0.4

    print("=" * 70)
    print("凸松弛优化方法 - 不同数据到达量下的性能测试")
    print("=" * 70)
    print(f"工人数量: {num_workers}")
    print(f"参数服务器数量: {num_servers}")
    print(f"数据到达量测试范围: 0-40")
    print(f"α值: {alpha}")
    print("=" * 70)

    # 创建网络环境
    env_obj = NetworkEnvironment(
        num_workers=num_workers,
        num_servers=num_servers,
        num_clients=num_clients,
        link_capacity_range=link_capacity_range,
        worker_capacity_range=worker_capacity_range,
        server_capacity_range=server_capacity_range,
        data_arrival_range=data_arrival_range,
        alpha=alpha
    )

    env = env_obj.get_network()

    # 创建凸松弛优化器
    optimizer = ConvexRelaxationOptimizer(env)

    # 生成基础到达模式
    base_Ai = env['data_arrival']

    # 生成不同数据到达量的场景
    scenarios = generate_data_arrival_scenarios(num_workers, base_Ai, arrival_range=(0, 40), num_scenarios=21)

    # 存储结果
    results = []

    print("\n开始运行凸松弛优化...")
    print("-" * 70)
    print(f"{'数据到达量':<12} {'吞吐量':<12} {'总成本':<12} {'成本效率':<12} {'状态':<15} {'求解时间(s)':<12}")
    print("-" * 70)

    for level, Ai in scenarios:
        # 运行凸松弛优化
        result = optimizer.auto_detect_case(Ai)

        # 存储结果
        results.append({
            'arrival_level': level,
            'Ai': Ai.copy(),
            'throughput': result['throughput'],
            'total_cost': result['total_cost'],
            'cost_efficiency': result['cost_efficiency'],
            'status': result['status'],
            'solve_time': result['solve_time'],
            'feasible': result.get('feasible', True),
            'admission_ratio': result.get('admission_ratio', 1.0)
        })

        # 打印当前结果
        print(f"{level:<12.2f} {result['throughput']:<12.2f} "
              f"{result['total_cost']:<12.2f} {result['cost_efficiency']:<12.6f} "
              f"{result['status']:<15} {result['solve_time']:<12.2f}")

    print("-" * 70)

    # 分析结果
    analyze_results(results)

    # 绘制性能曲线
    plot_performance_curves(results)

    return results


def analyze_results(results):
    """
    分析实验结果
    """
    print("\n" + "=" * 70)
    print("实验结果分析")
    print("=" * 70)

    # 找到转折点（从可行到不可行的点）
    feasible_results = [r for r in results if r['feasible']]
    infeasible_results = [r for r in results if not r['feasible']]

    if feasible_results:
        max_feasible_arrival = max([r['arrival_level'] for r in feasible_results])
        print(f"最大可行数据到达量: {max_feasible_arrival:.2f}")

    # 找到最佳成本效率点
    valid_results = [r for r in results if r['cost_efficiency'] > 0 and not np.isinf(r['cost_efficiency'])]
    if valid_results:
        best_efficiency = max(valid_results, key=lambda x: x['cost_efficiency'])
        print(f"最佳成本效率点:")
        print(f"  数据到达量: {best_efficiency['arrival_level']:.2f}")
        print(f"  成本效率: {best_efficiency['cost_efficiency']:.6f}")
        print(f"  吞吐量: {best_efficiency['throughput']:.2f}")
        print(f"  总成本: {best_efficiency['total_cost']:.2f}")

    # 计算平均求解时间
    avg_solve_time = np.mean([r['solve_time'] for r in results])
    print(f"平均求解时间: {avg_solve_time:.2f}秒")

    # 统计求解状态
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    print("求解状态统计:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}次")


def plot_performance_curves(results):
    """
    绘制性能曲线
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 提取数据
    arrival_levels = [r['arrival_level'] for r in results]
    throughputs = [r['throughput'] for r in results]
    costs = [r['total_cost'] for r in results]
    efficiencies = [r['cost_efficiency'] for r in results]
    solve_times = [r['solve_time'] for r in results]
    admission_ratios = [r.get('admission_ratio', 1.0) for r in results]
    feasible_flags = [r['feasible'] for r in results]

    # 1. 吞吐量 vs 数据到达量
    ax1 = axes[0, 0]
    ax1.plot(arrival_levels, throughputs, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('数据到达量', fontsize=12)
    ax1.set_ylabel('吞吐量', fontsize=12)
    ax1.set_title('吞吐量 vs 数据到达量', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 2. 总成本 vs 数据到达量
    ax2 = axes[0, 1]
    # 过滤掉无穷大的成本
    valid_costs = [(x, y) for x, y in zip(arrival_levels, costs) if not np.isinf(y)]
    if valid_costs:
        x_vals, y_vals = zip(*valid_costs)
        ax2.plot(x_vals, y_vals, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('数据到达量', fontsize=12)
    ax2.set_ylabel('总成本', fontsize=12)
    ax2.set_title('总成本 vs 数据到达量', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. 成本效率 vs 数据到达量
    ax3 = axes[0, 2]
    # 过滤掉无效的效率值
    valid_efficiencies = [(x, y) for x, y in zip(arrival_levels, efficiencies) if y > 0 and not np.isinf(y)]
    if valid_efficiencies:
        x_vals, y_vals = zip(*valid_efficiencies)
        ax3.plot(x_vals, y_vals, 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('数据到达量', fontsize=12)
    ax3.set_ylabel('成本效率', fontsize=12)
    ax3.set_title('成本效率 vs 数据到达量', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 4. 求解时间 vs 数据到达量
    ax4 = axes[1, 0]
    ax4.plot(arrival_levels, solve_times, 'm-o', linewidth=2, markersize=6)
    ax4.set_xlabel('数据到达量', fontsize=12)
    ax4.set_ylabel('求解时间 (秒)', fontsize=12)
    ax4.set_title('求解时间 vs 数据到达量', fontsize=14)
    ax4.grid(True, alpha=0.3)

    # 5. 数据接纳率 vs 数据到达量
    ax5 = axes[1, 1]
    ax5.plot(arrival_levels, admission_ratios, 'c-o', linewidth=2, markersize=6)
    ax5.set_xlabel('数据到达量', fontsize=12)
    ax5.set_ylabel('数据接纳率', fontsize=12)
    ax5.set_title('数据接纳率 vs 数据到达量', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.1])

    # 6. 可行/不可行区域
    ax6 = axes[1, 2]
    feasible_arrival = [arrival_levels[i] for i in range(len(arrival_levels)) if feasible_flags[i]]
    infeasible_arrival = [arrival_levels[i] for i in range(len(arrival_levels)) if not feasible_flags[i]]

    if feasible_arrival:
        ax6.scatter(feasible_arrival, [0.5] * len(feasible_arrival),
                    color='green', s=100, label='可行', marker='o')
    if infeasible_arrival:
        ax6.scatter(infeasible_arrival, [0.5] * len(infeasible_arrival),
                    color='red', s=100, label='不可行', marker='x')

    ax6.set_xlabel('数据到达量', fontsize=12)
    ax6.set_ylabel('可行性', fontsize=12)
    ax6.set_title('问题可行性 vs 数据到达量', fontsize=14)
    ax6.set_ylim([0, 1])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convex_relaxation_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 额外绘制成本效率的详细分析图
    plt.figure(figsize=(10, 6))
    valid_points = [(x, y) for x, y in zip(arrival_levels, efficiencies) if y > 0 and not np.isinf(y)]
    if valid_points:
        x_vals, y_vals = zip(*valid_points)
        plt.plot(x_vals, y_vals, 'b-o', linewidth=2, markersize=8, markerfacecolor='yellow')
        plt.xlabel('数据到达量', fontsize=14)
        plt.ylabel('成本效率', fontsize=14)
        plt.title('凸松弛优化方法 - 成本效率分析', fontsize=16)
        plt.grid(True, alpha=0.3)

        # 标记最佳效率点
        if valid_points:
            best_idx = np.argmax(y_vals)
            plt.annotate(f'最佳点: ({x_vals[best_idx]:.1f}, {y_vals[best_idx]:.4f})',
                         xy=(x_vals[best_idx], y_vals[best_idx]),
                         xytext=(x_vals[best_idx] + 2, y_vals[best_idx]),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         fontsize=12)

        plt.savefig('convex_relaxation_efficiency_detail.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_results_to_file(results, filename='convex_relaxation_results.csv'):
    """
    将结果保存到CSV文件
    """
    import pandas as pd

    data = []
    for r in results:
        data.append({
            'arrival_level': r['arrival_level'],
            'throughput': r['throughput'],
            'total_cost': r['total_cost'],
            'cost_efficiency': r['cost_efficiency'],
            'status': r['status'],
            'solve_time': r['solve_time'],
            'feasible': r['feasible'],
            'admission_ratio': r.get('admission_ratio', 1.0)
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\n结果已保存到文件: {filename}")

    # 显示统计摘要
    print("\n结果统计摘要:")
    print(df.describe())


if __name__ == "__main__":
    # 运行凸松弛优化实验
    results = run_convex_relaxation_experiments()

    # 保存结果到文件
    save_results_to_file(results)

    # 打印总结
    print("\n" + "=" * 70)
    print("实验完成总结")
    print("=" * 70)
    print(f"总测试场景数: {len(results)}")
    print(f"数据到达量范围: 0-40")
    print(f"性能曲线图已保存为: convex_relaxation_performance.png")
    print(f"详细效率图已保存为: convex_relaxation_efficiency_detail.png")
    print(f"结果数据已保存为: convex_relaxation_results.csv")