# @Date       2025/12/22 下午4:03
# @Author     2024级电子信息计算机方向 艾春慧
# @University MUC
import pickle
import os
from datetime import datetime

# ===================== 基础配置 =====================
# 目标准确率（与原代码保持一致）
TARGET_ACCURACY = 92

# 数据文件路径（沿用你原有配置，包含所有模型）
DATA_PATHS = {
    'proposed_iid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_True_DML_alpha_0.3.pkl',
    'proposed_noniid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_False_DML_alpha_0.3.pkl',
    'conventional_iid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_True_local_alpha_0.3.pkl',
    'conventional_noniid': '../saving/time/Ai_6_P_10_epoch_100_is_iid_False_local_alpha_0.3.pkl',
    'ramfl_iid': r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_60_is_iid_True_local_alpha_0.3_Final_Acc_92.4300_2025-12-15-12-23-41.pkl',
    'ramfl_noniid': r'D:\project\DML-0716\results-1217\RAMFL\Ai_6_P_10_epoch_155_is_iid_False_local_alpha_0.3_Final_Acc_92.4600_2025-12-16-01-13-51.pkl'
}


# ===================== 工具函数 =====================
def convert_to_python(value):
    """张量转标量（处理PyTorch张量等非Python原生类型）"""
    if hasattr(value, 'item'):
        return value.item()
    return value


def calculate_cumulative_time(times):
    """将每轮时间转换为累计时间"""
    cumulative_times = []
    total_time = 0.0
    for single_round_time in times:
        total_time += convert_to_python(single_round_time)
        cumulative_times.append(round(total_time, 4))  # 保留4位小数，增强可读性
    return cumulative_times


def get_time_to_target_accuracy(times, accuracies, target_accuracy):
    """
    获取达到目标准确率时的累计时间
    返回：(是否达到目标, 达到目标的累计时间, 最终准确率)
    """
    # 先计算累计时间
    cumulative_times = calculate_cumulative_time(times)
    # 转换准确率为标量
    acc_scalars = [convert_to_python(acc) for acc in accuracies]

    # 遍历数据，找到首次达到目标的时间
    for idx, (cum_time, acc) in enumerate(zip(cumulative_times, acc_scalars)):
        if acc >= target_accuracy:
            return True, cum_time, acc, idx  # idx为达到目标时的轮次

    # 若未达到目标，返回最终状态
    final_time = cumulative_times[-1] if cumulative_times else 0.0
    final_acc = acc_scalars[-1] if acc_scalars else 0.0
    return False, final_time, final_acc, len(cumulative_times) - 1


def load_single_pkl_file(file_path):
    """加载单个pkl文件，包含异常处理"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ 警告：读取文件 {file_path} 失败 - {e}")
        return None


# ===================== 核心逻辑：加载数据并提取时间 =====================
def print_time_to_target():
    """主函数：加载所有模型数据，计算并打印达到目标准确率的累计时间"""
    # 1. 加载所有数据
    print("📌 开始加载各模型数据...")
    data_dict = {
        model_name: load_single_pkl_file(file_path)
        for model_name, file_path in DATA_PATHS.items()
    }

    # 2. 校验数据加载有效性
    invalid_models = [name for name, data in data_dict.items() if data is None]
    if invalid_models:
        print(f"❌ 以下模型数据加载失败，将跳过：{', '.join(invalid_models)}")

    # 3. 确定有效AI数量（取所有有效模型的最小AI数）
    valid_data = {name: data for name, data in data_dict.items() if data is not None}
    if not valid_data:
        print("❌ 无有效模型数据，程序退出！")
        return

    num_ais = min(len(data['ais']) for data in valid_data.values())
    if num_ais == 0:
        print("❌ 未找到有效AI数据，程序退出！")
        return
    print(f"✅ 检测到有效AI数量：{num_ais} 个\n")

    # 4. 遍历每个模型和AI，计算并打印时间
    print("=" * 80)
    print(f"🎯 目标准确率：{TARGET_ACCURACY}%")
    print("=" * 80)
    print(
        f"{'模型名称':<20} {'AI索引':<10} {'是否达标':<10} {'达标累计时间(s)':<20} {'最终准确率(%)':<15} {'达标轮次':<10}")
    print("-" * 80)

    for model_name, data in valid_data.items():
        for ai_idx in range(num_ais):
            # 获取当前AI的轮次时间和准确率
            try:
                round_times = data['global_round_times'][ai_idx]
                round_accs = data['global_round_accuracies'][ai_idx]
            except KeyError as e:
                print(f"{model_name:<20} {ai_idx:<10} {'异常':<10} {'-':<20} {'-':<15} {'-':<10} （缺失键：{e}）")
                continue

            # 计算达到目标的时间
            is_reached, target_time, final_acc, target_round = get_time_to_target_accuracy(
                round_times, round_accs, TARGET_ACCURACY
            )

            # 格式化输出
            reach_status = "是" if is_reached else "否"
            time_str = f"{target_time:.4f}" if is_reached else "-"
            acc_str = f"{final_acc:.4f}"
            round_str = f"{target_round}" if is_reached else "-"

            print(f"{model_name:<20} {ai_idx:<10} {reach_status:<10} {time_str:<20} {acc_str:<15} {round_str:<10}")

    print("=" * 80)
    print(f"📝 结果打印完成！当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


# ===================== 执行程序 =====================
if __name__ == "__main__":
    print_time_to_target()