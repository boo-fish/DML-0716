import os
import pickle
import re
import torch
from collections import defaultdict
import numpy as np

# 设置根文件夹路径
root_directory = r'D:\project\DML-0716\results-cifar10-3d-1228'

# 定义需要处理的子文件夹
target_folders = ['DML', 'benchmark', 'RAMFL']

# 每个分组读取的文件数量
K = 5

# 用于存储分组结果的字典
# 结构: {文件夹名: { (isIID值, 数字, alpha值): [文件列表] }}
grouped_files = defaultdict(lambda: defaultdict(list))

# 编译正则表达式来提取isIID、数字和alpha信息
# 匹配 数字（3/5/10）、isIID_True/False、alpha值
pattern = re.compile(r'(\d+)_epoch.*?isIID_(True|False)_alpha_([0-9.]+)')

# 定义需要筛选的alpha值
target_alphas = {'0.3', '0.6', '0.8', '1.0'}
# 定义需要筛选的数字
target_numbers = {'3', '5', '10'}

# 第一步：遍历所有目标文件夹，按规则分组文件
for folder in target_folders:
    folder_path = os.path.join(root_directory, folder)

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在，跳过")
        continue

    # 获取文件夹下所有.pkl文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

    # 对每个文件进行分组
    for filename in sorted(pkl_files):
        # 使用正则表达式提取信息
        match = pattern.search(filename)
        if match:
            number = match.group(1)  # 提取3/5/10
            is_iid = match.group(2)  # 提取True或False
            alpha = match.group(3)  # 提取alpha值

            # 只处理指定的数字和alpha值
            if number in target_numbers and alpha in target_alphas:
                key = (is_iid, number, alpha)
                grouped_files[folder][key].append(filename)

# 打印分组统计信息
print("=" * 120)
print("分组统计信息：")
print("=" * 120)
for folder, groups in grouped_files.items():
    print(f"\n文件夹: {folder}")
    for (is_iid, number, alpha), files in groups.items():
        count = len(files)
        print(f"  isIID={is_iid}, 数字={number}, alpha={alpha}: 共{count}个文件 (取前{min(K, count)}个)")

# 第二步：处理每个分组的前K个文件，统计精度轮数和累计时间
print("\n" + "=" * 120)
print("精度轮数和累计时间统计结果：")
print("=" * 120)
# 打印表头
header = (f"{'文件夹':<10} {'分组(isIID,数字,alpha)':<30} {'文件名':<70} "
          f"{'round_55':>10} {'time_55(s)':>12} "
          f"{'round_60':>10} {'time_60(s)':>12} "
          f"{'round_61':>10} {'time_61(s)':>12}")
print(header)
print("=" * len(header))

# 遍历每个分组处理文件
for folder, groups in grouped_files.items():
    for (is_iid, number, alpha), files in groups.items():
        group_label = f"({is_iid}, {number}, {alpha})"

        # 取前K个文件
        files_to_process = files[:K]

        for filename in files_to_process:
            filepath = os.path.join(root_directory, folder, filename)
            try:
                # 读取pkl文件
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                # 读取global_round_accuracies和global_round_times的第一个元素
                round_accuracies = data['global_round_accuracies'][0]
                round_times = data['global_round_times'][0]

                # 转换为float（处理tensor）
                round_accuracies_float = []
                for acc in round_accuracies:
                    if isinstance(acc, torch.Tensor):
                        round_accuracies_float.append(acc.item())
                    else:
                        round_accuracies_float.append(float(acc))

                # 转换时间为float（处理tensor）
                round_times_float = []
                for t in round_times:
                    if isinstance(t, torch.Tensor):
                        round_times_float.append(t.item())
                    else:
                        round_times_float.append(float(t))

                # 查找首次达到55、60、61精度的轮数（轮数从1开始）
                round_55 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 55), None)
                round_60 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 60), None)
                round_61 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 61), None)

                # 计算达到对应精度的累计时间
                time_55 = None
                if round_55 and round_55 <= len(round_times_float):
                    # 累计时间：前round_55轮的时间总和
                    time_55 = sum(round_times_float[:round_55])

                time_60 = None
                if round_60 and round_60 <= len(round_times_float):
                    time_60 = sum(round_times_float[:round_60])

                time_61 = None
                if round_61 and round_61 <= len(round_times_float):
                    time_61 = sum(round_times_float[:round_61])

                # 格式化输出
                print(f"{folder:<10} {group_label:<30} {filename:<70} "
                      f"{str(round_55):>10} {f'{time_55:.2f}' if time_55 else '-':>12} "
                      f"{str(round_60):>10} {f'{time_60:.2f}' if time_60 else '-':>12} "
                      f"{str(round_61):>10} {f'{time_61:.2f}' if time_61 else '-':>12}")

            except Exception as e:
                print(f"{folder:<10} {group_label:<30} {filename:<70} "
                      f"{'Error':>10} {'Error':>12} {'Error':>10} {'Error':>12} {'Error':>10} {'Error':>12}")
                print(f"                      {' ':<70}  [ERROR] 无法解析该文件: {e}")

# 第三步：生成汇总统计
print("\n" + "=" * 120)
print("汇总统计（每个分组的平均轮数和平均累计时间）：")
print("=" * 120)
summary = defaultdict(lambda: defaultdict(dict))

# 重新处理文件，收集每个分组的轮数和时间数据
for folder, groups in grouped_files.items():
    for (is_iid, number, alpha), files in groups.items():
        group_key = (folder, is_iid, number, alpha)
        files_to_process = files[:K]

        # 初始化存储列表
        summary[group_key]['55_rounds'] = []
        summary[group_key]['55_times'] = []
        summary[group_key]['60_rounds'] = []
        summary[group_key]['60_times'] = []
        summary[group_key]['61_rounds'] = []
        summary[group_key]['61_times'] = []
        summary[group_key]['count'] = 0

        for filename in files_to_process:
            filepath = os.path.join(root_directory, folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                round_accuracies = data['global_round_accuracies'][0]
                round_times = data['global_round_times'][0]

                # 转换为float
                round_accuracies_float = [acc.item() if isinstance(acc, torch.Tensor) else float(acc)
                                          for acc in round_accuracies]
                round_times_float = [t.item() if isinstance(t, torch.Tensor) else float(t)
                                     for t in round_times]

                # 获取达到各精度的轮数和累计时间
                round_55 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 55), None)
                round_60 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 60), None)
                round_61 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 61), None)

                # 计算累计时间
                time_55 = sum(round_times_float[:round_55]) if (
                        round_55 and round_55 <= len(round_times_float)) else None
                time_60 = sum(round_times_float[:round_60]) if (
                        round_60 and round_60 <= len(round_times_float)) else None
                time_61 = sum(round_times_float[:round_61]) if (
                        round_61 and round_61 <= len(round_times_float)) else None

                # 收集有效数据
                if round_55 and time_55:
                    summary[group_key]['55_rounds'].append(round_55)
                    summary[group_key]['55_times'].append(time_55)
                if round_60 and time_60:
                    summary[group_key]['60_rounds'].append(round_60)
                    summary[group_key]['60_times'].append(time_60)
                if round_61 and time_61:
                    summary[group_key]['61_rounds'].append(round_61)
                    summary[group_key]['61_times'].append(time_61)

                summary[group_key]['count'] += 1

            except Exception as e:
                continue

# 打印汇总结果
header_summary = (f"{'文件夹':<10} {'isIID':<10} {'数字':<10} {'alpha':<10} {'样本数':<8} "
                  f"{'平均round_55':>12} {'平均time_55(s)':>15} "
                  f"{'平均round_60':>12} {'平均time_60(s)':>15} "
                  f"{'平均round_61':>12} {'平均time_61(s)':>15}")
print(header_summary)
print("=" * len(header_summary))

for (folder, is_iid, number, alpha), values in summary.items():
    count = values['count']

    # 计算平均值
    avg_55_round = np.mean(values['55_rounds']) if values['55_rounds'] else None
    avg_55_time = np.mean(values['55_times']) if values['55_times'] else None

    avg_60_round = np.mean(values['60_rounds']) if values['60_rounds'] else None
    avg_60_time = np.mean(values['60_times']) if values['60_times'] else None

    avg_61_round = np.mean(values['61_rounds']) if values['61_rounds'] else None
    avg_61_time = np.mean(values['61_times']) if values['61_times'] else None

    # 格式化输出
    print(f"{folder:<10} {is_iid:<10} {number:<10} {alpha:<10} {count:<8} "
          f"{f'{avg_55_round:.2f}':>12} {f'{avg_55_time:.2f}' if avg_55_time else '-':>15} "
          f"{f'{avg_60_round:.2f}':>12} {f'{avg_60_time:.2f}' if avg_60_time else '-':>15} "
          f"{f'{avg_61_round:.2f}':>12} {f'{avg_61_time:.2f}' if avg_61_time else '-':>15}")

# 第四步：可选 - 生成每个AI的详细统计（如果需要）
print("\n" + "=" * 120)
print("每个AI的详细时间统计（可选）：")
print("=" * 120)

# 按需启用：如果需要查看每个AI的详细时间统计，取消下面的注释
"""
for folder in target_folders:
    folder_path = os.path.join(root_directory, folder)
    if not os.path.isdir(folder_path):
        continue

    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    for filename in sorted(pkl_files)[:K]:  # 只处理前K个文件
        filepath = os.path.join(root_directory, folder, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            print(f"\n文件: {folder}/{filename}")
            # 遍历所有AI
            num_ais = len(data.get('global_round_accuracies', []))
            for ai_idx in range(num_ais):
                try:
                    round_times = data['global_round_times'][ai_idx]
                    round_accs = data['global_round_accuracies'][ai_idx]

                    # 转换为float
                    round_times_float = [t.item() if isinstance(t, torch.Tensor) else float(t) for t in round_times]
                    round_accs_float = [a.item() if isinstance(a, torch.Tensor) else float(a) for a in round_accs]

                    # 查找各精度点
                    round_55 = next((i + 1 for i, acc in enumerate(round_accs_float) if acc >= 55), None)
                    round_60 = next((i + 1 for i, acc in enumerate(round_accs_float) if acc >= 60), None)
                    round_61 = next((i + 1 for i, acc in enumerate(round_accs_float) if acc >= 61), None)

                    # 计算累计时间
                    time_55 = sum(round_times_float[:round_55]) if (round_55 and round_55 <= len(round_times_float)) else None
                    time_60 = sum(round_times_float[:round_60]) if (round_60 and round_60 <= len(round_times_float)) else None
                    time_61 = sum(round_times_float[:round_61]) if (round_61 and round_61 <= len(round_times_float)) else None

                    print(f"  AI {ai_idx}: "
                          f"55%精度(轮数:{round_55}, 时间:{time_55:.2f}s) | "
                          f"60%精度(轮数:{round_60}, 时间:{time_60:.2f}s) | "
                          f"61%精度(轮数:{round_61}, 时间:{time_61:.2f}s)")

                except KeyError as e:
                    print(f"  AI {ai_idx}: 缺失键 {e}")
                except Exception as e:
                    print(f"  AI {ai_idx}: 错误 {e}")

        except Exception as e:
            print(f"  无法解析文件: {e}")

except Exception as e:
    pass
"""