import os
import pickle
import re
import torch
from collections import defaultdict

# 设置根文件夹路径
root_directory = r'D:\project\DML-0716\results-mnist-Final-tao'

# 定义需要处理的子文件夹
target_folders = ['DML', 'benchmark', 'RAMFL']

# 每个分组读取的文件数量
K = 5

# 用于存储分组结果的字典
# 结构: {文件夹名: { (isIID值, 数字): [文件列表] }}
grouped_files = defaultdict(lambda: defaultdict(list))

# 编译正则表达式来提取isIID和数字信息
# 匹配 isIID_True/False 和前面的数字（5/10）
pattern = re.compile(r'(\d+)_epoch.*?isIID_(True|False)')

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
            number = match.group(1)  # 提取5或10
            is_iid = match.group(2)  # 提取True或False

            # 只处理5和10的情况
            if number in ['5', '10']:
                key = (is_iid, number)
                grouped_files[folder][key].append(filename)

# 打印分组统计信息
print("=" * 100)
print("分组统计信息：")
print("=" * 100)
for folder, groups in grouped_files.items():
    print(f"\n文件夹: {folder}")
    for (is_iid, number), files in groups.items():
        count = len(files)
        print(f"  isIID={is_iid}, 数字={number}: 共{count}个文件 (取前{min(K, count)}个)")

# 第二步：处理每个分组的前K个文件，统计精度轮数
print("\n" + "=" * 100)
print("精度轮数统计结果：")
print("=" * 100)
# 打印表头
header = f"{'文件夹':<10} {'分组(isIID,数字)':<20} {'文件名':<60} {'round_80':>10} {'round_85':>10} {'round_90':>10}"
print(header)
print("=" * len(header))

# 遍历每个分组处理文件
for folder, groups in grouped_files.items():
    for (is_iid, number), files in groups.items():
        group_label = f"({is_iid}, {number})"

        # 取前K个文件
        files_to_process = files[:K]

        for filename in files_to_process:
            filepath = os.path.join(root_directory, folder, filename)
            try:
                # 读取pkl文件
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                # 读取global_round_accuracies的第一个元素
                round_accuracies = data['global_round_accuracies'][0]

                # 转换为float（处理tensor）
                round_accuracies_float = []
                for acc in round_accuracies:
                    if isinstance(acc, torch.Tensor):
                        round_accuracies_float.append(acc.item())
                    else:
                        round_accuracies_float.append(float(acc))

                # 查找首次达到80、85、90精度的轮数（轮数从1开始）
                round_80 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 80), None)
                round_85 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 85), None)
                round_90 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 90), None)

                # 格式化输出
                print(f"{folder:<10} {group_label:<20} {filename:<60} "
                      f"{str(round_80):>10} {str(round_85):>10} {str(round_90):>10}")

            except Exception as e:
                print(f"{folder:<10} {group_label:<20} {filename:<60} "
                      f"{'Error':>10} {'Error':>10} {'Error':>10}")
                print(f"                      {' ':<60}  [ERROR] 无法解析该文件: {e}")

# 第三步：生成汇总统计
print("\n" + "=" * 100)
print("汇总统计（每个分组的平均轮数）：")
print("=" * 100)
summary = defaultdict(lambda: defaultdict(list))

# 重新处理文件，收集每个分组的轮数数据
for folder, groups in grouped_files.items():
    for (is_iid, number), files in groups.items():
        group_key = (folder, is_iid, number)
        files_to_process = files[:K]

        for filename in files_to_process:
            filepath = os.path.join(root_directory, folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)

                round_accuracies = data['global_round_accuracies'][0]
                round_accuracies_float = [acc.item() if isinstance(acc, torch.Tensor) else float(acc)
                                          for acc in round_accuracies]

                round_80 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 80), None)
                round_85 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 85), None)
                round_90 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 90), None)

                # 只收集有效的数值
                if round_80 is not None:
                    summary[group_key]['80'].append(round_80)
                if round_85 is not None:
                    summary[group_key]['85'].append(round_85)
                if round_90 is not None:
                    summary[group_key]['90'].append(round_90)

            except Exception:
                continue

# 打印汇总结果
print(
    f"{'文件夹':<10} {'isIID':<10} {'数字':<10} {'样本数':<8} {'平均round_80':>12} {'平均round_85':>12} {'平均round_90':>12}")
print("=" * 80)
for (folder, is_iid, number), values in summary.items():
    count = len(values.get('80', []))
    avg_80 = sum(values['80']) / count if count and values['80'] else None
    avg_85 = sum(values['85']) / len(values['85']) if values['85'] else None
    avg_90 = sum(values['90']) / len(values['90']) if values['90'] else None

    print(f"{folder:<10} {is_iid:<10} {number:<10} {count:<8} "
          f"{f'{avg_80:.2f}':>12} {f'{avg_85:.2f}':>12} {f'{avg_90:.2f}':>12}")