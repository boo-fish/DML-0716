import os
import pickle
import torch

# 设置目标文件夹路径
# directory = 'saving/0918/本文方法'
directory = 'saving/0918/基准方法'

# 获取所有 .pkl 文件
pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

# 记录结果
results = []

print(f"{'文件名':<60} {'round_80':>10} {'round_85':>10} {'round_90':>10}")
print("=" * 95)

for filename in sorted(pkl_files):
    filepath = os.path.join(directory, filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # 读取 global_round_accuracies 的第一个元素
        round_accuracies = data['global_round_accuracies'][0]

        # 转换为 float（如果是 tensor）
        round_accuracies_float = [acc.item() if isinstance(acc, torch.Tensor) else float(acc)
                                  for acc in round_accuracies]

        # 查找首次达到 80 和 85 的轮数
        round_80 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 80), None)
        round_85 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 85), None)
        round_90 = next((i + 1 for i, acc in enumerate(round_accuracies_float) if acc >= 90), None)


        # 打印结果
        print(f"{filename:<60} {str(round_80):>10} {str(round_85):>10} {str(round_90):>10}")

    except Exception as e:
        print(f"{filename:<60} {'Error':>10} {'Error':>10}")
        print(f"  [ERROR] 无法解析该文件: {e}")
