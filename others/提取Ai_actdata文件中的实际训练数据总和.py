# @Date       2025/7/26 下午9:48
# @Author     2024级电子信息计算机方向 艾春慧
# @University MUC
import pickle

# 读取 Ai_actdata.pkl 文件
with open('Ai_actdata.pkl', 'rb') as f:
    data = pickle.load(f)

# 输出每次实验中所有客户端的训练数据总和
for i, (Ai, training_sizes) in enumerate(data):
    total_training = sum(training_sizes)
    print(f"实验 {i+1}: Ai = {Ai}, 实际总训练数据量 = {total_training:.2f} 兆")
