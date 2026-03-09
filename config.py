# @Date       2025/12/9 下午9:28
# @Author     2024级电子信息计算机方向 艾春慧
# @University MUC
# config.py
class Config:
    """实验配置参数"""

    # 网络参数
    NUM_WORKERS = 50
    NUM_SERVERS = 10
    LINK_CAPACITY_RANGE = (50, 150)  # Mbps
    WORKER_CAPACITY_RANGE = (5, 45)  # Mbps
    SERVER_CAPACITY_RANGE = (30, 60)  # Mbps

    # 数据到达参数
    DATA_ARRIVAL_RANGE = (10, 30)  # 基础范围
    ARRIVAL_TEST_RANGE = (0, 40)  # 测试范围
    NUM_ARRIVAL_SCENARIOS = 21  # 测试场景数量

    # 算法参数
    ALPHA = 0.4  # 数据缩放系数
    SOLVER_MAX_ITER = 5000  # 求解器最大迭代次数
    SOLVER_EPS = 1e-4  # 求解器精度

    # 随机种子
    RANDOM_SEED = 42

    @classmethod
    def display_config(cls):
        """显示配置信息"""
        print("实验配置:")
        print(f"  工人数量: {cls.NUM_WORKERS}")
        print(f"  参数服务器数量: {cls.NUM_SERVERS}")
        print(f"  数据到达测试范围: {cls.ARRIVAL_TEST_RANGE[0]}-{cls.ARRIVAL_TEST_RANGE[1]}")
        print(f"  测试场景数量: {cls.NUM_ARRIVAL_SCENARIOS}")
        print(f"  α值: {cls.ALPHA}")