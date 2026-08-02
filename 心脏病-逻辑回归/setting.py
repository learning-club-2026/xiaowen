#超参数设置
import os

# 数据文件路径
DATA_PATH = "heart.csv"

# 随机种子（保证可复现）
RANDOM_SEED = 421

# 训练/测试划分比例
TEST_SIZE = 0.2

# 逻辑回归超参数
LR_C = 1.0                 # 正则化强度（倒数），越小正则化越强）
LR_LEARNING_RATE = 0.01  # 学习率
LR_MAX_ITER = 1000          # 最大迭代次数
LR_PENALTY = 'l2'           # 正则化类型

# 输出图片文件名
ROC_PLOT = "roc_curve.png"