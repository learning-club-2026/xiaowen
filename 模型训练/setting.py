# 超参数
import os

# 数据路径
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

# 模型超参数
LEARNING_RATE = 0.01      # 可调整
EPOCHS = 1000             # 迭代轮数
BATCH_SIZE = 256          # 小批量大小（若用全量可设为 len(X_train)）

# 随机种子
RANDOM_SEED = 42

# 输出目录
FIGURE_SAVE_DIR = "figures"
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)