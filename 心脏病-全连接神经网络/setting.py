import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- 模型结构参数 ----------
HIDDEN_DIM1 = 64      # 第一个隐藏层的神经元数量
HIDDEN_DIM2 = 32      # 第二个隐藏层的神经元数量
OUTPUT_DIM = 1        # 输出维度（二分类任务）

# ---------- 训练超参数 ----------
BATCH_SIZE = 64       # 每次训练喂入的样本数量
LEARNING_RATE = 0.001 # 学习率（步长）
EPOCHS = 100          # 训练轮数（整个数据集过几遍）
DROPOUT_RATE = 0.2    # Dropout 比率（随机丢弃神经元，防止过拟合）

# ---------- 随机种子 ----------
RANDOM_SEED = 42      # 固定随机种子，确保结果可复现