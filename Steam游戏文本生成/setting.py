"""Steam 结构化数据到游戏简介的训练配置。"""
import os
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 使用已下载到本项目目录的 CSV，路径不受启动命令所在目录影响。
DATA_PATH = os.path.join(CURRENT_DIR, "steam_games.csv")
DATA_ENCODING = "utf-8-sig"
TARGET_COLUMN = "short_description"
OUTPUT_DIR = os.path.join(CURRENT_DIR, "训练过程")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model")
REPORT_PATH = os.path.join(OUTPUT_DIR, "train_report.txt")
CURVE_SAVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")

# 使用轻量级 encoder-decoder 模型做条件文本生成；首次运行需要下载模型权重。
MODEL_NAME = "google/flan-t5-small"
MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 128

TEST_SIZE = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_SAMPLES = 5000  # None 表示使用所有有效样本
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
