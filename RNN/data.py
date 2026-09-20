import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


def map_label(star):
    """CSV 里是 1~5 星，映射为三分类：负(0) / 中(1) / 正(2)"""
    if star <= 2:
        return 0
    elif star == 3:
        return 1
    else:
        return 2


class YelpDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx])
        return item


def get_dataloaders(config):
    # CSV 文件和 data.py 在同一目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, 'train.csv')
    test_path  = os.path.join(current_dir, 'test.csv')

    # 读取 CSV（无表头，两列：label, text）
    train_df = pd.read_csv(train_path, header=None, names=['label', 'text'])
    test_df  = pd.read_csv(test_path,  header=None, names=['label', 'text'])

    # 标签映射：1~5 星 -> 0/1/2
    train_df['label'] = train_df['label'].apply(map_label)
    test_df['label']  = test_df['label'].apply(map_label)

    # 截取子集（先跑通，确认没问题后可删掉这两行用全量）
    train_df = train_df.head(2000)
    test_df  = test_df.head(500)

    print(f"训练样本数: {len(train_df)}, 测试样本数: {len(test_df)}")
    print(f"标签分布(训练集):\n{train_df['label'].value_counts().sort_index()}")

    # 分词
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

    train_encodings = tokenizer(
        train_df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_len
    )
    test_encodings = tokenizer(
        test_df['text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_len
    )

    train_dataset = YelpDataset(train_encodings, train_df['label'].tolist())
    test_dataset  = YelpDataset(test_encodings,  test_df['label'].tolist())

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False)

    return train_loader, test_loader
