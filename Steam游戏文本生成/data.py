"""数据读取、字段分类与 T5 输入编码。"""
import os
import re

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from setting import (
    BATCH_SIZE,
    DATA_PATH,
    DATA_ENCODING,
    MAX_SAMPLES,
    MAX_SOURCE_LEN,
    MAX_TARGET_LEN,
    MODEL_NAME,
    NUM_WORKERS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)


# 按下载的 steam_games.csv 实际表头分类。
FIELD_GROUPS = {
    "游戏名称": ("name",),
    "发行信息": ("release_date", "developers", "publishers"),
    "类型与标签": ("genres", "categories", "tags"),
    "玩法与平台": ("windows", "mac", "linux"),
    "价格与内容": ("price", "price_status", "dlc_count"),
    "玩家反馈": ("metacritic_score", "positive", "negative", "recommendations"),
}


def _clean_value(value):
    """将缺失、列表和空白值统一成可读文本。"""
    if pd.isna(value):
        return ""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = re.sub(r"\s+", " ", value).strip()
    if value.lower() in {"nan", "none", "null", "n/a", "-"}:
        return ""
    return value


def classify_columns(columns):
    """基于现有列名返回分类字段映射，未识别的列不会被误当成输入。"""
    lower_to_original = {str(column).strip().lower(): column for column in columns}
    groups = {}
    used = set()
    for group_name, aliases in FIELD_GROUPS.items():
        selected = []
        for alias in aliases:
            actual = lower_to_original.get(alias.lower())
            if actual is not None and actual not in used:
                selected.append(actual)
                used.add(actual)
        groups[group_name] = selected
    return groups


def _find_target(columns):
    lower_to_original = {str(column).strip().lower(): column for column in columns}
    if TARGET_COLUMN.lower() in lower_to_original:
        return lower_to_original[TARGET_COLUMN.lower()]
    available = ", ".join(map(str, columns))
    raise ValueError(
        f"找不到配置的目标列 {TARGET_COLUMN}。CSV 实际列名：{available}"
    )


def _serialize_row(row, groups):
    sections = []
    for group_name, columns in groups.items():
        values = []
        for column in columns:
            value = _clean_value(row[column])
            if value:
                values.append(f"{column}: {value}")
        if values:
            sections.append(f"{group_name} — " + "; ".join(values))
    return "根据以下 Steam 游戏结构化信息生成简洁、准确的游戏简介：\n" + "\n".join(sections)


class SteamTextDataset(Dataset):
    def __init__(self, sources, targets, tokenizer):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.sources[idx], max_length=MAX_SOURCE_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        target = self.tokenizer(
            text_target=self.targets[idx], max_length=MAX_TARGET_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }


def load_and_preprocess():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"未找到数据文件：{DATA_PATH}\n"
            "请从 Kaggle 下载 steam_games.csv，并放到 Steam游戏文本生成 目录。"
        )

    print(f"数据来源: {DATA_PATH}")
    columns = pd.read_csv(DATA_PATH, encoding=DATA_ENCODING, nrows=0).columns
    target_column = _find_target(columns)
    groups = classify_columns(columns)
    input_columns = [column for columns in groups.values() for column in columns]
    if not input_columns:
        raise ValueError(f"未识别到可用于生成的结构化字段，CSV 列名：{list(columns)}")

    # 原文件约 1 GB，仅加载输入字段和目标简介，减少内存占用。
    df = pd.read_csv(
        DATA_PATH, encoding=DATA_ENCODING, low_memory=False,
        usecols=input_columns + [target_column],
    )
    if df.empty:
        raise ValueError(f"数据文件没有样本：{DATA_PATH}")

    # 简介不能作为输入特征，避免目标泄漏；同时丢弃空简介和空输入。
    df["_target"] = df[target_column].map(_clean_value)
    df = df[df["_target"].str.len() > 0].copy()
    df["_source"] = df.apply(lambda row: _serialize_row(row, groups), axis=1)
    df = df[df["_source"].str.contains(r":\s*\S", regex=True)].drop_duplicates("_target")
    if MAX_SAMPLES is not None:
        df = df.head(MAX_SAMPLES)
    if len(df) < 2:
        raise ValueError("清理后有效样本不足 2 条，无法划分训练集和测试集。")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SteamTextDataset(train_df["_source"].tolist(), train_df["_target"].tolist(), tokenizer)
    test_dataset = SteamTextDataset(test_df["_source"].tolist(), test_df["_target"].tolist(), tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    print(f"目标文本列: {target_column}")
    print("结构字段分类:")
    for group_name, columns in groups.items():
        if columns:
            print(f"  {group_name}: {', '.join(map(str, columns))}")
    print(f"训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    return train_loader, test_loader, tokenizer, groups
