"""训练 Steam 游戏结构化信息到简介的文本生成模型。"""
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from data import load_and_preprocess
from model import SteamTextGenerator
from setting import (
    CURVE_SAVE_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    REPORT_PATH,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(loader, model, optimizer=None, epoch=1):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    phase = "训练" if training else "验证"
    # 每完成一个 batch 更新进度；显示批次数、耗时、剩余时间和 Loss。
    with context, tqdm(
        total=len(loader), desc=f"Epoch {epoch}/{EPOCHS} {phase}",
        unit="batch", dynamic_ncols=True, mininterval=0.5,
    ) as progress:
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            progress.set_postfix(
                loss=f"{batch_loss:.4f}", avg_loss=f"{total_loss / batch_idx:.4f}",
                refresh=False,
            )
            progress.update(1)
    return total_loss / max(len(loader), 1)


def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "s-", label="Validation Loss")
    plt.title("Steam Structured-to-Text Generation")
    plt.xlabel("Epoch")
    plt.ylabel("Token Cross Entropy Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=150)
    plt.close()
    print(f"训练曲线已保存: {CURVE_SAVE_PATH}")


def train():
    set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}", flush=True)
    print("正在读取和预处理 CSV…", flush=True)
    train_loader, test_loader, _, _ = load_and_preprocess()
    print(f"每轮训练 {len(train_loader)} 批，验证 {len(test_loader)} 批，共 {EPOCHS} 轮。", flush=True)
    print(f"正在加载模型 {MODEL_NAME}…", flush=True)
    model = SteamTextGenerator(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    with open(REPORT_PATH, "w", encoding="utf-8") as report:
        report.write(f"模型: {MODEL_NAME}\n设备: {DEVICE}\n随机种子: {RANDOM_SEED}\n")
        report.write(f"{'Epoch':>6} | {'Train Loss':>12} | {'Validation Loss':>16}\n")
        report.write("-" * 42 + "\n")

    training_start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_loss = run_epoch(train_loader, model, optimizer, epoch=epoch)
        val_loss = run_epoch(test_loader, model, epoch=epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        line = f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>16.4f}"
        print(line)
        with open(REPORT_PATH, "a", encoding="utf-8") as report:
            report.write(line + "\n")
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(MODEL_SAVE_PATH)
            print(f"  -> 最佳模型已保存: {MODEL_SAVE_PATH}")
        elapsed = time.perf_counter() - training_start
        remaining = elapsed / epoch * (EPOCHS - epoch)
        timing = (
            f"本轮耗时 {(time.perf_counter() - epoch_start) / 60:.1f} 分钟，"
            f"累计 {elapsed / 60:.1f} 分钟，"
            f"剩余轮次预计 {remaining / 60:.1f} 分钟（按已完成轮次估算）"
        )
        print(timing, flush=True)
        with open(REPORT_PATH, "a", encoding="utf-8") as report:
            report.write(timing + "\n")

    plot_curves(history)
    print(f"训练完成，最佳验证 Loss: {best_loss:.4f}")
    print(f"训练日志: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练已停止；已完成轮次保存的最佳模型保留，当前轮次未保存。", flush=True)
