import torch
import torch.nn as nn
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import random
import matplotlib.pyplot as plt

from setting import Config
from data import get_dataloaders
from model import BertRNNClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(loader, model, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch_idx, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 20 == 0:
            print(f"  Epoch {epoch} - Batch {batch_idx+1}/{len(loader)}")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


def evaluate(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), acc, f1


def plot_curves(history, save_path='training_curves.png'):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   's-', label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'o-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   's-', label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # F1
    axes[2].plot(epochs, history['train_f1'], 'o-', label='Train F1')
    axes[2].plot(epochs, history['val_f1'],   's-', label='Val F1')
    axes[2].set_title('F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 图表已保存: {save_path}")


def main():
    config = Config()
    set_seed(config.seed)

    print(f"Using device: {config.device}")

    train_loader, test_loader = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model = BertRNNClassifier(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'train_f1':   [], 'val_f1':   [],
    }

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        print(f"\n===== Epoch {epoch}/{config.epochs} =====")
        train_loss, train_acc, train_f1 = train_epoch(
            train_loader, model, optimizer, criterion, config.device, epoch
        )
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = evaluate(test_loader, model, criterion, config.device)
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("  -> Best model saved.")

    print(f"\n✅ Training complete. Best test accuracy: {best_acc:.4f}")

    plot_curves(history, save_path='training_curves.png')


if __name__ == "__main__":
    main()
