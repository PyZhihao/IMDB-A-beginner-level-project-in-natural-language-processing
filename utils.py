import os
import random

from matplotlib import pyplot as plt

import torch
from torch.nn.utils import clip_grad_norm_


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, length, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        # lengths 可以留在 CPU
        optimizer.zero_grad()  # 梯度清零
        logits = model(x, length)  # 前向传播
        loss = loss_fn(logits, y)  # loss
        loss.backward()  # 反向传播
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()  # 参数更新
        batch_size = y.size(0)
        total_loss += (loss.item() * batch_size)  # 统计 loss
        probs = torch.sigmoid(logits)  # logits -> probability
        preds = (probs > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

    avg_loss = (total_loss / total_samples)
    accuracy = (total_correct / total_samples)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, length, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, length)
        loss = loss_fn(logits, y)
        batch_size = y.size(0)
        total_loss += (loss.item() * batch_size)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

    avg_loss = (total_loss / total_samples)
    accuracy = (total_correct / total_samples)
    return avg_loss, accuracy


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(
        torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    )


def plot_loss(train_losses, val_losses, layers, save_dir, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"loss_curve_{model_name}_{layers}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
