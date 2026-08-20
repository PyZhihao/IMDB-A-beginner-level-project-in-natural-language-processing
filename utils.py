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


def predict(model, texts, vocab, tokenizer, device, max_len=256):
    """对文本（单条/多条）进行情感分类预测"""
    model.eval()

    # 单条文本转为列表
    if isinstance(texts, str):
        texts = [texts]

    token_ids_list = []
    lengths = []

    # 文本预处理
    for text in texts:
        token_ids = tokenizer(text)  # text -> tokens
        token_ids = [vocab[token] for token in token_ids]  # tokens -> ids
        token_ids = token_ids[:max_len]  # 截断
        length = len(token_ids)  # 真实长度
        if length == 0:  # 空文本处理
            token_ids = [vocab.unk]
            length = 1
        if length < max_len:  # padding
            token_ids += [vocab.pad] * (max_len - length)

        token_ids_list.append(token_ids)
        lengths.append(length)

    x = torch.tensor(token_ids_list, dtype=torch.long)
    x = x.to(device)
    lengths = torch.tensor(lengths, dtype=torch.long)
    lengths = lengths.to(device)


    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.sigmoid(logits)

    probs = probs.cpu().numpy()

    results = []
    for prob in probs:
        label = ("positive" if prob > 0.5 else "negative")
        results.append({"label": label, "probability": prob})

    return results
