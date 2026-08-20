import os
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader

from data import IMDBDataset, load_imdb_data
from models import GRUClassifier
from utils import evaluate, load_checkpoint, plot_loss, save_checkpoint, set_seed, train_epoch
from vocab import Vocab


if __name__ == "__main__":
    seed = 2026
    set_seed(seed)

    # IMDB数据集下载地址: https://ai.stanford.edu/~amaas/data/sentiment/
    data_root = "./aclImdb"

    os.makedirs("best_model", exist_ok=True)
    os.makedirs("loss_curve", exist_ok=True)

    # 参数配置
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10
    max_len = 256  # 每条评论最多保留 256 个 token, 超过则截断/不足则<pad>补齐
    max_vocab_size = 20000  # 最大词表大小, 只保留最高频 token, 其余会被映射为<unk>
    embedding_dim = 128  # 每个 token 转换为 128 维词向量
    hidden_dim = 128  # GRU 隐藏状态维度
    gru_layers = 1  # GRU 层数
    dropout = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_texts, train_labels = load_imdb_data(data_root, split="train")
    print(f"len(train_texts): {len(train_texts)}")

    # full_trian -> train(80%) / validation(20%)
    # validation 不能参与 vocab 的构建, 只能使用 train 构建 vocab
    # validation 用于: 模型选择/超参数选择/最优模型选择
    num_samples = len(train_texts)
    train_size = int(len(train_texts) * 0.8)
    val_size = len(train_texts) - train_size
    indices = torch.randperm(len(train_texts),
                             generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_texts_split = [train_texts[i] for i in train_indices]
    train_labels_split = [train_labels[i] for i in train_indices]
    val_texts_split = [train_texts[i] for i in val_indices]
    val_labels_split = [train_labels[i] for i in val_indices]

    vocab = Vocab(train_texts_split, min_freq=2, max_size=max_vocab_size)  # 只使用真正的训练集构建 Vocab
    print(f"vocab size: {len(vocab)}")
    with open("best_model/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    train_dataset = IMDBDataset(train_texts_split, train_labels_split, vocab, max_len=max_len)
    val_dataset = IMDBDataset(val_texts_split, val_labels_split, vocab, max_len=max_len)
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")

    # test
    # 不能重新用 test 构建一个 vocab
    test_texts, test_labels = load_imdb_data(data_root, split="test")
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len=max_len)
    print(f"test samples: {len(test_dataset)}")

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建 GRU 模型
    model = GRUClassifier(vocab_size=len(vocab),
                          embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim,
                          gru_layers=gru_layers,
                          dropout=dropout,
                          pad_idx=vocab.pad)
    model = model.to(device)
    print(f"model: {model}")

    # 当前任务: 二分类 positive/negative
    # 定义 损失函数 BCEWithLogitsLoss = Sigmoid + Binary Cross Entropy
    loss_fn = nn.BCEWithLogitsLoss()

    # 定义 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练
    min_val_loss = float("inf")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"| "
            f"Train Loss: {train_loss:.4f} "
            f"| "
            f"Train Acc: {train_acc:.4f} "
            f"| "
            f"Val Loss: {val_loss:.4f} "
            f"| "
            f"Val Acc: {val_acc:.4f}"
        )
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(model, f"best_model/best_model_gru_{gru_layers}.pt")

    print(f"Min val loss: {min_val_loss:.4f}")
    print(f"Best val Acc: {best_val_acc:.4f}")

    # 画 loss 曲线
    plot_loss(train_losses, val_losses, gru_layers, "loss_curve", "gru")

    # 加载模型参数
    load_checkpoint(model, f"best_model/best_model_gru_{gru_layers}.pt", device)

    # 测试
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f} ")
    print(f"Test Acc: {test_acc:.4f}")
