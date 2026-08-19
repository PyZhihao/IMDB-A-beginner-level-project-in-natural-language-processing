import os
import re
import random
from collections import Counter
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_imdb_data(root, split="train"):
    texts = []
    labels = []

    # 消极评论 negative
    neg_dir = os.path.join(root, split, "neg")
    for filename in sorted(os.listdir(neg_dir)):
        path = os.path.join(neg_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        texts.append(text)
        labels.append(0)

    # 积极评论 positive
    pos_dir = os.path.join(root, split, "pos")
    for filename in sorted(os.listdir(pos_dir)):
        path = os.path.join(pos_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        texts.append(text)
        labels.append(1)

    return texts, labels


def tokenize(text):
    """
    简单英文分词:
    1、转小写
    2、提取英文单词、数字和标点符号
    3、支持 i'm  don't 等
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)*|\d+|[^\w\s]", text)
    return tokens


class Vocab:
    def __init__(self, texts, min_freq=2, max_size=20000):
        counter = Counter()
        for text in texts:  # 统计词频
            tokens = tokenize(text)
            counter.update(tokens)
        # index -> token, 特殊 token: <pad> = 0, <unk> = 1
        self.idx_to_token = ["<pad>", "<unk>"]
        # 按词频从高到低排序
        token_freqs = counter.most_common(max_size)
        for token, freq in token_freqs:
            if freq < min_freq:
                break
            if len(self.idx_to_token) >= max_size:
                break
            self.idx_to_token.append(token)
        # token -> index
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        """
        token -> index
        如果 token 不在词表中， 返回 <unk> 的索引
        """
        return self.token_to_idx.get(token, self.unk)

    def to_tokens(self, indices):
        """
        index -> token
        """
        if isinstance(indices, int):
            return self.idx_to_token[indices]

        return [self.idx_to_token[index] for index in indices]

    @property
    def pad(self):
        return 0

    @property
    def unk(self):
        return 1


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = tokenize(text)  # 文本 -> tokens
        token_ids = [self.vocab[token] for token in tokens]  # tokens -> token ids
        token_ids = token_ids[:self.max_len]  # 截断

        # 记录 tokens 真实长度
        # 注意: 必须在 padding 之前记录
        length = len(token_ids)
        if length == 0:
            token_ids = [self.vocab.unk]
            length = 1

        if len(token_ids) < self.max_len:  # padding
            padding_length = (self.max_len - len(token_ids))
            token_ids += ([self.vocab.pad] * padding_length)

        # 此时无论原始评论多长:
        # len(token_ids) 都必须等于 max_len
        assert len(token_ids) == self.max_len

        # 转换为 tensor
        x = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)
        length = torch.tensor(length, dtype=torch.long)

        return x, length, y


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bias=True,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, length):
        # x: [batch_size, seq_len]
        # 1. Embedding
        x = self.embedding(x)  # -> [batch_size, seq_len, embedding_dim]
        # 2. 打包序列, 告诉 lstm 每个 token 的真实长度， 后面的 <pad> 不需要参与计算
        packed_x = pack_padded_sequence(x,
                                        length.cpu(),
                                        batch_first=True,
                                        enforce_sorted=False)
        # 3. lstm
        # h_n: [num_layers, batch_size, hidden_dim]
        # c_n: [num_layers, batch_size, hidden_dim]
        packed_output, (h_n, c_n) = self.lstm(packed_x)
        # 4. 最后一层隐藏状态
        h = h_n[-1]  # -> [batch_size, hidden_dim]
        # 5. Dropout
        h = self.dropout(h)
        # 6. 分类
        logits = self.linear(h)  # -> [batch_size, 1]
        return logits.squeeze(1)


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


def plot_loss(train_losses, val_losses, layers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"loss_curve_lstm_{layers}.png")

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
    hidden_dim = 128  # lstm 隐藏状态维度
    lstm_layers = 1  # lstm 层数
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

    # 创建 lstm 模型
    model = LSTMClassifier(vocab_size=len(vocab),
                          embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim,
                          lstm_layers=lstm_layers,
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
            torch.save(model.state_dict(), f"best_model/best_model_lstm_{lstm_layers}.pt")

    print(f"Min val loss: {min_val_loss:.4f}")
    print(f"Best val Acc: {best_val_acc:.4f}")

    # 画 loss 曲线
    plot_loss(train_losses, val_losses, lstm_layers, "loss_curve")

    # 加载模型参数
    model.load_state_dict(
        torch.load(
            f"best_model/best_model_lstm_{lstm_layers}.pt",
            map_location=device,
            weights_only=True
        )
    )

    # 测试
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f} ")
    print(f"Test Acc: {test_acc:.4f}")
