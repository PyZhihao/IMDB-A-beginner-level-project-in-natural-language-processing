import os
import re

import torch
from torch.utils.data import Dataset


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
