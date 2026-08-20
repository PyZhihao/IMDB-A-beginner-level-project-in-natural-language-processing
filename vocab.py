from collections import Counter

from data import tokenize


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
