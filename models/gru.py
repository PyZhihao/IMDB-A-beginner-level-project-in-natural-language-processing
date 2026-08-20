from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, gru_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=pad_idx)
        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=gru_layers,
                          bias=True,
                          batch_first=True,
                          dropout=dropout if gru_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, length):
        # x: [batch_size, seq_len]
        # 1. Embedding
        x = self.embedding(x)  # -> [batch_size, seq_len, embedding_dim]
        # 2. 打包序列, 告诉 GRU 每个 token 的真实长度， 后面的 <pad> 不需要参与计算
        packed_x = pack_padded_sequence(x,
                                        length.cpu(),
                                        batch_first=True,
                                        enforce_sorted=False)
        # 3. GRU
        # h_n: [num_layers, batch_size, hidden_dim]
        packed_output, h_n = self.gru(packed_x)
        # 4. 最后一层隐藏状态
        h = h_n[-1]  # -> [batch_size, hidden_dim]
        # 5. Dropout
        h = self.dropout(h)
        # 6. 分类
        logits = self.linear(h)  # -> [batch_size, 1]
        return logits.squeeze(1)
