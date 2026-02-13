import torch
import torch.nn as nn
import torch.nn.functional as F


class NextTokenLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        o, _ = self.lstm(e)
        h = o[:, -1, :]
        return self.fc(h)


def loss_fn(logits, y):
    return F.cross_entropy(logits, y)
