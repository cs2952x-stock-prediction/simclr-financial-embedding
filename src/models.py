import torch
import torch.nn as nn


class SimpleLstmEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim, hidden_dim, output_dim):
        super(SimpleLstmEncoder, self).__init__()
        self.embedding = nn.Linear(n_features, embedding_dim)
        # NOTE: to set output dimension of the lstm, use the 'pro_size' parameter
        # otherwise, it is identical to hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # (batch_sz, seq_len, embedding_dim)
        x, _ = self.lstm(x)  # (batch_sz, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch_sz, hidden_dim, seq_len) for pooling
        x = self.pooling(x).squeeze(-1)  # (batch_sz, hidden_dim)
        output = self.fc(x)  # (batch_sz, output_dim)
        return output
