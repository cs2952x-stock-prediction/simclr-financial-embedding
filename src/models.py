import torch.nn as nn


class SimpleLstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, proj_size=0):
        super(SimpleLstmEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            batch_first=True,
        )
        proj_size = proj_size if proj_size > 0 else hidden_size
        self.fc1 = nn.Linear(proj_size, proj_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(proj_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        z = self.fc1(h_n[-1])
        z = self.relu(z)
        z = self.fc2(z)
        return z
