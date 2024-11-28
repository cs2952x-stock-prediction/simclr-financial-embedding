import torch.nn as nn


class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, proj_size=0):
        if hidden_size == proj_size:
            proj_size = 0
        super(LstmEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            proj_size=proj_size,
            batch_first=True,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class DenseLayers(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None):
        super(DenseLayers, self).__init__()

        hidden_layers = [] if hidden_layers is None else hidden_layers
        layers = []

        activation_funcs = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "none": nn.Identity(),
        }

        # Build layers with specified activations
        prev_size = input_size
        for size, activation in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            activation = activation_funcs.get(activation, None)
            prev_size = size
            if activation is None:
                raise ValueError(f"Unknown activation function: '{activation}'")
            layers.append(activation)

        # Add output layer without activation
        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CnnEncoder(nn.Module):
    def __init__(self, input_size, out_channels, kernel_size):
        super(CnnEncoder, self).__init__()
        self.conv_1d = nn.Conv1d(
            in_channels = 30,
            out_channels = out_channels,
            kernel_size = kernel_size,
        )

    def forward(self, x):
        x = self.conv_1d(x)
        x = x.squeeze(-1)
        return x