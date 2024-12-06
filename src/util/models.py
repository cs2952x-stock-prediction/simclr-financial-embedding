import torch.nn as nn



# class LstmEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, proj_size=0):
#         if hidden_size == proj_size:
#             proj_size = 0
#         super(LstmEncoder, self).__init__()
#         self.lstm = nn.LSTM(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             proj_size=proj_size if proj_size else None,
#             batch_first=True,
#         )

#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         return h_n[-1]

class LstmEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, proj_size=0):
        if proj_size > hidden_size:
            raise ValueError("proj_size must be smaller than hidden_size")
        proj_size = proj_size if proj_size > 0 else 0
        super(LstmEncoder, self).__init__()
        if proj_size == hidden_size:
            proj_size = 0
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
    def __init__(self, in_channels, out_channels, kernel_size, num_layers = 1):
        super(CnnEncoder, self).__init__()
        self.num_layers = num_layers
        self.conv_1d_layers = nn.ModuleList()
        for layer_i in range(self.num_layers):
            self.conv_1d_layers.append(
                nn.Conv1d(
                    in_channels = in_channels if layer_i == 0 else out_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    
                )
            )
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.conv_1d_layers:
            x = self.ReLU(layer(x))
        x = x.view(x.size(0), -1)
        return x