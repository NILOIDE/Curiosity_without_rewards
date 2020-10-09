import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, x_dim, y_dim, device='cpu'):
        # type: (tuple, tuple, str) -> None
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.device = device

    def get_z_dim(self) -> tuple:
        return self.z_dim

    def get_x_dim(self) -> tuple:
        return self.x_dim

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(dtype=torch.float32)
        assert isinstance(x, torch.FloatTensor)
        if len(tuple(x.shape)) == 1:  # Add batch dimension to 1D tensor
            x = x.unsqueeze(0)
        x = x.to(self.device)
        return x


class Network1D(Network):

    def __init__(self, x_dim, y_dim, hidden_dim=(32,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        super().__init__(x_dim, y_dim, device)
        self.hidden_dim = hidden_dim
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, y_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        s_t = self.apply_tensor_constraints(s_t)
        return self.model(s_t)


class Network1DStochastic(Network):

    def __init__(self, x_dim, y_dim, hidden_dim=(32,), device='cpu'):
        # type: (tuple, tuple, tuple, str) -> None
        super().__init__(x_dim, y_dim, device)
        self.hidden_dim = hidden_dim
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Linear(h_dim_prev, y_dim[0])
        self.sigma_head = nn.Sequential(nn.Linear(h_dim_prev, y_dim[0]), nn.ReLU())

    def forward(self, s_t: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        s_t = self.apply_tensor_constraints(s_t)
        x = self.model(s_t)
        return self.mu_head(x), self.sigma_head(x)


class Network2D(Network):

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    def __init__(self, x_dim, y_dim, fc_dim=512, device='cpu', **kwargs):
        # type: (tuple, tuple, int, str, dict) -> None
        super().__init__(x_dim, y_dim, device)
        self.conv_layers = kwargs['conv_layers']
        self.layers = []
        prev_channels = x_dim[0]
        prev_dim_x = x_dim[1]
        prev_dim_y = x_dim[2]
        for layer in self.conv_layers:
            self.layers.append(nn.Conv2d(prev_channels,
                                         layer['channel_num'],
                                         kernel_size=layer['kernel_size'],
                                         stride=layer['stride'],
                                         padding=layer['padding']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, y_dim[0]))
        self.model = nn.Sequential(*self.layers).to(device=device)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        s_t = self.apply_tensor_constraints(s_t)
        # s_t = s_t.to(memory_format=torch.channels_last)
        return self.model(s_t)

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        assert type(x) == torch.Tensor
        if len(tuple(x.shape)) != 3 and len(tuple(x.shape)) != 4:
            raise ValueError("Encoder input tensor should be 3D (single image) or 4D (batch).")
        if len(tuple(x.shape)) == 3:  # Add batch dimension to 3D tensor
            x = x.unsqueeze(0)
        if self.cuda and not x.is_cuda:
            x = x.to(self.device)
        assert tuple(x.shape[1:]) == self.x_dim, f'{ tuple(x.shape[1:])} {self.x_dim}'
        return x

