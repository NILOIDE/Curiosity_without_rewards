import torch
import torch.nn as nn


class BaseEncoder(nn.Module):

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size()[0], -1)

    STANDARD_CONV = ({'channel_num': 32, 'kernel_size': 8, 'stride': 4, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 4, 'stride': 2, 'padding': 0},
                     {'channel_num': 64, 'kernel_size': 3, 'stride': 1, 'padding': 0})

    def __init__(self, x_dim, z_dim, device='cpu'):
        # type: (tuple, tuple, str) -> None
        """"
        This is the base class for the encoders below. Contains all the required shared variables and functions.
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        if device in {'cuda', 'cpu'}:
            self.device = device
            self.cuda = True if device == 'cuda' else False
        print('Encoder has dimensions:', x_dim, '->', z_dim, 'Device:', self.device)

    def apply_tensor_constraints(self, x: torch.Tensor) -> torch.Tensor:
        assert type(x) == torch.Tensor
        input_dims = len(self.x_dim)
        expected_dims = len(tuple(x.shape))
        if input_dims == expected_dims:  # Add batch dimension to tensor
            x = x.unsqueeze(0)
        elif input_dims == expected_dims+1:
            raise ValueError("Encoder input tensor should be "+str(expected_dims)+"D (single example) or "+
                             str(expected_dims+1)+"D (batch).")
        x = x.to(self.device)
        assert tuple(x.shape[-expected_dims:]) == self.x_dim
        return x

    def get_z_dim(self) -> tuple:
        return self.z_dim

    def get_x_dim(self) -> tuple:
        return self.x_dim


# ---------------------- RANDOM ENCODERS ------------------------------

class RandomEncoder_1D(BaseEncoder):
    def __init__(self, x_dim, hidden_dim=(64,), z_dim=(64,), batch_norm=False, device='cpu'):
        # type: (tuple, tuple, tuple, bool, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(h_dim_prev, z_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)


class RandomEncoder_2D(BaseEncoder):

    def __init__(self, x_dim=(3, 84, 84), conv_layers=None, fc_dim=256, z_dim=(32,), device='cpu'):
        # type: (tuple, tuple, int, tuple, str) -> None
        """"
        This enconder has static weights as no gradients will be calculated. It provides static features.
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.conv_layers = self.STANDARD_CONV if conv_layers is None else conv_layers
        self.layers = []
        prev_channels = self.x_dim[0]
        prev_dim_x = self.x_dim[1]
        prev_dim_y = self.x_dim[2]
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
            if prev_dim_x <= 0 or prev_dim_y <= 0 or prev_channels <= 0:
                raise ValueError("Conv dimensions must be positive: " + str((prev_channels, prev_dim_x, prev_dim_y)))
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, *self.z_dim))
        self.model = nn.Sequential(*self.layers).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and std with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)

# ---------------------- LEARNT ENCODERS ------------------------------

class Encoder_2D_Sigma(BaseEncoder):

    def __init__(self, x_dim, z_dim, conv_layers=None, fc_dim=512, batch_norm=True, device='cpu'):
        # type: (tuple, tuple, tuple, int, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.conv_layers = self.STANDARD_CONV if conv_layers is None else conv_layers
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
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(layer['channel_num']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        # if batch_norm:
        #     self.layers.append(nn.BatchNorm1d(fc_dim))  # Causes issues if batch size is 1
        self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)
        self.sigma_head = nn.Sequential(nn.Linear(fc_dim, *self.z_dim)).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mu = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mu, log_sigma


class Encoder_2D(BaseEncoder):

    def __init__(self, x_dim, z_dim, conv_layers=None, fc_dim=512, batch_norm=True, device='cpu'):
        # type: (tuple, tuple, tuple, int, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.conv_layers = self.STANDARD_CONV if conv_layers is None else conv_layers
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
            if batch_norm:
                self.layers.append(nn.BatchNorm2d(layer['channel_num']))
            self.layers.append(nn.ReLU())
            prev_channels = layer['channel_num']
            prev_dim_x = (prev_dim_x + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
            prev_dim_y = (prev_dim_y + 2 * layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
        self.layers.append(self.Flatten())
        self.layers.append(nn.Linear(prev_dim_x * prev_dim_y * prev_channels, fc_dim))
        # if batch_norm:
        #     self.layers.append(nn.BatchNorm1d(fc_dim))  # Causes issues if batch size is 1
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(fc_dim, self.z_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns encodings with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)


class Encoder_1D_Sigma(BaseEncoder):

    def __init__(self, x_dim, z_dim, hidden_dim=(64,), batch_norm=False, device='cpu'):
        # type: (tuple, tuple, tuple, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = self.x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.model = nn.Sequential(*self.layers).to(self.device)
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim[-1], *self.z_dim)).to(self.device)
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim[-1], *self.z_dim)).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns mean and variance with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        output = self.model(x)
        mu = self.mu_head(output)
        log_sigma = self.sigma_head(output)
        return mu, log_sigma


class Encoder_1D(BaseEncoder):

    def __init__(self, x_dim, z_dim, hidden_dim=(64,), batch_norm=True, device='cpu'):
        # type: (tuple, tuple, tuple, bool, str) -> None
        """"
        Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
        Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
        """
        super().__init__(x_dim, z_dim, device)
        self.layers = []
        h_dim_prev = self.x_dim[0]
        for h_dim in hidden_dim:
            self.layers.append(nn.Linear(h_dim_prev, h_dim))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dim))
            self.layers.append(nn.ReLU())
            h_dim_prev = h_dim
        self.layers.append(nn.Linear(hidden_dim[-1], self.z_dim[0]))
        self.model = nn.Sequential(*self.layers).to(self.device)

    def forward(self, x):
        # type: (torch.Tensor) -> [torch.Tensor, torch.Tensor]
        """
        Perform forward pass of encoder. Returns encodings with shape [batch_size, z_dim].
        Make sure that any constraints are enforced.
        """
        x = self.apply_tensor_constraints(x)
        return self.model(x)
