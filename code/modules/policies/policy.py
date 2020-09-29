import torch
from .network import Network1D, Network2D
import os
import copy


class Policy(torch.nn.Module):
    def __init__(self, x_dim, a_dim, sigma, device='cpu', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.sigma = sigma  # Exploration gaussian noise Sigma
        self.device = device
        if len(x_dim) == 1:
            self.network = Network1D(x_dim=x_dim, y_dim=a_dim)
        elif len(x_dim) == 3:
            self.network = Network2D(x_dim=x_dim, y_dim=a_dim)
        else:
            raise ValueError('What are these policy input dims???', x_dim)
        print('Policy architecture:')
        print(self.network)
        self.train_steps = 0
        self.target_network_steps = kwargs['policy_target_net_steps']
        self.soft_target = kwargs['policy_soft_target']
        self.tau = kwargs['policy_tau']
        self.target_network = None
        if self.soft_target or self.target_network_steps != 0:
            self.target_network = copy.deepcopy(self.network).to(device)

    def forward(self, x_t):
        return self.network(x_t)

    def act(self, x_t, sigma=None):
        # type: (torch.Tensor, float) -> torch.Tensor
        """" Predict action a_t based on input state x_t.
        Args:
            x_t (torch.Tensor): Current observation.
            sigma (float): Standard deviation of Gaussian nosie to be added to action.
        Returns:
            (torch.Tensor): Action taken by policy.
        """
        with torch.no_grad():
            if self.target_network is None:
                a_t = self.network(x_t)
            else:
                a_t = self.target_network(x_t)
        sigma = self.sigma if sigma is None else sigma
        if sigma == 0.0:
            return a_t[0]
        else:
            return a_t[0] + torch.randn(self.a_dim[0])

    def update_target_network(self):
        if self.target_network is None:
            return
        if self.soft_target:
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
        else:
            assert self.target_network_steps != 0
            if self.train_steps % self.target_network_steps == 0:
                self.target_network = copy.deepcopy(self.network)

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.network.state_dict(), folder_path + 'policy.pt')

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
