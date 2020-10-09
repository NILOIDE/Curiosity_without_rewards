import torch
from .network import Network1D, Network2D
import os
import copy


class Policy(torch.nn.Module):
    def __init__(self, x_dim, a_dim, sigma, discrete_action=False, device='cpu', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.sigma = sigma  # Exploration gaussian noise Sigma
        self.discrete_action = discrete_action
        self.device = device
        if len(x_dim) == 1:
            self.network = Network1D(x_dim=x_dim, hidden_dim=kwargs['policy_h_dim'], y_dim=a_dim)
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

    def forward(self, x_t, noise=None):
        y = self.network(x_t)
        if noise is not None:
            y += noise
        if self.discrete_action:
            y = torch.nn.functional.softmax(y, dim=0)
        return y

    def predict(self, x_t, noise=None):
        with torch.no_grad():
            return self.forward(x_t, noise)

    def act(self, x_t, sigma=None):
        # type: (torch.Tensor, float) -> (torch.Tensor, torch.Tensor)
        """" Predict action a_t based on input state x_t.
        Args:
            x_t (torch.Tensor): Current observation.
            sigma (float): Standard deviation of Gaussian noise to be added to action.
        Returns:
            (torch.Tensor, torch.Tensor): Action taken by policy. Noise added to action.
        """
        with torch.no_grad():
            if self.target_network is None:
                a_t = self.network(x_t).squeeze(0)
            else:
                a_t = self.target_network(x_t).squeeze(0)
        sigma = self.sigma if sigma is None else sigma
        if sigma == 0.0:
            if self.discrete_action:
                a_t = torch.nn.functional.softmax(a_t, dim=0)
            return a_t, torch.zeros_like(a_t)
        else:
            noise = torch.randn(a_t.shape) * sigma
            a_t = a_t + noise
            if self.discrete_action:
                a_t = torch.nn.functional.softmax(a_t, dim=0)
            return a_t, noise

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


class PolicyStochastic(torch.nn.Module):
    def __init__(self, x_dim, a_dim, tau, discrete_action=False, device='cpu', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.tau = tau  # Temperature for action distribution
        self.discrete_action = discrete_action
        self.device = device
        if len(x_dim) == 1:
            self.network = Network1D(x_dim=x_dim, hidden_dim=kwargs['policy_h_dim'], y_dim=a_dim)
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
        mu, sigma = self.network(x_t)
        return mu + torch.randn(self.a_dim[0]) * (sigma + self.tau)

    def predict(self, x_t):
        with torch.no_grad():
            with torch.no_grad():
                if self.target_network is None:
                    mu, sigma = self.network(x_t)
                else:
                    mu, sigma = self.target_network(x_t)
        return mu + torch.randn(self.a_dim[0]) * (sigma + self.tau)

    def act(self, x_t, tau=None):
        # type: (torch.Tensor, float) -> (torch.Tensor, torch.Tensor)
        """" Predict action a_t based on input state x_t.
        Args:
            x_t (torch.Tensor): Current observation.
            tau (float): Temperature to be added to action distribution (Gaussian).
        Returns:
            (torch.Tensor, torch.Tensor): Action taken by policy. Noise added to action.
        """
        with torch.no_grad():
            if self.target_network is None:
                mu, sigma = self.network(x_t)
            else:
                mu, sigma = self.target_network(x_t)
        tau = self.tay if tau is None else tau
        noise = torch.randn(self.a_dim[0]) * (sigma + tau)
        return mu + noise, noise

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
