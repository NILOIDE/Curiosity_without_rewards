import torch
import torch.nn as nn
import numpy as np
from modules.world_models.forward_model import ForwardModel
import copy
import os



class EncodedWorldModel(nn.Module):
    def __init__(self, x_dim, a_dim, z_dim, encoder, device='cpu', **kwargs):
        # type: (tuple, tuple, tuple, nn.Module, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.device = device
        print('Observation space:', self.x_dim)
        self.encoder = encoder
        self.network = ForwardModel(x_dim, a_dim, hidden_dim=kwargs['wm_h_dim'], device=self.device)
        print('WM Architecture:')
        print(self.encoder)
        print(self.network)
        self.train_steps = 0
        self.target_network_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.tau = kwargs['wm_tau']
        self.target_network = None
        if self.soft_target or self.target_network_steps != 0:
            self.target_network = copy.deepcopy(self.network).to(device)

    def predict(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        with torch.no_grad():
            z_t = self.encoder(x_t)
            if self.target_network is None:
                z_tp1_prime = self.network(z_t, a_t)
            else:
                z_tp1_prime = self.target_network(z_t, a_t)
        return z_tp1_prime

    def forward(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        z_t = self.encoder(x_t)
        z_tp1_prime = self.network(z_t, a_t)
        return z_tp1_prime

    def get_target(self, x_tp1):
        with torch.no_grad():
            z_tp1 = self.encoder(x_tp1)
        z_tp1 = self.network.apply_state_constraints(z_tp1)
        return z_tp1

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
        torch.save(self.network.state_dict(), folder_path + 'wm.pt')

    def load(self, path):
        self.network.load_state_dict(torch.load(path))


class WorldModel(nn.Module):
    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.device = device
        print('Observation space:', self.x_dim)
        self.network = ForwardModel(x_dim, a_dim, hidden_dim=kwargs['wm_h_dim'], device=self.device)
        print('WM Architecture:')
        print(self.network)
        self.train_steps = 0
        self.target_network_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.tau = kwargs['wm_tau']
        self.target_network = None
        if self.soft_target or self.target_network_steps != 0:
            self.target_network = copy.deepcopy(self.network).to(device)

    def forward(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        return self.network(x_t, a_t)

    def predict(self, x_t, a_t):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        with torch.no_grad():
            if self.target_network is None:
                x_tp1_prime = self.network(x_t, a_t)
            else:
                x_tp1_prime = self.target_network(x_t, a_t)
        return x_tp1_prime

    def get_target(self, x_tp1):
        x_tp1 = self.network.apply_state_constraints(x_tp1)
        return x_tp1

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
        torch.save(self.network.state_dict(), folder_path + 'wm.pt')

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
