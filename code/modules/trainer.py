import torch


class Trainer:
    def __init__(self, x_dim, a_dim, policy, wm, encoder=None, **kwargs):
        self.train_steps = 0
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.policy = policy
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=kwargs['policy_lr'])
        self.wm = wm
        self.wm_opt = torch.optim.SGD(wm.parameters(), lr=kwargs['wm_lr'])
        self.wm_loss_func = torch.nn.MSELoss()
        self.encoder = encoder
        if encoder is not None:
            self.encoder_opt = torch.optim.SGD(encoder.parameters(), lr=kwargs['enc_lr'])  # Loss is provided by
        self.losses = {'policy': [], 'wm': [], 'encoder': []}

    def train_step(self, x_t, a_t, x_tp1, **kwargs):
        self.train_wm(x_t, a_t, x_tp1)
        self.train_policy(x_t, x_tp1)
        if self.encoder is not None:
            self.train_encoder(**{'x_t': x_t, 'a_t': a_t, 'x_tp1': x_tp1, **kwargs})
        self.train_steps += 1

    def train_wm(self, x_t, a_t, x_tp1):
        self.wm_opt.zero_grad()
        loss = self.wm_loss_func(self.wm(x_t, a_t), self.wm.get_target(x_tp1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.wm_opt.step()
        self.wm.train_steps += 1
        self.wm.update_target_network()
        self.losses['wm'].append(loss.item())

    def train_policy(self, x_t, x_tp1):
        self.policy_opt.zero_grad()
        a_t = self.policy(x_t)
        # We want to maximize world model loss (thus the negative sign)
        loss = -self.wm_loss_func(self.wm(x_t, a_t), self.wm.get_target(x_tp1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_opt.step()
        self.policy.train_steps += 1
        self.policy.update_target_network()
        self.losses['policy'].append(loss.item())

    def train_encoder(self, **kwargs):
        self.encoder_opt.zero_grad()
        loss = self.encoder.get_loss(**kwargs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.encoder_opt.step()
        self.encoder.train_steps += 1
        self.encoder.update_target_network()
        self.losses['encoder'].append(loss.item())

    def save_models(self, path):
        self.policy.save(path)
        self.wm.save(path)
        if self.encoder is not None:
            self.encoder.save(path)
