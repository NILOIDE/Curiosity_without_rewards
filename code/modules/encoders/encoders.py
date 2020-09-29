import torch
import torch.nn as nn
import os
import copy
from modules.encoders.base_encoders import Encoder_1D, Encoder_2D, RandomEncoder_1D, RandomEncoder_2D
from modules.encoders.vae import VAE
from modules.decoders.decoders import Decoder_2D, Decoder_2D_conv

class VAEFM(nn.Module):

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.vae = VAE(x_dim, device=device, **kwargs)  # type: VAE
        self.train_steps = 0
        self.target_encoder_steps = kwargs['wm_target_net_steps']
        self.soft_target = kwargs['wm_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.vae.encoder).to(device)
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, list, dict]
        """
        Forward pass of the world model. Returns processed intrinsic reward and keyworded
        loss components alongside the loss. Keyworded loss components are meant for bookkeeping
        and should be given as floats.
        """
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        if self.target_encoder is not None:
            with torch.no_grad():
                z_t, *_ = self.target_encoder(x_t)
                z_tp1, *_ = self.target_encoder(x_tp1)
        else:
            z_t = self.vae.encode(x_t)
            z_tp1 = self.vae.encode(x_tp1)
        z_diff = self.forward_model(z_t, a_t)
        assert not z_t.requires_grad
        assert not z_tp1.requires_grad
        loss_wm_vector = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        loss, loss_dict = None, None
        # Section necessary only for training (Calculate VAE loss and overall loss)
        if not eval:
            vae_loss, *_ = self.vae(x_t)
            loss_wm = loss_wm_vector.mean()
            self.train_steps += 1
            self.update_target_encoder()
            loss = vae_loss + loss_wm
            loss_dict = {'wm_loss': loss.detach().mean().item(),
                         'wm_trans_loss': loss_wm.detach().item(),
                         'wm_vae_loss': vae_loss.detach().item()}
        return loss_wm_vector.detach(), loss, loss_dict

    def update_target_encoder(self):
        if self.target_encoder is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_encoder.parameters(), self.vae.encoder.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_encoder_steps != 0
                if self.train_steps % self.target_encoder_steps == 0:
                    self.target_encoder = copy.deepcopy(self.vae.encoder).to(self.device)
        else:
            assert not self.soft_target or self.target_encoder_steps == 0

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), folder_path + 'encoder.pt')

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path))


class RandomEncodedFM(nn.Module):

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.trains_steps = 0

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return RandomEncoder_1D(x_dim=self.x_dim,
                                    z_dim=self.z_dim,
                                    batch_norm=kwargs['encoder_batchnorm'],
                                    device=self.device)
        else:
            return RandomEncoder_2D(x_dim=self.x_dim,
                                    conv_layers=kwargs['conv_layers'],
                                    z_dim=self.z_dim,
                                    device=self.device)

    def predict(self, x_t):
        return self.encoder(x_t)

    def forward(self, x_t):
        return self.encoder(x_t)

    def get_loss(self, **kwargs):
        return None

    def save(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), folder_path + 'encoder.pt')

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path))


class ContrastiveEncodedFM(nn.Module):
    class HingeLoss(torch.nn.Module):

        def __init__(self, hinge: float):
            super().__init__()
            self.hinge = hinge

        def apply_tensor_constraints(self, x, required_dim, name=''):
            # type: (torch.Tensor, tuple, str) -> torch.Tensor
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(dtype=torch.float32)
            assert isinstance(x, torch.Tensor), type(x)
            if len(tuple(x.shape)) == 1:  # Add batch dimension to 1D tensor
                if x.shape[0] == required_dim[0]:
                    x = x.unsqueeze(0)
                else:
                    x = x.unsqueeze(1)
            x = x.to(self.device)
            return x

        def forward(self, output, neg_samples):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            """
            Return the hinge loss between the output vector and its negative sample encodings.
            Expected dimensions of output is (batch, zdim) which gets repeated into(batch, negsamples, zdim).
            Negative sample encodings have expected shape (batch*negsamples, zdim).
            Implementation from https://github.com/tkipf/c-swm/blob/master/modules.py.
            """
            assert output.shape[0] == neg_samples.shape[0] and output.shape[1] == neg_samples.shape[2], \
                f'Received: {tuple(output.shape)} {neg_samples.shape}'
            output = output.unsqueeze(1).repeat((1, neg_samples.shape[1], 1))
            diff = output - neg_samples
            energy = diff.pow(2).sum(dim=2)
            # energy = energy.mean(dim=1)
            hinge_loss = -energy + self.hinge
            hinge_loss = hinge_loss.clamp(min=0.0)
            hinge_loss = hinge_loss.mean(dim=1)
            assert len(hinge_loss.shape) == 1
            return hinge_loss

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.train_steps = 0
        self.target_encoder_steps = kwargs['enc_target_net_steps']
        self.soft_target = kwargs['enc_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.encoder).to(device)
        self.tau = kwargs['enc_tau']
        self.neg_samples = kwargs['neg_samples']
        self.loss_func_distance = nn.MSELoss(reduction='none').to(device)
        self.loss_func_neg_sampling = self.HingeLoss(kwargs['hinge_value']).to(device)

    def forward(self, x_t, a_t, x_tp1, eval=False, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool, dict) -> [torch.Tensor, list, dict]
        if len(tuple(x_t.shape)) == 1:  # Add batch dimension to 1D tensor
            x_t = x_t.unsqueeze(0)
        if len(tuple(x_tp1.shape)) == 1:  # Add batch dimension to 1D tensor
            x_tp1 = x_tp1.unsqueeze(0)
        # Section necessary for training and eval (Calculate batch-wise translation error in latent space)
        z_t = self.encoder(x_t)
        z_diff = self.forward_model(z_t, a_t)
        if self.target_encoder is not None:
            with torch.no_grad():
                z_tp1 = self.target_encoder(x_tp1)
        else:
            z_tp1 = self.encoder(x_tp1)
        loss_trans = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        # Section necessary only for training (Calculate negative sampling error and overall loss)
        loss_ns, loss, loss_dict = None, None, None
        if not eval:
            if self.neg_samples > 0:
                if not isinstance(kwargs['memories'], torch.Tensor):
                    neg_samples = torch.from_numpy(kwargs['memories'].sample_states(self.neg_samples)).to(
                        dtype=torch.float32, device=self.device)
                else:
                    neg_samples = kwargs['memories']
                loss_ns = self.calculate_contrastive_loss(neg_samples, z_t=z_t, pos_examples_z=z_tp1)
                loss = (loss_trans + loss_ns).mean()
            else:
                loss = loss_trans.mean()
            self.train_steps += 1
            self.update_target_encoder()
            loss_dict = {'wm_loss': loss.detach().item(),
                         'wm_trans_loss': loss_trans.detach().mean().item(),
                         'wm_ns_loss': loss_ns.detach().mean().item()}
        return loss_trans.detach(), loss, loss_dict

    def calculate_contrastive_loss(self, neg_examples, x_t=None, z_t=None, pos_examples=None, pos_examples_z=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> [torch.Tensor]
        """
        Negative and Positive examples are converted into a (batch, examples, zdim) structure in this function.
        """
        if z_t is None:
            assert x_t is not None, "Either x_t or z_t should be None."
            z_t = self.encoder(x_t)
        else:
            assert x_t is None, "Either x_t or z_t should be None."
        if len(tuple(z_t.shape)) == 1:  # Add batch dimension to 1D tensor
            z_t = z_t.unsqueeze(0)
        if len(tuple(neg_examples.shape)) == 1:  # Add batch dimension to 1D tensor
            neg_examples = neg_examples.unsqueeze(0)

        neg_samples_z = self.encoder(neg_examples)
        neg_samples_z = neg_samples_z.unsqueeze(0)
        neg_samples_z = neg_samples_z.repeat((z_t.shape[0], 1, 1))
        loss = self.loss_func_neg_sampling(z_t, neg_samples_z)
        # Positive examples below
        if pos_examples is not None:
            assert pos_examples_z is None, "Either x_t or z_t should be None."
            if len(tuple(pos_examples.shape)) == 1:  # Add batch dimension to 1D tensor
                pos_examples = pos_examples.unsqueeze(0)
            pos_examples_z = self.encoder(pos_examples)
        if pos_examples_z is not None:
            if len(tuple(pos_examples_z.shape)) == 1:  # Add batch dimension to 1D tensor
                pos_examples_z = pos_examples_z.unsqueeze(0)
            if len(tuple(pos_examples_z.shape)) == 2:
                if z_t.shape[0] == 1:
                    pos_examples_z = pos_examples_z.unsqueeze(0)
                else:
                    # This accepts 2D pos example tensors where every z_t has the same amount of pos examples
                    assert pos_examples_z.shape[0] % z_t.shape[0] == 0, \
                        "There should be an equal amount of pos examples per z."
                    pos_examples_z = pos_examples_z.view((z_t.shape[0], pos_examples_z.shape[0] // z_t.shape[0], -1))
            z_t_expanded = z_t.unsqueeze(1).repeat((1, pos_examples_z.shape[1], 1))
            pos_loss = (z_t_expanded - pos_examples_z).pow(2).sum(dim=2) - self.loss_func_neg_sampling.hinge
            pos_loss = pos_loss.clamp(min=0.0).mean(dim=1)
            loss += pos_loss
        return loss.mean()

    def forward_fm_only(self, x_t, a_t, x_tp1, eval=False):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, bool) -> [torch.Tensor, list, dict]
        """"
        Forward pass where gradients are only applied to the forward model.
        """
        if len(tuple(x_t.shape)) == 1:  # Add batch dimension to 1D tensor
            x_t = x_t.unsqueeze(0)
        if len(tuple(x_tp1.shape)) == 1:  # Add batch dimension to 1D tensor
            x_tp1 = x_tp1.unsqueeze(0)
        with torch.no_grad():
            z_t = self.encoder(x_t)
            if self.target_encoder is not None:
                z_tp1 = self.target_encoder(x_tp1)
            else:
                z_tp1 = self.encoder(x_tp1)
        z_diff = self.forward_model(z_t, a_t)
        loss_vector = self.loss_func_distance(z_t + z_diff, z_tp1).sum(dim=1)
        loss = loss_vector.mean()
        loss_dict = None
        if not eval:
            self.train_steps += 1
            self.update_target_encoder()
            loss_dict = {'wm_loss': loss.detach().item(),
                         'wm_trans_loss': loss.detach().mean().item()}
        return loss_vector.detach(), loss, loss_dict

    def update_target_encoder(self):
        if self.target_encoder is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_encoder_steps != 0
                if self.train_steps % self.target_encoder_steps == 0:
                    self.target_encoder = copy.deepcopy(self.encoder)
        else:
            assert not self.soft_target or self.target_encoder_steps == 0

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Encoder_1D(x_dim=self.x_dim,
                              z_dim=self.z_dim,
                              batch_norm=kwargs['encoder_batchnorm'],
                              device=self.device)  # type: Encoder_1D
        else:
            return Encoder_2D(x_dim=self.x_dim,
                              conv_layers=kwargs['conv_layers'],
                              z_dim=self.z_dim,
                              device=self.device)  # type: Encoder_2D

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            z_t = self.encode(x_t)
            z_diff = self.forward_model(z_t, a_t)
            return z_t + z_diff

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path):
        self.encoder.load_state_dict(torch.load(path))


class DeterministicInvDynFeatFM:
    class InverseModel(torch.nn.Module):
        def __init__(self, phi_dim, a_dim, hidden_dim=(64,), batch_norm=False, device='cpu'):
            # type: (tuple, tuple, tuple, bool, str) -> None
            """"
            Network responsible for action prediction given s_t and s_tp1.
            Any number of arbitrary convolutional layers can used. A layer is represented using a dictionary.
            Only one fully-connected layer is used. The latent representation is assumed to be Gaussian.
            """
            super().__init__()
            self.phi_dim = phi_dim
            self.a_dim = a_dim
            self.device = device
            self.layers = []
            h_dim_prev = self.phi_dim[0] * 2
            for h_dim in hidden_dim:
                self.layers.append(nn.Linear(h_dim_prev, h_dim))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(h_dim))
                self.layers.append(nn.ReLU())
                h_dim_prev = h_dim
            self.layers.append(nn.Linear(hidden_dim[-1], self.a_dim[0]))
            self.layers.append(nn.Softmax(dim=1))
            self.model = nn.Sequential(*self.layers).to(self.device)

        def forward(self, phi_t, phi_tp1):
            # type: (torch.tensor, torch.tensor) -> torch.tensor
            x = torch.cat((phi_t, phi_tp1), dim=1)
            a_pred = self.model(x)
            return a_pred

    def __init__(self, x_dim, a_dim, device='cpu', **kwargs):
        # type: (tuple, tuple, str, dict) -> None
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = kwargs['z_dim']
        self.device = device
        self.encoder = self.create_encoder(len(x_dim), **kwargs)
        self.train_steps = 0
        self.target_encoder_steps = kwargs['enc_target_net_steps']
        self.soft_target = kwargs['enc_soft_target']
        self.target_encoder = None
        if self.soft_target or self.target_encoder_steps != 0:
            self.target_encoder = copy.deepcopy(self.encoder).to(device)
        self.tau = kwargs['enc_tau']
        self.inverse_model = self.InverseModel(self.encoder.get_z_dim(),
                                               self.a_dim,
                                               hidden_dim=kwargs['idf_inverse_hdim'],
                                               device=device)
        self.loss_func_inverse = nn.MSELoss().to(device)

    def predict(self, x_t):
        with torch.no_grad():
            if self.target_encoder is not None:
                z_t = self.target_encoder(x_t)
            else:
                z_t = self.encoder(x_t)
        return z_t

    def forward(self, x_t):
        return self.encoder(x_t)

    def get_loss(self, x_t, a_t, x_tp1, **kwargs):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, dict) -> torch.Tensor
        phi_t = self.encoder(x_t)
        phi_tp1 = self.encoder(x_tp1)
        if self.target_encoder is not None:
            with torch.no_grad():
                phi_t = self.target_encoder(x_t)
                phi_tp1 = self.target_encoder(x_tp1)
        a_t_pred = self.inverse_model(phi_t, phi_tp1)
        loss = self.loss_func_inverse(a_t_pred, a_t)
        return loss.mean()

    def update_target_encoder(self):
        if self.target_encoder is not None:
            if self.soft_target:
                for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - 0.01) + param.data * 0.01)
            else:
                assert self.target_encoder_steps != 0
                if self.train_steps % self.target_encoder_steps == 0:
                    self.target_encoder = copy.deepcopy(self.encoder)
        else:
            assert not self.soft_target or self.target_encoder_steps == 0

    def create_encoder(self, input_dim, **kwargs):
        if input_dim == 1:
            return Encoder_1D(x_dim=self.x_dim,
                              z_dim=self.z_dim,
                              batch_norm=kwargs['encoder_batchnorm'],
                              device=self.device)  # type: Encoder_1D
        else:
            return Encoder_2D(x_dim=self.x_dim,
                              conv_layers=kwargs['conv_layers'],
                              z_dim=self.z_dim,
                              device=self.device)  # type: Encoder_2D

    def encode(self, x):
        return self.encoder(x)

    def target_encode(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def next_z_from_z(self, z_t, a_t):
        with torch.no_grad():
            return self.forward_model(z_t, a_t)

    def next(self, x_t, a_t):
        with torch.no_grad():
            z_t = self.encode(x_t)
            return self.forward_model(z_t, a_t)

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path))

