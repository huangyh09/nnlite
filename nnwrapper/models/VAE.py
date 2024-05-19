# Adapted from tutorial examples:
# https://medium.com/@rekalantar/dce2d2fe0f5f
# https://github.com/pytorch/examples/blob/main/vae/main.py (PyTorch default)
# https://towardsdatascience.com/3a06bee395ed (with pytorch-lightiing)

# TODO: to test the pbmc_3k data (pre-processed h5ad data to be included here)

import math
import torch
import torch.nn as nn

class AE_base(nn.Module):
    def __init__(self, dim_x, dim_z, hidden_dims=[], 
        fc_activation=torch.nn.ReLU()):
        """
        Autoencoder supporting variable number of hidden layers
        """
        super(AE_base, self).__init__()

        # check hidden layers
        H = len(hidden_dims)
        encode_dim = dim_x if H == 0 else hidden_dims[-1]
        decode_dim = dim_z if H == 0 else hidden_dims[0]

        # encoder
        self.encoder = torch.nn.Sequential(nn.Identity())
        for h, out_dim in enumerate(hidden_dims):
            in_dim = dim_x if h == 0 else hidden_dims[h - 1]
            self.encoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            self.encoder.add_module("A%s" %(h), fc_activation)
        self.encoder.add_module("L%s" %(H), nn.Linear(encode_dim, dim_z))
        
        # decoder
        self.decoder = nn.Sequential()
        for h, out_dim in enumerate(hidden_dims[::-1]):
            in_dim = dim_z if h == 0 else hidden_dims[::-1][h - 1]
            self.decoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            self.decoder.add_module("A%s" %(h), fc_activation)
        self.decoder.add_module("L%s" %(H), nn.Linear(decode_dim, dim_x))

        # criterion
        self.criterion = nn.MSELoss()
    
    # def criterion(self, input, target, fc_loss=nn.MSELoss()):
    #     # _loss = torch.mean(torch.square(input - target))
    #     _loss = fc_loss(input, target)
    #     return _loss
        
    def forward(self, x):
        x_pred = self.decoder(self.encoder(x))
        return x_pred


def loss_VAE_Gaussian(x, x_hat, x_logstd, z_mean, z_logstd, beta=1.0):
    """
    VAE loss with Gaussian noise model
    Note, this will be in a batch format
    Eq. 10 & Appendix B in VAE paper: https://arxiv.org/abs/1312.6114
    """
    def gaussian_loglik(x, x_hat, x_logstd):
        _loglik = (-0.5 * torch.square((x_hat - x) / torch.exp(x_logstd)) - 
                   x_logstd * math.log(math.sqrt(2 * math.pi)))
        # _loglik = (-0.5 * torch.square((x_hat - x)))
        #torch.mean(torch.sum(_loglik, axis=-1))
        return torch.mean(_loglik)

    def kl_divergence(z_mean, z_logstd):
        """
        Monte carlo KL divergence with one sample
        prior_mu=0.0, prior_sigma=1.0
        """
        # TO check
        _kl = -0.5 * (1 + 2 * z_logstd - torch.square(z_mean) - 
                      torch.square(torch.exp(z_logstd)))
        #torch.mean(torch.sum(_kl, dim=-1))
        return torch.mean(_kl)

    loglik = gaussian_loglik(x, x_hat, x_logstd)
    kl = kl_divergence(z_mean, z_logstd)

    return -(loglik - beta * kl)


class VAE_base(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dims=[], fit_xscale=True, 
        device='cpu', fc_activate=torch.nn.ReLU()):
        # TODO: to support activation function as argument
        """
        Variational auto-encoder base model:
        An implementation supporting customized hidden layers via a list.

        Parameters
        ----------
        
        Examples
        --------
        my_VAE = VAE_base()
        my_VAE.encoder = resnet18_encoder(False, False) # to change encoder
        """
        super(VAE_base, self).__init__()
        self.device = device

        # check hiden layers
        # TODO: check int and None
        H = len(hidden_dims)
        encode_dim = x_dim if H == 0 else hidden_dims[-1]
        decode_dim = z_dim if H == 0 else hidden_dims[0]

        # encoder
        self.encoder = torch.nn.Sequential(nn.Identity())
        for h, out_dim in enumerate(hidden_dims):
            in_dim = x_dim if h == 0 else hidden_dims[h - 1]
            self.encoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            self.encoder.add_module("A%s" %(h), torch.nn.ReLU())

        # latent mean and diagonal variance 
        self.fc_z_mean = nn.Linear(encode_dim, z_dim)
        self.fc_z_logstd = nn.Linear(encode_dim, z_dim)
        
        # decoder
        self.decoder = nn.Sequential()
        for h, out_dim in enumerate(hidden_dims[::-1]):
            in_dim = z_dim if h == 0 else hidden_dims[::-1][h - 1]
            self.decoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            self.decoder.add_module("A%s" %(h), torch.nn.ReLU())
        # self.decoder.add_module("L%s" %(H), nn.Linear(decode_dim, x_dim))

        # reconstruction mean and diagonal variance (likelihood)
        self.fc_x_mean = nn.Linear(decode_dim, x_dim)
        self.fc_x_logstd = nn.Linear(1, x_dim, bias=False) #(1, x_dim)
    
    def encode(self, x):
        _x = self.encoder(x)
        z_mean, z_logstd = self.fc_z_mean(_x), self.fc_z_logstd(_x)
        return z_mean, z_logstd

    def reparameterization(self, z_mean, z_logstd):
        epsilon = torch.randn_like(z_mean).to(self.device)
        z = z_mean + torch.exp(z_logstd) * epsilon
        return z

    def decode(self, z):
        _z = self.decoder(z)
        x_mean = self.fc_x_mean(_z)
        # x_logstd = self.fc_x_logstd(_z)
        x_logstd = self.fc_x_logstd(torch.ones(1).to(self.device)).reshape((1, -1))
        return x_mean, x_logstd

    def forward(self, x):
        z_mean, z_logstd = self.encode(x)
        z = self.reparameterization(z_mean, z_logstd)
        x_hat, x_logstd = self.decode(z)
        return x_hat, x_logstd, z, z_mean, z_logstd

    def criterion(self, input, target, fc_loss=loss_VAE_Gaussian):
        # x_hat, z, z_mean, z_logstd, x_scale 
        x_hat, x_logstd, z, z_mean, z_logstd = input
        # print(x_hat[0].shape, x_logstd[0].shape, z_logstd[0].shape, target[0].shape)

        _loss = fc_loss(target, x_hat, x_logstd, z_mean, z_logstd)
        # _loss = torch.mean(torch.square(x_hat - target))
        return _loss
