# Amortized variational inference for Latent Dirichlet Allocation model
# Paper: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
# Wiki: https://en.wikipedia.org/wiki/Topic_model#Algorithms

import math
import torch
import torch.nn as nn
from functools import partial
from .stats import negabin_loglik, gaussian_kl

class nnLDA(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dims=[], fit_xscale=True, 
        device='cpu', fc_activate=torch.nn.ReLU()):
        """
        NeuralNet variational inference for Latent Dirichlet Allocation model.
        An implementation supporting customized hidden layers via a list.

        For the likelihood, the original Dirichlet-multinomial likelihood
        will be replaced (approximated) by negative-binomial likelihood.

        Parameters
        ----------
        
        Examples
        --------
        my_nnLDA = nnLDA()
        my_nnLDA.encoder = resnet18_encoder(False, False) # to change encoder
        """
        super(nnLDA, self).__init__()
        self.device = device

        # check hiden layers
        # TODO: check int and None
        h_layers = len(hidden_dims)
        encode_dim = x_dim if h_layers == 0 else hidden_dims[-1]
        decode_dim = z_dim if h_layers == 0 else hidden_dims[0]

        # encoder
        self.encoder = torch.nn.Sequential(nn.Identity())
        for h, out_dim in enumerate(hidden_dims):
            in_dim = x_dim if h == 0 else hidden_dims[h - 1]
            self.encoder.add_module("L%s" %(h), nn.Linear(in_dim, out_dim))
            self.encoder.add_module("A%s" %(h), torch.nn.ReLU())

        # latent mean and diagonal variance 
        self.fc_z_mean = nn.Linear(encode_dim, z_dim)
        self.fc_z_logstd = nn.Linear(encode_dim, z_dim)
        
        # decoder (inherited from VAE style) as parameters
        self.W = nn.parameter.Parameter(torch.zeros(z_dim, x_dim))
        self.offsets = nn.parameter.Parameter(torch.zeros(1, x_dim))
        self.log_phi = nn.parameter.Parameter(torch.zeros(1, x_dim))
        # self.log_phi = torch.zeros(1, x_dim).to(self.device)
    
    def encode(self, x):
        _x = self.encoder(x)
        z_mean, z_logstd = self.fc_z_mean(_x), self.fc_z_logstd(_x)
        return z_mean, z_logstd

    def reparameterization(self, z_mean, z_logstd):
        epsilon = torch.randn_like(z_mean).to(self.device)
        z = z_mean + torch.exp(z_logstd) * epsilon
        return z

    def forward(self, x):
        z_mean, z_logstd = self.encode(x)
        z = self.reparameterization(z_mean, z_logstd)
        H = torch.nn.functional.softmax(z, dim=-1)
        W = torch.nn.functional.softmax(self.W + self.offsets, dim=-1)
        
        x_prop = H @ W
        return x_prop, self.log_phi, z, z_mean, z_logstd


def Loss_nnLDA_NB(result, target, lib_size=None, beta=1.0, fix_phi_log=-1.0):
    """
    Negative binomial approximates Dirichlet-multinomial
    (https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution#
    Related_distributions)
    """
    # Parse input
    x_prop, log_phi, z, z_mean, z_logstd = result
    x_target = target
    if lib_size is None:
        lib_size = torch.sum(x_target, dim=-1, keepdim=True)
    x_hat = x_prop * lib_size

    # Check if using a fixed phi or the learned phi
    if fix_phi_log is False:
        pass
    elif fix_phi_log is True:
        log_phi = log_phi * 0 - 1
    else:
        log_phi = log_phi * 0 + float(fix_phi_log)

    # Calculate loss
    loglik = negabin_loglik(x_target, x_hat, log_phi)
    kl = gaussian_kl(z_mean, z_logstd)

    return -(loglik - beta * kl)
