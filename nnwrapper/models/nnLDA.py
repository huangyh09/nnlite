# Amoritized variational inference for Latent Dirichlet Allocation model
# Paper: https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

import math
import torch
import torch.nn as nn
from functools import partial

# NOTE: still draft prototype
class nnLDA(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dims=[], fit_xscale=True, 
        device='cpu', fc_activate=torch.nn.ReLU()):
        """
        Variational auto-encoder base model:
        An implementation supporting customized hidden layers via a list.

        For the likelihood, the original Dirichlet-multinomial likelihood
        will be replaced (approximated) by negative-binomial likelhood.

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
        self.W = nn.parameter.Parameter(torch.zeros(decode_dim, x_dim))
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

    def forward(self, x, lib_size):
        z_mean, z_logstd = self.encode(x)
        z = self.reparameterization(z_mean, z_logstd)
        H = nn.Softmax(z, dim=-1)
        W = nn.Softmax(self.W + self.offsets, dim=-1)
        
        x_hat = H @ W * lib_size
        return x_hat, self.log_phi, z, z_mean, z_logstd


def Loss_nnLDA_NB(result, target, beta=1.0, fix_phi_log=True):
    def negabin_loglik(x, x_hat, log_phi):
        """negative binomial approximates Dirichlet-multinomial
        (https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution#
        Related_distributions)

        Parameters
        ----------
        x: observed n_failure
        x_hat: expected n_failure
        log_phi: log scale of over dispersion
        """
        from torch.special import gammaln
        phi = torch.exp(log_phi)     # over-dispersion
        var = x_hat + phi * x_hat**2 # variance
        n = 1 / phi                  # n_success, i.e., concentration
        p = n / (x_hat + n)          # probability of success
        log_p = -log_phi - torch.log(x_hat + n)
        log_q = torch.log(x_hat) - torch.log(x_hat + n)
        _loglik = (gammaln(x + n)  - gammaln(x + 1) - gammaln(n) +
                   n * log_p + x * log_q)
        return torch.mean(torch.sum(_loglik, dim=-1))
    def kl_divergence(z_mean, z_logstd, prior_mu=0.0, prior_sigma=1.0):
        """logit-normal distribution
        Not implemented yet, requiring Monte Carlo or heuristic approximation
        https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
        """
        return 0

    # Parse input
    x_hat, log_phi, z, z_mean, z_logstd = result
    if fix_x_var:
        log_phi = log_phi * 0 - 1.0

    # Calculate loss
    loglik = negabin_loglik(target, x_hat, log_phi)
    kl = kl_divergence(z_mean, z_logstd, prior_mu, prior_sigma)
    return -(loglik - beta * kl)