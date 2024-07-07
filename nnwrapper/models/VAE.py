# Variational Auto-Encoder with flexible hidden layers

# Adapted from tutorial examples:
# https://medium.com/@rekalantar/dce2d2fe0f5f
# https://github.com/pytorch/examples/blob/main/vae/main.py (PyTorch default)
# https://towardsdatascience.com/3a06bee395ed (with pytorch-lightiing)


import math
import torch
import torch.nn as nn
from functools import partial

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
    
    def forward(self, x):
        x_pred = self.decoder(self.encoder(x))
        return x_pred


class VAE_base(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dims=[], fit_xscale=True, 
        device='cpu', fc_activate=torch.nn.ReLU()):
        """
        Variational auto-encoder base model:
        An implementation supporting customized hidden layers via a list.

        For the likelihood, there are three common choices for the variance:
        1) a scalar of variance as hyper-parameter, like PCA
        2) a vector of variance as hyper-parameter, like factor analysis
        3) a matrix of variance as amoritized over z, but it can be unstable
        Here, we choose option 2) via fc_x_logstd() for better stabilization.
        We also use torch.clamp to clip the very small values.
        Now, we also leave this choice as an argument of the loss function,
        including option 0) with a predefined value 1, like MSE loss.

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

        # reconstruction mean and diagonal variance (likelihood)
        self.fc_x_mean = nn.Linear(decode_dim, x_dim)
        self.fc_x_logstd = nn.Linear(1, x_dim, bias=False) #(1, 1)
    
    def encode(self, x):
        """For variational posterior distribution"""
        _x = self.encoder(x)
        z_mean, z_logstd = self.fc_z_mean(_x), self.fc_z_logstd(_x)
        return z_mean, z_logstd

    def reparameterization(self, z_mean, z_logstd):
        epsilon = torch.randn_like(z_mean).to(self.device)
        z = z_mean + torch.exp(z_logstd) * epsilon
        return z

    def decode(self, z):
        """For exact posterior by reconstruction-based likelihood"""
        _z = self.decoder(z)
        x_mean = self.fc_x_mean(_z)
        x_logstd = self.fc_x_logstd(torch.ones(1, 1).to(self.device))
        x_logstd = torch.clamp(x_logstd, min=-2, max=5)
        return x_mean, x_logstd

    def forward(self, x):
        z_mean, z_logstd = self.encode(x)
        z = self.reparameterization(z_mean, z_logstd)
        x_hat, x_logstd = self.decode(z)
        return x_hat, x_logstd, z, z_mean, z_logstd


def loss_VAE_Gaussian(result, target, fix_x_var=False, beta=1.0, 
    prior_mu=0.0, prior_sigma=1.0):
    """
    VAE loss with Gaussian noise model
    Note, this will be in a batch format
    Eq. 10 & Appendix B in VAE paper: https://arxiv.org/abs/1312.6114

    Issue 1: lr
    -----------
    By using torch.mean(torch.sum(..., dim=-1)) instead of torch.mean(...), 
    the gradient is x_dim (or z_dim) times larger, so it requires to use 
    smaller learning rate, e.g., lr=1e-3

    Issue 2: beta
    -------------
    The prior might be too strong, depending on x_dim and z_dim. It may help
    to use a smaller beta, e.g., 1e-2, or set a broader prior, e.g., 
    prior_sigma = 3.0

    Issue 3: x_logstd
    -----------------
    The variance of x for defining likelihood. See the three (+1) options in
    the VAE model. We can fix this variance by setting fix_x_var=True
    """
    # TODO: somehow when beta>1e-3, the model doesn't converge well.
    def gaussian_loglik(x, x_hat, x_logstd):
        _loglik = (-0.5 * torch.square((x_hat - x) / torch.exp(x_logstd)) - 
                   x_logstd * math.log(math.sqrt(2 * math.pi)))
        return torch.mean(torch.sum(_loglik, dim=-1))

    def kl_divergence(z_mean, z_logstd, prior_mu=0.0, prior_sigma=1.0):
        """
        Analytical KL divergence for Gaussian:
        https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
        """
        mu1, var1 = z_mean, torch.square(torch.exp(z_logstd))
        mu2, var2 = prior_mu, prior_sigma**2
        _kl = (torch.square(mu1 - mu2) / (2 * var2) + 
               0.5 * (var1 / var2 - 1 - 2 * z_logstd + math.log(var2)))
        return torch.mean(torch.sum(_kl, dim=-1))

    x_hat, x_logstd, z, z_mean, z_logstd = result
    if fix_x_var:
        x_logstd = x_logstd * 0 #torch.zeros(1, 1)#.to(self.device)

    # Calculate loss
    loglik = gaussian_loglik(target, x_hat, x_logstd)
    kl = kl_divergence(z_mean, z_logstd, prior_mu, prior_sigma)
    return -(loglik - beta * kl)
