
import math
import torch
import torch.nn as nn


def gaussian_loglik(x, x_hat, x_logstd):
    """Gaussian loglikelihood"""
    _loglik = (-0.5 * torch.square((x_hat - x) / torch.exp(x_logstd)) - 
                x_logstd * math.log(math.sqrt(2 * math.pi)))
    return torch.mean(torch.sum(_loglik, dim=-1))

def gaussian_kl(z_mean, z_logstd, prior_mu=0.0, prior_sigma=1.0):
    """Analytical KL divergence for Gaussian:
    https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
    
    
    KL divergence of univariate normal distribution
    Not implemented yet, requiring Monte Carlo or heuristic approximation
    https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
    """
    mu1, var1 = z_mean, torch.square(torch.exp(z_logstd))
    mu2, var2 = prior_mu, prior_sigma**2
    _kl = (torch.square(mu1 - mu2) / (2 * var2) + 
            0.5 * (var1 / var2 - 1 - 2 * z_logstd + math.log(var2)))
    return torch.mean(torch.sum(_kl, dim=-1))

def logitnormal_kl(z_mean, z_logstd, prior_mu=0.0, prior_sigma=1.0):
    """KL divergence of logit-normal distribution
    Not implemented yet, requiring Monte Carlo or heuristic approximation
    https://en.wikipedia.org/wiki/Logit-normal_distribution#Moments
    """
    #TODO: to implement; using gaussian_kl for approximate for now
    return gaussian_kl(z_mean, z_logstd, prior_mu, prior_sigma)

def negabin_loglik(x, x_hat, log_phi):
    """Negative binomial loglikelihood

    Parameters
    ----------
    x: observed n_failure
    x_hat: expected n_failure
    log_phi: log scale of over dispersion; 
    n: n_success, denoting concertration & phi = 1/n_success
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
