import torch

def logLikelihoodGaussian(x, mu, logvar, mask = None):
    '''
    logLikelihoodGaussian(x, mu, logvar, mask):
    
    Log-likeihood of a real-valued observation given a Gaussian distribution with mean 'mu' 
    and standard deviation 'exp(0.5*logvar), mask out artifacts'
    
    Arguments:
        - x (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - mu (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - logvar (torch.tensor or torch.Tensor): tensor scalar or Tensor of size batch-size x time-step x input dimensions
        - mask (torch.Tensor): Optional, Tensor of of size batch-size x time-step x input dimensions
    '''
    from math import log,pi
    eps = 1e-5
    if mask is not None:
        loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar)+eps))) * mask).mean()
    else:
        loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar)+eps)))).mean()
    return loglikelihood

def logLikelihoodPoisson(k, lam, mask = None):
    '''
    logLikelihoodPoisson(k, lam)
    Log-likelihood of Poisson distributed counts k given intensity lam.
    Arguments:
        - k (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - lam (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - mask (torch.Tensor): Optional, Tensor of of size batch-size x time-step x input dimensions
    '''
    if mask is not None:
        loglikelihood = ((k * torch.log(lam) - lam - torch.lgamma(k + 1)) * mask).mean()
    else:
        loglikelihood = (k * torch.log(lam) - lam - torch.lgamma(k + 1)).mean()
    return loglikelihood

def mse_loss(pred, gt, mask=None):
    if mask is not None:
        mse_loss = (((pred - gt)**2) * mask).mean()
    else:
        mse_loss = ((pred - gt)**2).mean()
    return mse_loss