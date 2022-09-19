# Some utility functions
import torch
import numpy as np

def resample(x, w, n):
    """Multinomial resampling

    Arguments:
    `x` A tensor to be resampled (along the first dimension)
    `w` Weights (will be normalised to sum to 1 if they do not already)
    `n` Required output size
    """
    i = torch.multinomial(w, n, replacement=True)
    return x[i,...]

def effective_sample_size(w, log_input=False):
    """Effective sample size of weights or log_weights

    `w` is a 1-dimensional tensor of weights or log weights (can be unnormalised in either case)
    `log_input` denotes whether `w` is log weights
    """
    if log_input:
        log_weights = w
        max_log_weight = torch.max(log_weights)
        if max_log_weight > -np.inf:
            weights = torch.exp(log_weights - max_log_weight)
        else:
            return torch.tensor(0.)
    else:
        weights = w

    sum_weights = weights.sum()
    if sum_weights == 0:
        return torch.tensor(0.)

    ess = (sum_weights ** 2.0) / (weights ** 2.0).sum()
    return ess

def norm_to_unif(x, a=0., b=1.):
    """Convert N(0,1) draws to U(a,b) draws"""
    ## TODO: avoid repeatedly initialising standard_normal, by just pasting cdf code here
    standard_normal = torch.distributions.Normal(0., 1.)
    y = standard_normal.cdf(x)
    y = a + (b-a)*y
    return y