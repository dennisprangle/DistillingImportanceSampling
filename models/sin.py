import torch
from torch.distributions import MultivariateNormal
import numpy as np
from models.models import Model
from models.weighted_sample import InitialFinalWeightedSample

class SinModel(Model):
    """A simple sinusoidal model"""
    def __init__(self):
        self.max_eps = 1.
        self.initial_target = MultivariateNormal(torch.zeros(2), 4*torch.eye(2))
        ## i.e. two independent N(0,2^2) distributions

    def log_initial_target(self, inputs):
        """Calculate (unnormalised) log initial target

        This is a bivariate N(0,4I) density"""
        return self.initial_target.log_prob(inputs)

    def log_final_target(self, inputs):
        """Calculate (unnormalised) log final target"""
        th1 = inputs[:,0]
        th2 = inputs[:,1]
        n = th1.shape[0]
        p = torch.where(torch.abs(th1) < np.pi,
                        torch.ones(n), torch.zeros(n))
        p = torch.log(p)
        p += -100. * torch.pow((th2 - torch.sin(th1)), 2)
        return p

    def run(self, inputs, log_proposal):
        return InitialFinalWeightedSample(
            inputs,
            log_proposal,
            self.log_initial_target(inputs),
            self.log_final_target(inputs)
        )
