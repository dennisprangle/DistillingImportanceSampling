import torch
import numpy as np
from torch.distributions import MultivariateNormal
from models.models import SimulatorModel
from utils import norm_to_unif

class SinModel(SimulatorModel):
    """A simple sinusoidal model"""
    def __init__(self, observations=torch.zeros([1,1])):
        self.standard_normal = torch.distributions.Normal(0., 1.)
        self.prior = MultivariateNormal(torch.zeros(2), torch.eye(2))
        super().__init__(observations, observations_are_data=True)

    def log_prior(self, inputs):
        """Calculate vector of log prior densities of `inputs`

        For this model they are independent N(0,1)"""
        return torch.sum(self.standard_normal.log_prob(inputs), dim=1)
    
    def simulator(self, inputs):
        """Perform simulations"""
        theta = norm_to_unif(inputs[:,0], -np.pi, np.pi)
        x = inputs[:,1]
        y = -torch.sin(theta) + x
        return y.reshape([-1,1])