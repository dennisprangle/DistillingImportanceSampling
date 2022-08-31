import torch
from torch.distributions.uniform import Uniform
from torch.distributions.independent import Independent
import numpy as np
from models.models import SimulatorModel

class SinModel(SimulatorModel):
    """A simple sinusoidal model"""
    def __init__(self, observations=torch.zeros([1,1])):
        self.prior = Uniform(
            low = torch.tensor([-np.pi, -np.pi]),
            high = torch.tensor([np.pi, np.pi]),
            validate_args=False
        )
        self.prior = Independent(self.prior, 1)
        ## i.e. two independent uniform distributions on [-pi, pi]
        super().__init__(observations, observations_are_data=True)

    def log_prior(self, inputs):
        """Calculate vector of log prior densities of `inputs`

        For this model they are two independent uniform distributions on [-pi, pi]"""
        supported = torch.all(
            torch.abs(inputs) <= np.pi,
            dim=1
        )
        return torch.log(supported)

    def simulator(self, inputs):
        """Perform simulations"""
        th1 = inputs[:,0]
        th2 = inputs[:,1]
        x = th2 - torch.sin(th1)
        return x.reshape([-1,1])