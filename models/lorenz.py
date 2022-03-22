import numpy as np
import torch
from torch.distributions import Exponential, Independent
from models.models import LikelihoodModel
from models.sde import LorenzSDE

class LorenzModel(LikelihoodModel):
    """Class to encapsulate Lorenz model"""
    def __init__(self, x0, T, dt, obs_indices, obs_data, obs_scale=None):
        """
        `x0` is initial state
        `T` is the number of time steps
        `dt` is the discretisation time step
        `obs_indices` is a numpy array of state indices for observation times
        `obs_data` is tensor of observed data, shape `[len(obs_indices)]`
        `obs_scale` is a fixed observation scale. Leave as `None` for scale to be inferred.
        """
        self.T = T
        self.obs_indices = obs_indices
        self.nobs = len(obs_indices)
        self.obs_data = obs_data
        if obs_scale is None or torch.is_tensor(obs_scale):
            self.obs_scale = obs_scale
        else:
            self.obs_scale = torch.tensor(obs_scale)
        self.max_eps = 1.
        self.prior = Independent(
            Exponential(rate=torch.full([4], 0.1)),
            1
        )
        self.lorenz_dist = LorenzSDE(x0, T, dt, self.prior, replace_rejected_samples=True)

    def log_prior(self, inputs): # Log density of prior + model
        return self.lorenz_dist.log_prob(inputs)

    def log_likelihood(self, inputs): # Log density of observations
        if self.obs_scale is None:
            obs_scale = inputs[:,3]
        else:
            obs_scale = self.obs_scale

        obs_var = torch.square(obs_scale)
        paths = inputs[:, 4:].reshape([-1, self.T, 3])
        sim_data = paths[:, self.obs_indices, :]
        sq_diff = torch.square(self.obs_data - sim_data).sum(dim=(1,2))
        obs_log_density = -3. * self.nobs * torch.log(obs_scale) - \
                          0.5 * sq_diff / obs_var
        obs_log_density = torch.where(
            obs_scale > 0.,
            obs_log_density,
            torch.full_like(obs_log_density, -np.inf)
        )
        log_likelihood = obs_log_density
        return log_likelihood
