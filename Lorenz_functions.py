from SDE import LorenzSDE
import numpy as np
import tensorflow as tf

class Lorenz_model:
    """Class to encapsulate Lorenz model"""
    def __init__(self, x0, T, dt, obs_indices, obs_data, prior, initial_target, obs_scale=None):
        """
        `x0` is initial state
        `T` is the number of time steps
        `dt` is the discretisation time step
        `obs_indices` is a numpy array of state indices for observation times
        `obs_data` is numpy array of observed data, of length `len(obs_indices)`
        `prior` is the prior distribution of parameters
        `initial_target` is used in early stages of tuning
        `obs_scale` is a fixed observation scale - leave as None for scale to be inferred
        """
        self.T = T
        self.obs_indices = obs_indices
        self.nobs = len(obs_indices)
        self.obs_data = obs_data
        self.prior = prior
        self.initial_target = initial_target
        self.obs_scale = obs_scale
        self.max_eps = 1.
        self.lz_dist = LorenzSDE(x0, T, dt, prior)

    def likelihood_prelims(self, inputs):
        """Store preliminary calculations required for evaluating likelihood"""
        self.initial_log_density = self.initial_target.log_prob(inputs)
        lz_log_density = self.lz_dist.log_prob(inputs)
        if self.obs_scale is None:
            obs_scale = inputs[:,3]
            obs_var = tf.pow(obs_scale, 2.)
        else:
            obs_scale = self.obs_scale
            obs_var = obs_scale ** 2.
        paths = tf.reshape(inputs[:, 4:], (-1, self.T, 3))
        sim_data = tf.gather(paths, self.obs_indices, axis=1)
        sq_diff = tf.reduce_sum((tf.pow(self.obs_data - sim_data, 2.0)),
                                axis=(1,2))
        obs_log_density = -3. * self.nobs * tf.log(obs_scale) - sq_diff / (2. * obs_var)
        obs_log_density = tf.where(obs_scale > 0.,
                                   obs_log_density,
                                   tf.fill(obs_log_density.shape, -np.Inf))
        self.log_unnorm_posterior = lz_log_density + obs_log_density

    def log_tempered_target(self, eps):
        """Calculate log of unnormalised tempered target density

        Requires likelihood_prelim to have already been run.
        """
        if eps == 0.:
            # Avoid NaNs from zero times negative infinity
            return self.log_unnorm_posterior
        else:
            return eps * self.initial_log_density + (1. - eps) * self.log_unnorm_posterior
