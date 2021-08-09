import numpy as np
from itertools import chain
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tf.enable_eager_execution()
import matplotlib.pyplot as plt
plt.ion()

class RickerModel:
    """Class to encapsulate Ricker model"""
    def __init__(self, x0, T, obs, prior):
        """
        `x0` is initial state
        `T` is the number of observations
        `obs` is numpy array of observed data, of length `T`
        `prior` is the prior distribution of parameters
        """
        self.T = T
        obs = tf.constant(obs, dtype='float32')
        self.obs = tf.expand_dims(obs, axis=0)
        self.prior = prior
        self.max_eps = 1.
        self.ricker_dist = RickerDistribution(x0, T, prior)

    def likelihood_prelims(self, inputs):
        """Store preliminary calculations required for evaluating likelihood"""
        self.ricker_log_density = self.ricker_dist.log_prob(inputs)
        paths = inputs[:, 2:] * 1000.
        ##print("1st path", paths[0,:])
        self.any_inf = tf.reduce_any(tf.is_inf(paths), 1)
        ## Next line is log Poisson prob, neglecting a constant term
        self.obs_log_prob = self.obs * tf.log(paths) - paths
        self.obs_log_prob = tf.reduce_sum(self.obs_log_prob, axis=1)
        self.neg_inf_vec = -np.inf * tf.ones_like(self.obs_log_prob)

    def log_tempered_target(self, eps):
        """Calculate log of unnormalised tempered target density

        Requires likelihood_prelim to have already been run.
        """
        log_prob = self.ricker_log_density
        if eps < 1.: # Avoids 0*inf difficulties when eps=1
            log_prob += (1. - eps) * self.obs_log_prob
        log_prob = tf.where(self.any_inf, self.neg_inf_vec, log_prob)
        return log_prob


class RickerDistribution():
    """Class to encapsulate Ricker model distribution

    `x0` is the initial population size
    `T` is the number of time steps
    `theta_dist` is a distribution for the parameters
    """
    def __init__(self, x0, T, theta_dist):
        self.x0 = x0
        self.T = T
        self.theta_dist = theta_dist
        self.standard_normal = tfd.Normal(0, 1)


    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + states

        Returns a tensor of shape `[nreps, 2 + self.T]`.
        Each row is a sample.
        There are `nreps` sampled rows.
        The first 2 columns are the parameters.
        The remaining columns are states.
        """
        theta = self.theta_dist.sample(nreps)
        b0 = theta[:,0]
        b1 = theta[:,1]
        x = tf.zeros(shape=(nreps, self.T))
        xt = tf.fill([nreps], self.x0)
        log_xt = tf.log(xt)
        xlist = [xt]
        z = tf.random_normal(shape=(nreps, self.T-1))
        for t in range(1, self.T):
            log_xt += b0 + b1*xt + z[:,t-1]
            xt = tf.exp(log_xt)
            xlist += [xt]
        x = tf.stack(xlist, axis=1)
        return tf.concat([theta, x], axis=1)

    def _get_z(self, x, theta):
        """Return base random variables given parameters and state

        `x` is a tensor of shape `[nreps, self.T]`
        `theta` is a tensor of shape `[nreps, 2]`

         The output is a tensor of shape `[nreps, self.T-1]` containing the
         standard normal values which would simulate these states.
        """
        nreps = x.shape[0]
        b0 = theta[:,0]
        b1 = theta[:,1]
        z = tf.zeros(shape=(nreps, self.T-1))
        xt = x[:,0]
        zlist = []
        for t in range(1, self.T):
            xt_prev = xt
            log_xt_no_noise = tf.log(xt_prev) + b0 + b1*xt_prev
            xt = x[:,t]
            zt = tf.log(xt) - log_xt_no_noise
            zlist += [zt]
        return z

    def _log_prob_x(self, x, theta):
        """Return log density of paths conditional on parameters

        `x` is a tensor of shape `[nreps, self.T, self.ncomps]`
        `theta` is a tensor of shape `[nreps, self.npars]`

        The output is a vector of length `nreps`.

        Technically this outputs the log density of the *log* paths.
        Getting the density of the raw paths would involve another Jacobian term.
        But I omit this as these terms would simply cancel in the ELBO.
        """
        z = self._get_z(x, theta)
        log_prob = tf.reduce_sum(self.standard_normal.log_prob(z), 1)
        ## TO DO: MAKE -INFTY IF X INVALID? (NEGATIVE, INFINITE OR NAN)
        ## ESSENTIALLY REJECTION SAMPLING TO REMOVE "EXTREME" PATHS
        return log_prob

    def log_prob(self, data):
        """Return log density of parameters + paths

        `data` is matrix whose first columns are parameters.
        The remaining columns are flattened paths.
        Each row represents a different parameter+path.

        A vector of log probabilities for each row is returned.
        """
        nreps = data.shape[0]
        theta = data[:, 0:2]
        x = data[:, 2:]

        return self.theta_dist.log_prob(theta) + self._log_prob_x(x, theta)


class RickerApprox():
    """Class to encapsulate variational approximation to Ricker model

    `x0` is the initial population size
    `T` is the number of time steps
    `theta_dist` is a distribution for the parameters
    `obs` is observed data
    `hidden_size_x` is list of hidden layer widths for noise modification
    `receptive_field` how many future observations to consider
    """
    def __init__(self, x0, T, obs, hidden_size_x, hidden_size_theta,
                 receptive_field=3, flow_layers=8):
        self.x0 = x0
        self.T = T
        self.standard_normal = tfd.Normal(0, 1)        
        self.obs = tf.constant(obs, dtype='float32')
        self.receptive_field = receptive_field
        if receptive_field > 1:
            end_pad = tf.fill([receptive_field], -1.)
            self.padded_obs = tf.concat([obs, end_pad], axis=0)
        else:
            self.padded_obs = obs

        ## Initialise neural network for noise modification
        self.nn_x = self._nn_template(hidden_size_x)
        ## Ensure variables initialised at correct dimension
        self.nn_x(tf.zeros(shape=(2, 2+receptive_field)))

        ## Initialise flow for parameters
        bichain=list(chain.from_iterable([
            tfb.Permute([1,0]), # swap order
            tfb.RealNVP(1, shift_and_log_scale_fn =
            tfb.real_nvp_default_template(hidden_size_theta,
                activation=tf.nn.elu,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                kernel_regularizer=tf.keras.regularizers.l1(1E-2)
            ))
        ] for _ in range(flow_layers)))
        bichain = bichain[1:] # remove final permutation
        bijector = tfb.Chain(bichain)
        base = tfd.MultivariateNormalDiag(loc=tf.zeros(2))
        self.theta_dist = tfd.TransformedDistribution(base, bijector)
        self.theta_dist.sample() # Initialises variables


    def _nn_template(self, hidden_size):
        """Create neural network for noise modification

        Uses approach of tensorflow probability normalising flow code to ensure
        variables are stored and added to `trainable_variables`
        """
        def _fn(inputs):
            h = tf.layers.batch_normalization(inputs)
            for n in hidden_size:
                h = tf.layers.dense(
                    inputs=h,
                    units=n,
                    activation=tf.nn.elu,
                    kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                    kernel_regularizer=tf.keras.regularizers.l1(1E-2)
                )

            outputs = tf.layers.dense(
                inputs=h,
                units=2,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                kernel_regularizer=tf.keras.regularizers.l1(1E-2)
            )
            return outputs

        return tf.make_template("ricker_network", _fn)


    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + states

        Returns a tensor of shape `[nreps, 2 + self.T]`.
        Each row is a sample.
        There are `nreps` sampled rows.
        The first 2 columns are the parameters.
        The remaining columns are states.
        """
        theta = self.theta_dist.sample(nreps)
        b0 = theta[:,0]
        b1 = theta[:,1]
        ##print("b0", b0.numpy()[0:10])
        ##print("b1", b1.numpy()[0:10])
        xt = tf.fill([nreps], self.x0)
        log_xt = tf.log(xt)
        ##print("t", 0)
        ##print("log_xt", log_xt.numpy()[0:10])
        ##print("xt", xt.numpy()[0:10])
        xlist = [xt]
        z = tf.random_normal(shape=(nreps, self.T-1))
        for t in range(1, self.T):
            ##print("t", t)
            xt_prev = xt
            features = [tf.fill([nreps,1], t-1.)]
            features += [tf.expand_dims(log_xt, axis=1)]
            next_obs = self.padded_obs[t:t+self.receptive_field]
            features += [tf.broadcast_to(next_obs,
                                         [nreps, self.receptive_field])]
            features = tf.concat(features, axis=1)
            ##print("features", features)
            nn_out = self.nn_x(features)
            mu = nn_out[:,0]
            ##print("mu", mu.numpy()[0:10])
            sigma = tf.math.softplus(nn_out[:,1])
            ##print("sigma", sigma.numpy()[0:10])
            log_xt += b0 + b1*xt_prev + mu + sigma*z[:,t-1]
            ##print("log_xt", log_xt.numpy()[0:10])
            xt = tf.exp(log_xt)
            ##print("xt", xt.numpy()[0:10])
            xlist += [xt]
        x = tf.stack(xlist, axis=1)
        return tf.concat([theta, x], axis=1)


    def _get_z(self, x, theta):
        """Return base random variables given parameters and state

        `x` is a tensor of shape `[nreps, self.T]`
        `theta` is a tensor of shape `[nreps, 2]`

         The output is a tensor of shape `[nreps, self.T-1]` containing the
         standard normal values which would simulate these states.
        """
        nreps = x.shape[0]
        b0 = theta[:,0]
        b1 = theta[:,1]
        xt = x[:,0]
        log_xt = tf.log(xt)
        zlist = []        
        for t in range(1, self.T):
            features = [tf.fill([nreps,1], t-1.)]
            features += [tf.expand_dims(log_xt, axis=1)]
            next_obs = self.padded_obs[t:t+self.receptive_field]
            features += [tf.broadcast_to(next_obs,
                                         [nreps, self.receptive_field])]
            features = tf.concat(features, axis=1)
            nn_out = self.nn_x(features)
            mu = nn_out[:,0]
            sigma = tf.math.softplus(nn_out[:,1])
            log_xt_no_noise = log_xt + b0 + b1*xt - mu
            xt = x[:,t]
            log_xt = tf.log(xt)
            zt = (log_xt - log_xt_no_noise) / sigma
            zlist += [zt]
        return tf.stack(zlist, axis=1)


    def _log_det_jacobian(self, z, theta):
        """Returns log determinant of Jacobian for forward transformation from
        standard normal samples `z` and parameters `theta` to paths
        """
        nreps = z.shape[0]
        b0 = theta[:,0]
        b1 = theta[:,1]
        xt = tf.fill([nreps], self.x0)
        log_xt = tf.log(xt)
        ldj = tf.zeros(nreps) # Log determinant of Jacobian
        for t in range(1, self.T):
            features = [tf.fill([nreps,1], t-1.)]
            features += [tf.expand_dims(log_xt, axis=1)]
            next_obs = self.padded_obs[t:t+self.receptive_field]
            features += [tf.broadcast_to(next_obs,
                                         [nreps, self.receptive_field])]
            features = tf.concat(features, axis=1)
            nn_out = self.nn_x(features)
            mu = nn_out[:,0]
            sigma = tf.math.softplus(nn_out[:,1])
            log_xt += b0 + b1*xt + mu + sigma*z[:,t-1]
            xt = tf.exp(log_xt)
            ldj -= tf.log(sigma)
        return ldj

    def _log_prob_x(self, x, theta):
        """Return log density of paths conditional on parameters

        `x` is a tensor of shape `[nreps, self.T, self.ncomps]`
        `theta` is a tensor of shape `[nreps, self.npars]`

        The output is a vector of length `nreps`.

        Technically this outputs the log density of the *log* paths.
        Getting the density of the raw paths would involve a Jacobian term.
        But I omit this as these terms would simply cancel in the ELBO.
        """
        z = self._get_z(x, theta) # 1st pass to get z from x and theta
        log_prob = tf.reduce_sum(self.standard_normal.log_prob(z), 1)
        log_prob += self._log_det_jacobian(z, theta) # 2nd pass to get Jacobian
                                                     # term from z and theta
        ## n.b. We could get the Jacobian term from the first pass.
        ## But it wouldn't be correctly differentiable.
        ## (We want to assume our z is fixed, not x.)

        ## Can get nans or inf in log prob calculation because:
        ## 1) Original simulation overflowed
        ## 2) Variational parameters changed since simulation
        ## Set any nans or infs to infinite log prob (causes them to be ignored)
        lp = log_prob.numpy()
        lp_inf = np.isinf(lp)
        lp_nan = np.isnan(lp)
        lp_bad = np.logical_or(lp_inf, lp_nan)
        if any(lp_bad):
            log_prob = tf.where(lp_bad,
                                np.inf*tf.ones_like(log_prob),
                                log_prob)

        return log_prob

    def log_prob(self, data):
        """Return log density of parameters + paths

        `data` is matrix whose first columns are parameters.
        The remaining columns are flattened paths.
        Each row represents a different parameter+path.

        A vector of log probabilities for each row is returned.
        """
        nreps = data.shape[0]
        theta = data[:, 0:2]
        x = data[:, 2:]

        return self.theta_dist.log_prob(theta) + self._log_prob_x(x, theta)
