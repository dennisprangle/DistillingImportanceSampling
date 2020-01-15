import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from itertools import chain

class SDE():
    """Distribution based on Euler-Maruyama discretisation of a SDE

    `x0` is the initial state, a tensor of shape [ncomps]
    `ncomps` is number of state components
    `npars` is length of theta
    `T` is the number of time steps
    `dt` is the discretisation time step
    `theta_dist` is a distribution for the parameters

    I use duck typing rather than extend tensorflow probability's `Distribution`
    class. This avoids having to implement edge cases not used in my algorithm.
    """
    def __init__(self, x0, ncomps, npars, T, dt, theta_dist):
        self.x0 = x0
        self.ncomps = ncomps
        self.npars = npars
        self.T = T
        self.dt = dt
        self.theta_dist = theta_dist
        self.root_dt = np.sqrt(dt)
        self.standard_normal = tfd.Normal(0, 1)

    def _get_coefs(self, x, theta, time_index):
        """Output drift and root diffusion coefficients

        Should be implemented by subclasses.

        `x` is a tensor of observations of shape [..., self.ncomps]
        `theta` is a tensor of parameters of shape [..., self.npars]
        `time_index` is a tensor of times of shape [...]
        The leading dimensions of the inputs should match.

        The output is drift, which has the same shape as x,
        and rootdiff, which has shape [..., self.ncomps, self.ncomps].
        The last two dimensions of rootdiff form a lower triangular matrix.
        """
        raise NotImplementedError

    def _sample_theta(self, nreps):
        """Sample `nreps` replications of parameters

        Returns a tensor of shape `[nreps, self.npars]`
        """
        return self.theta_dist.sample(nreps)


    def _sample_x(self, theta):
        """Sample discretised SDE paths given parameters

        `theta` should be a tensor of shape `[nreps, self.npars]`

        Returns a tensor of shape `[nreps, self.T, self.ncomps]`
        """
        nreps = theta.shape[0]
        z = tf.random_normal(shape=(nreps, self.T, self.ncomps))
        x0_tiled = tf.tile(tf.expand_dims(self.x0, axis=0), [nreps, 1])
        time_indices = tf.fill([nreps], 0)
        drift0, rootdiff0 = self._get_coefs(x0_tiled, theta, time_indices)
        xcurr = self.x0 + self.dt * drift0
        xcurr += self.root_dt * tf.linalg.matvec(rootdiff0, z[:,0,:])
        xlist = [xcurr]
        for t in range(1, self.T):
            time_indices = tf.fill([nreps], t)
            drift, rootdiff = self._get_coefs(xcurr, theta, time_indices)
            xcurr += self.dt * drift
            xcurr += self.root_dt * tf.linalg.matvec(rootdiff, z[:,t,:])
            xlist += [xcurr]
        return tf.stack(xlist, axis=1)

    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        There are `nreps` sampled rows.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        theta = self._sample_theta(nreps)
        x = self._sample_x(theta)
        x = tf.reshape(x, [nreps, -1]) # Flatten each path
        return tf.concat([theta, x], axis=1)

    def _log_prob_theta(self, theta):
        """Calculate log density for each row of `theta` matrix
        """
        return self.theta_dist.log_prob(theta)

    def _get_z(self, x, theta):
        """Return base random variables given SDE paths and parameters

        `x` is a tensor of shape `[nreps, self.T, self.ncomps]`
        `theta` is a tensor of shape `[nreps, self.npars]`

         The output is a tensor of the same shape as `x` containing the
         standard normal values which would simulate these paths.
        """
        nreps = x.shape[0]
        x0_tiled = tf.reshape(self.x0, [1,1,self.ncomps])
        x0_tiled = tf.tile(x0_tiled, [nreps, 1, 1])
        theta_tiled = tf.tile(
            tf.reshape(theta, [nreps, 1, self.npars]),
            [1, self.T, 1]
        )
        oldx = tf.concat((x0_tiled, x[:,:-1,:]), axis=1)
        time_indices = tf.reshape(range(self.T), [1, self.T])
        time_indices = tf.tile(time_indices, [nreps, 1])
        drift, rootdiff = self._get_coefs(oldx, theta_tiled, time_indices)
        xdiff = x - oldx
        z = tf.linalg.triangular_solve(rootdiff,
                                       tf.expand_dims(xdiff - self.dt*drift, 3))
        z /= self.root_dt
        z = tf.squeeze(z, axis=3)
        return z

    def _log_det_jacobian(self, z, theta):
        """Returns log determinant of Jacobian for forward transformation from
        standard normal samples `z` and parameters `theta` to paths
        """
        nreps = z.shape[0]
        x0_tiled = tf.tile(tf.expand_dims(self.x0, axis=0), [nreps, 1])
        time_indices = tf.fill([nreps], 0)
        drift0, rootdiff0 = self._get_coefs(x0_tiled, theta, time_indices)
        xcurr = self.x0 + self.dt * drift0
        xcurr += self.root_dt * tf.linalg.matvec(rootdiff0, z[:,0,:])
        # Now get trace of logged root diffusion matrix
        out = tf.reduce_sum(tf.log(tf.matrix_diag_part(rootdiff0)), [-1])
        for t in range(1, self.T):
            time_indices = tf.fill([nreps], t)
            drift, rootdiff = self._get_coefs(xcurr, theta, time_indices)
            xcurr += self.dt * drift
            xcurr += self.root_dt * tf.linalg.matvec(rootdiff, z[:,t,:])
            # Now add traces of logged root diffusion matrices times root dt
            out += tf.reduce_sum(tf.log(self.root_dt *
                                        tf.matrix_diag_part(rootdiff)), [-1])
        return out

    def _log_prob_x(self, x, theta):
        """Return log density of paths conditional on parameters

        `x` is a tensor of shape `[nreps, self.T, self.ncomps]`
        `theta` is a tensor of shape `[nreps, self.npars]`

        The output is a vector of length `nreps`.
        """
        # Both the following lines of code calculate drifts and diffusions
        # This is not inefficient as only the 2nd line creates correctly
        # differentiable versions.
        z = self._get_z(x, theta)
        base_log_prob = tf.reduce_sum(self.standard_normal.log_prob(z), [1,2])
        ldj = self._log_det_jacobian(z, theta)
        return base_log_prob - ldj

    def log_prob(self, data):
        """Return log density of parameters + paths

        `data` is matrix whose first columns are parameters.
        The remaining columns are flattened paths.
        Each row represents a different parameter+path.

        A vector of log probabilities for each row is returned.
        """
        nreps = data.shape[0]
        theta = data[:, 0:self.npars]
        x = data[:, self.npars:]
        x = tf.reshape(x, [nreps, self.T, self.ncomps])
        return self._log_prob_theta(theta) + self._log_prob_x(x, theta)


class LorenzSDE(SDE):
    """Lorenz 63 SDE

    A discretised SDE whose drift and diffusion terms follow
    the Lorenz 63 system

    `x0` is the initial state, a tensor of shape [2]
    `T` is the number of states
    `dt` is the discretisation timestep
    `theta_dist` is a distribution for the parameters
    `cap_x` is maximum allowed magnitude during simulation
    `replace_rejected_samples` If True `sample` uses rejection sampling to give the requested
    number of samples. If False rejected samples are simply not returned.
    """

    def __init__(self, x0, T, dt, theta_dist, cap_x=1E3, replace_rejected_samples=False):
        self.diff_scale = np.sqrt(10.) # Diffusion scale (hard-coded for now)
        self.cap_x = cap_x
        self.replace_rejected_samples = replace_rejected_samples
        super().__init__(x0=x0, ncomps=3, npars=4, T=T, dt=dt,
                         theta_dist=theta_dist)

    def _get_coefs(self, x, theta, time_index):
        """Output Lorenz 63 drift and root diffusion

        `x` is a tensor of observations of shape [..., 3]
        `theta` is a tensor of parameters of shape [..., 4]
        `time_index` is a tensor of times of shape [...]
        The leading dimensions of the inputs should match.

        The output is `drift`, which has the same shape as `x`,
        and `rootdiff`, which has shape [..., 3, 3].
        The last two dimensions of `rootdiff` form a lower triangular matrix.
        """
        x1, x2, x3 = tf.split(x, [1,1,1], -1)
        th1, th2, th3, sig = tf.split(theta, [1,1,1,1], -1)
        drift1 = th1 * (x2 - x1)
        drift2 = th2 * x1 - x2 - x1 * x3
        drift3 = x1 * x2 - th3 * x3
        drift = tf.concat((drift1, drift2, drift3), axis=-1)
        #drift = tf.where(tf.abs(x) < self.cap_x, drift, -x)
        tempshape = x.shape.as_list()[:-1]
        rootdiff = self.diff_scale * tf.eye(3, batch_shape=tempshape)
        return drift, rootdiff
    
    def _log_det_jacobian(self, z, theta):
        """Returns log determinant of Jacobian for forward transformation from
        standard normal samples `z` and parameters `theta` to paths

        For this model the Jacobian is constant, so this can be done very quickly
        """
        nreps = z.shape[0]
        # sum of log root diffusion diagonal values over 1:T and 1:ncomps
        ldj = self.T * 3. * np.log(self.root_dt * self.diff_scale)
        return ldj * tf.ones(nreps)

    def _sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE
        Samples with any x values above `self.cap_x` in magnitude are not returned

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        theta = self._sample_theta(nreps)
        x = self._sample_x(theta)
        x = tf.reshape(x, [nreps, -1]) # Flatten each path
        npx = x.numpy()
        max_abs_x = np.amax(np.abs(npx), axis=1) # Max abs value in each path
        keep = np.where(max_abs_x < self.cap_x)[0]
        x = tf.gather(x, keep, axis=0)
        theta = tf.gather(theta, keep, axis=0)
        return tf.concat([theta, x], axis=1)
    
    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE
        If `self.replace_rejected_samples` is False, x values above `self.cap_x` in magnitude
        are not returned. Otherwise they are replaced with fresh samples until there are `nreps`
        samples/

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        nleft = nreps
        out = self._sample(nreps)
        if self.replace_rejected_samples:
            nleft = nreps - out.shape[0]
        else:
            nleft = 0
            
        while nleft > 0:
            out = tf.concat([out, self._sample(nleft)], axis=0)
            nleft = nreps - out.shape[0]

        return out    


class NeuralLorenzSDE(LorenzSDE):
    """Lorenz SDE with learnable drift and diffusion modifications
    and learnable parameter distribution

    `x0` is the initial state, a tensor of shape [3]
    `T` is the number of states
    `dt` is the discretisation time step
    `obs_indices` is array whose entries give indices of observed times
    `obs_data` is observed data, shape `len(obs_indices), 3`
    `hidden_size_x` is list of hidden layers widths for drift
    `hidden_size_theta` is a tuple of hidden layers in each flow layer for theta
    `nlayers_theta` is how many hidden flow layers to use for theta
    """

    def __init__(self, x0, T, dt,
                 obs_indices, obs_data,
                 hidden_size_x, hidden_size_theta, nlayers_theta):
        self.diff_mult_const = np.sqrt(10.) / np.log(2.) # A repeatedly used constant
        ## Create lookup tables for features
        time_till_lookup = np.zeros(T, dtype=np.float32)
        obs_data_lookup = np.zeros((T,3), dtype=np.float32)
        obs_counter = 0
        next_time_index = obs_indices[0]
        time_till_next = obs_indices[0]
        next_obs = obs_data[0,:]
        past_last_obs = False
        for t in range(T):
            if t == next_time_index:
                obs_counter += 1
                if obs_counter < len(obs_indices):
                    next_time_index = obs_indices[obs_counter]
                    time_till_next = obs_indices[obs_counter] - t
                    next_obs = obs_data[obs_counter,:]
                else:
                    past_last_obs = True
            time_till_lookup[t] = time_till_next
            obs_data_lookup[t,:] = next_obs
            if past_last_obs == False:
                time_till_next -= 1
        self.time_till_lookup = tf.constant(time_till_lookup)
        self.obs_data_lookup = tf.constant(obs_data_lookup)

        ## Initialise neural network for drift
        self.nn_x = self._nn_template(hidden_size_x, 3)
        # and ensure variables initialised, at correct dimension
        self.nn_x(tf.zeros(shape=(2, 12)))

        ## Initialise flow for parameters
        bichain=list(chain.from_iterable([
            tfb.Permute(np.random.permutation(4)), # random permutation
            tfb.RealNVP(2, shift_and_log_scale_fn =
            tfb.real_nvp_default_template(hidden_size_theta,
                activation=tf.nn.elu,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                kernel_regularizer=tf.keras.regularizers.l1(1E-2)
            ))
        ] for _ in range(nlayers_theta)))
        bichain = bichain[1:] # remove final permutation
        # Transform so that N(0,1) mapped to Exp(0.1)
        norm2exp = [tfb.AffineScalar(scale=-10.), tfb.Invert(tfb.Exp()),
                    tfb.NormalCDF()]
        # Transform so that N(0,1) mapped to Exp(0.1) truncated to [0,50]
        # lower_end = np.exp(-5.)
        # norm2exp = [tfb.AffineScalar(scale=-10.), tfb.Invert(tfb.Exp()),
        #             tfb.AffineScalar(shift=np.float32(lower_end), scale=np.float32(1.-lower_end)),
        #             tfb.NormalCDF()]
        bijector = tfb.Chain(norm2exp + bichain)
        base = tfd.MultivariateNormalDiag(loc=tf.zeros(4))
        theta_dist = tfd.TransformedDistribution(base, bijector)
        theta_dist.sample() # Initialises variables

        super().__init__(x0=x0, T=T, dt=dt, theta_dist=theta_dist)

    def _nn_template(self, hidden_size, ncomps):
        """Create neural network for drift modification

        Uses approach of tensorflow probability normalising flow code to ensure
        variables are stored and added to `trainable_variables`
        """
        def _fn(inputs):
            first = True
            for n in hidden_size:
                if first:
                    h = tf.layers.dense(
                        inputs=inputs,
                        units=n,
                        activation=tf.nn.elu,
                        kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                        kernel_regularizer=tf.keras.regularizers.l1(1E-2)
                    )
                    first = False
                else:
                    h = tf.layers.dense(
                        inputs=h,
                        units=n,
                        activation=tf.nn.elu,
                        kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                        kernel_regularizer=tf.keras.regularizers.l1(1E-2)
                    )

            outputs = tf.layers.dense(
                inputs=h,
                units=ncomps+1,
                kernel_initializer=tf.initializers.truncated_normal(stddev=0.001),
                kernel_regularizer=tf.keras.regularizers.l1(1E-2)
            )
            return outputs

        return tf.make_template("drift_network", _fn)

    def _get_coefs(self, x, theta, time_index):
        """Estimate conditioned drift and root diffusion coefficients

        `x` is a tensor of observations of shape [..., self.ncomps]
        `theta` is a tensor of parameters of shape [..., self.npars]
        `time_index` is a tensor of times of shape [...]
        The leading dimensions of the inputs should match.

        The output is drift, which has the same shape as x,
        and rootdiff, which has shape [..., self.ncomps, self.ncomps].
        The last two dimensions of rootdiff form a lower triangular matrix.

        The drift and diffusion are taken from `baseSDE`.
        Then a modification is added to the drift,
        taken from a trainable neural network.
        """
        x_mat = tf.reshape(x, [-1, self.ncomps])
        theta_mat = tf.reshape(theta, [-1, self.npars])
        time_indices = tf.reshape(time_index, [-1])
        time_till_next_obs = tf.gather(self.time_till_lookup, time_indices)
        time_till_next_obs = tf.reshape(time_till_next_obs, [-1, 1])
        next_obs = tf.gather(self.obs_data_lookup, time_indices)
        times = self.dt * tf.cast(time_indices, 'float32')
        times = tf.reshape(times, [-1, 1])
        inputs = tf.concat((x_mat, theta_mat, time_till_next_obs, next_obs,
                            times), axis=1)
        outputs = self.nn_x(inputs)
        drift_modification = tf.reshape(outputs[:, 1:4], x.shape)
        
        drift, _ = super()._get_coefs(x, theta, time_index)
        drift += drift_modification
        diffshape = x.shape.as_list()[:-1]
        diff_mult = tf.reshape(outputs[:,0], diffshape + [1,1])
        diff_mult = tf.math.softplus(diff_mult) * self.diff_mult_const
        rootdiff = diff_mult * tf.eye(3, batch_shape=diffshape)        
        return drift, rootdiff

    def _log_det_jacobian(self, z, theta):
        ## Over-ride LorenzSDE method (which assumed fixed diffusion term)
        ## and use more general method of SDE
        return SDE._log_det_jacobian(self, z, theta)
