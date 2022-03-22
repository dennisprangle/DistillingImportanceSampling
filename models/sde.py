import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.nn import Module
from torch.nn.functional import softplus, elu
from nflows import distributions, flows, nn
from nflows.transforms import RandomPermutation, CompositeTransform, PointwiseAffineTransform, Transform
from nflows_mods import AffineCouplingTransform, MLP

class SDE(Module):
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
        super().__init__()
        self.x0 = x0
        self.ncomps = ncomps
        self.npars = npars
        self.T = T
        self.dt = dt
        self.theta_dist = theta_dist
        self.root_dt = np.sqrt(dt)
        self.standard_normal = Normal(0., 1.)

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
        # Clamping to avoid numerical errors
        return self.theta_dist.sample(nreps).clamp(1e-6, 1e2)


    def _sample_x(self, theta):
        """Sample discretised SDE paths given parameters

        `theta` should be a tensor of shape `[nreps, self.npars]`

        Returns a tensor of shape `[nreps, self.T, self.ncomps]`
        """
        nreps = theta.shape[0]
        out_shape = (nreps, self.T, self.ncomps)
        z = torch.randn(out_shape)
        x0_tiled = torch.tile(self.x0, (nreps,1))
        time_indices = torch.zeros(nreps, dtype=int)
        drift0, rootdiff0 = self._get_coefs(x0_tiled, theta, time_indices)
        xcurr = self.x0 + self.dt * drift0
        xcurr = xcurr + self.root_dt * \
                torch.matmul(rootdiff0, z[:,0,:,None]).squeeze(-1)
        xlist = [xcurr]
        for t in range(1, self.T):
            time_indices = torch.full([nreps], t)
            drift, rootdiff = self._get_coefs(xcurr, theta, time_indices)
            xcurr = xcurr + self.dt * drift
            xcurr = xcurr + self.root_dt * \
                    torch.matmul(rootdiff, z[:,t,:,None]).squeeze(-1)
            xlist.append(xcurr)
        return torch.stack(xlist, dim=1)

    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        There are `nreps` sampled rows.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        theta = self._sample_theta(nreps)
        x = self._sample_x(theta).reshape([nreps, -1]) # Flattened paths
        return torch.cat([theta, x], axis=1)

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
        x0_tiled = torch.reshape(self.x0, [1,1,-1])
        x0_tiled = torch.tile(
            x0_tiled, 
            [nreps,1,1]
        )
        theta_tiled = torch.tile(
            theta.unsqueeze(1),
            [1, self.T, 1]
        )
        oldx = torch.cat([x0_tiled, x[:,:-1,:]], axis=1)
        time_indices = torch.tile(
            torch.arange(self.T),
            [nreps, 1]
        )
        drift, rootdiff = self._get_coefs(oldx, theta_tiled, time_indices)
        xdiff = x - oldx
        z, _ = torch.triangular_solve(
            (xdiff - self.dt*drift).unsqueeze(3),
            rootdiff
        )
        z = z / self.root_dt
        return z.squeeze(3)

    def _log_det_jacobian(self, z, theta):
        """Returns log determinant of Jacobian for forward transformation from
        standard normal samples `z` and parameters `theta` to paths
        """
        nreps = z.shape[0]
        x0_tiled = torch.reshape(self.x0, [1,-1])
        x0_tiled = torch.tile(
            x0_tiled, 
            [nreps,1]
        )
        time_indices = torch.zeros(nreps, dtype=int)
        drift0, rootdiff0 = self._get_coefs(x0_tiled, theta, time_indices)
        xcurr = self.x0 + self.dt * drift0 + self.root_dt * \
                torch.matmul(rootdiff0, z[:,0,:,None]).squeeze(-1)
        # Now get trace of logged root diffusion matrix
        out = torch.log(torch.diagonal(rootdiff0, dim1=1, dim2=2)).sum(-1)
        for t in range(1, self.T):
            time_indices = torch.full([nreps], t)
            drift, rootdiff = self._get_coefs(xcurr, theta, time_indices)
            xcurr = xcurr + self.dt * drift + self.root_dt * \
                    torch.matmul(rootdiff, z[:,t,:,None]).squeeze(-1)
            # Now add traces of logged root diffusion matrices times root dt
            out += torch.log(self.root_dt *
                             torch.diagonal(rootdiff, dim1=1, dim2=2)).sum(-1)
        return out

    def _log_prob_x(self, x, theta):
        """Return log density of paths conditional on parameters

        `x` is a tensor of shape `[nreps, self.T, self.ncomps]`
        `theta` is a tensor of shape `[nreps, self.npars]`

        The output is a vector of length `nreps`.
        """
        # Both _get_z and _log_det_jacobian calculate drifts & diffusions.
        # This is not inefficient as only the 2nd time creates correctly
        # differentiable versions.
        z = self._get_z(x, theta)
        base_log_prob = self.standard_normal.log_prob(z).sum([1,2])
        ldj = self._log_det_jacobian(z, theta)
        log_prob_x = base_log_prob - ldj
        x_max, _ = torch.abs(x[:,:,0]).max(dim=1)
        log_prob_x = torch.where(
            x_max == 1e6,
            torch.full_like(log_prob_x, -np.inf),
            log_prob_x
        )
        return log_prob_x

    def log_prob(self, data):
        """Return log density of parameters + paths

        `data` is matrix whose first columns are parameters.
        The remaining columns are flattened paths.
        Each row represents a different parameter+path.

        A vector of log probabilities for each row is returned.
        """
        nreps = data.shape[0]
        theta = data[:, 0:self.npars]
        x = data[:, self.npars:].reshape([nreps, self.T, self.ncomps])
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

    def __init__(self, x0, T, dt, theta_dist, cap_x=1e4,
                 replace_rejected_samples=False):
        super().__init__(x0=x0, ncomps=3, npars=4, T=T, dt=dt,
                         theta_dist=theta_dist)
        self.diff_scale = np.sqrt(10.) # Diffusion scale (hard-coded for now)
        self.cap_x = cap_x
        self.replace_rejected_samples = replace_rejected_samples

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
        (x1, x2, x3) = torch.unbind(x, -1)
        (th1, th2, th3, sig) = torch.unbind(theta, -1)
        drift1 = th1 * (x2 - x1)
        drift2 = th2 * x1 - x2 - x1 * x3
        drift3 = x1 * x2 - th3 * x3
        drift = torch.stack([drift1, drift2, drift3], axis=-1)
        dims = list(x.shape[:-1]) + [1,1]
        rootdiff = self.diff_scale * torch.tile(torch.eye(3), dims=dims)
        return drift, rootdiff
    
    def _log_det_jacobian(self, z, theta):
        """Returns log determinant of Jacobian for forward transformation from
        standard normal samples `z` and parameters `theta` to paths

        For this model the Jacobian is constant, so this can be done very quickly
        """
        nreps = z.shape[0]
        # sum of log root diffusion diagonal values over 1:T and 1:ncomps
        ldj = self.T * 3. * np.log(self.root_dt * self.diff_scale)
        return torch.full([nreps], ldj)

    def _sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE
        Samples with any x values above `self.cap_x` in magnitude are not returned

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        theta = self._sample_theta(nreps)
        x = self._sample_x(theta).reshape([nreps, -1]) # Flattened paths
        keep = torch.all(x < self.cap_x, dim=1, keepdim=True)
        theta = torch.masked_select(theta, keep).reshape([-1, self.npars])
        x = torch.masked_select(x, keep).reshape([-1, self.T*self.ncomps])
        return torch.cat([theta, x], axis=1)
    
    def sample(self, nreps=1):
        """Sample `nreps` replications of parameters + paths from SDE
        If `self.replace_rejected_samples` is False, x values above `self.cap_x` in magnitude
        are not returned. Otherwise they are replaced with fresh samples until there are `nreps`
        samples.

        Returns a tensor of shape `[n, self.npars + self.T * self.ncomps]`.
        Each row is a sample.
        The first `self.npars` columns are the parameters.
        The remaining columns are a flattened path.
        """
        out = self._sample(nreps)
        if self.replace_rejected_samples:
            nleft = nreps - out.shape[0]
            while nleft > 0:
                out = torch.cat([out, self._sample(nleft)], axis=0)
                nleft = nreps - out.shape[0]

        return out    


# Now follow some definitions used in next SDE class

# Initialise weights, following https://stackoverflow.com/a/49433937
def _init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.001)
        m.bias.data.fill_(0.)


# Transform so that Exp(0.1) mapped to N(0,1)
class ExpToNorm(Transform):
    def __init__(self):
        self.standard_normal = Normal(0., 1.)
        self.log10 = np.log(10.)
        super().__init__()

    def forward(self, inputs, context=None):
        x = inputs
        y_log_cdf = -0.1 * x
        y = self.standard_normal.icdf(torch.exp(y_log_cdf))
        y_log_pdf = -0.5 * np.log(2 * np.pi) - 0.5 * torch.square(y)
        abs_log_dx_dy = self.log10 + y_log_pdf - y_log_cdf
        ldj = -abs_log_dx_dy.sum(1)
        return y, ldj

    def inverse(self, inputs, context=None):
        y = inputs
        y_log_cdf = torch.log(self.standard_normal.cdf(y))
        x = -10. * y_log_cdf
        y_log_pdf = -0.5 * np.log(2 * np.pi) - 0.5 * torch.square(y)
        abs_log_dx_dy = self.log10 + y_log_pdf - y_log_cdf
        ldj = abs_log_dx_dy.sum(1)
        return x, ldj


class NeuralLorenzSDE(LorenzSDE):
    """Lorenz SDE with learnable drift and diffusion modifications
    and learnable parameter distribution

    `x0` is the initial state, a tensor of shape [3]
    `T` is the number of states
    `dt` is the discretisation time step
    `obs_indices` is array whose entries give indices of observed times
    `obs_data` is observed data, shape `len(obs_indices), 3`
    `hidden_size_x` is list of hidden layers widths for drift
    `nlayers_theta` is how many hidden flow layers to use for theta
    """

    def __init__(self, x0, T, dt,
                 obs_indices, obs_data,
                 hidden_size_x, nlayers_theta):
        transform = [ ExpToNorm() ]
        mask = torch.tensor([1., 1., 0., 0.])

        def create_net(in_features, out_features):
            return nn.nets.ResidualNet(
                in_features, out_features, hidden_features=20, num_blocks=10
            )

        for _ in range(nlayers_theta):
            transform.append(RandomPermutation(features=4))
            tf = AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_net,
                scale_activation=lambda x : (torch.exp(x) + 1e-3).clamp(0., 3.)
            )
            transform.append(tf)

        base_dist = distributions.StandardNormal(shape=[4])
        transform = CompositeTransform(transform)
        theta_dist = flows.Flow(transform, base_dist)
        theta_dist.apply(_init_weights)
        super().__init__(x0=x0, T=T, dt=dt, theta_dist=theta_dist)

        # A repeatedly used constant
        self.diff_mult_const = np.sqrt(10.) / np.log(2.) 
        ## Create lookup tables for features
        time_till_lookup = torch.zeros(T)
        obs_data_lookup = torch.zeros(T,3)
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
        self.time_till_lookup = time_till_lookup
        self.obs_data_lookup = obs_data_lookup

        ## Initialise neural network for drift
        self.nn_x = MLP(
            (15,), (4,), hidden_size_x, activation=elu
        )

        self.nn_x.apply(_init_weights)


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
        drift, _ = super()._get_coefs(x, theta, time_index) # Unmodified Lorenz drift
        x_mat = x.reshape([-1, self.ncomps])
        drift_mat = drift.reshape([-1, self.ncomps])
        theta_mat = theta.reshape([-1, self.npars])
        time_indices = time_index.reshape([-1])
        time_till_next_obs = self.time_till_lookup[time_indices]
        time_till_next_obs = time_till_next_obs.reshape([-1, 1])
        next_obs = self.obs_data_lookup[time_indices, :]
        times = self.dt * time_indices.reshape([-1,1])
        inputs = torch.cat(
            [x_mat, drift_mat, theta_mat, time_till_next_obs, next_obs, times],
            axis=1
        )
        outputs = self.nn_x(inputs)        
        drift_modification = outputs[:, 1:4].reshape(x.shape)
        drift = drift + drift_modification
        dims = list(x.shape[:-1]) + [1,1]
        diff_mult = outputs[:,0].reshape(dims)
        # Adding 1e-3 avoids near-singular matrices
        diff_mult = softplus(diff_mult) * self.diff_mult_const + 1e-3
        rootdiff = diff_mult * torch.tile(torch.eye(3), dims=dims)
        return drift, rootdiff

    def _log_det_jacobian(self, z, theta):
        ## Over-ride LorenzSDE method (which assumed fixed diffusion term)
        ## and use more general method of SDE
        return SDE._log_det_jacobian(self, z, theta)
