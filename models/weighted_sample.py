import torch
import numpy as np
from utils import effective_sample_size

class WeightedSample:
    """A weighted sample of model inputs"""
    def __init__(self, particles, weights, epsilon):
        """Arguments are:
        `particles` $n \times d$ tensor whose rows are model input vectors
        `weights` length $n$ tensor of particle weights (can be unnormalised)
        `epsilon` current tempering parameter to use
        """
        self.particles = particles
        wsum = weights.sum()
        if not wsum > 0.:
            raise ValueError(f'Sum of weights is {wsum} but should be positive')
        self.weights = weights / wsum
        self.epsilon = epsilon
        self.ess = effective_sample_size(self.weights)
        self.other = {}
        # `other` should be implemented by subclasses
        # It should contain 1-dim tensors of other particle properties

    def sample(self, m):
        """Returns `m` unweighted samples"""
        i = torch.multinomial(self.weights, m, replacement=True)
        return self.particles[i,:]

    def get_alternate_weights(self, epsilon):
        """Return weights appropriate to another `epsilon` value"""
        raise NotImplementedError

    def update_epsilon(self, epsilon):
        """Update epsilon and modify weights appropriately"""
        self.weights = self.get_alternate_weights(epsilon)
        self.epsilon = epsilon
        self.ess = effective_sample_size(self.weights)

    def find_eps(self, target_ess, upper, bisection_its=50):
        """Return epsilon value <= `upper` giving ess matching `target_ess` as closely as possible

        Bisection search is performed using `bisection_its` iterations
        """
        w = self.get_alternate_weights(upper)
        ess = effective_sample_size(w)
        if ess < target_ess:
            return upper

        lower = 0.
        for i in range(bisection_its):
            eps_guess = (lower + upper) / 2.
            w = self.get_alternate_weights(eps_guess)
            ess = effective_sample_size(w)
            if ess > target_ess:
                upper = eps_guess
            else:
                lower = eps_guess

        # Consider returning eps=0 if it's still an endpoint
        if lower == 0.:
            w = self.get_alternate_weights(0.)
            ess = effective_sample_size(w)
            if ess > target_ess:
                return 0.

        # Be conservative by returning upper end of remaining range
        return upper

    def truncate_weights(self, max_weight):
        """Truncate weights in-place

        All calling no weights are above `max_weight` times total of weights.

        The function modifies the weights in-place
        and returns their unnormalised sum"""
        S = sum(self.weights)
        to_trunc = (self.weights > S*max_weight)
        n_to_trunc = sum(to_trunc)
        if n_to_trunc == 0:
            S = sum(self.weights)
            if not S > 0.:
                raise ValueError(f'Sum of weights is {S} but should be positive')
            self.weights /= S
            return S
    
        print(f"Truncating {n_to_trunc:d} weights")
        to_not_trunc = torch.logical_not(to_trunc)
        sum_untrunc = sum(self.weights[to_not_trunc])
        if sum_untrunc == 0:
            # Impossible to truncate further!
            S = sum(self.weights)
            if not S > 0.:
                raise ValueError(f'Sum of weights is {S} but should be positive')
            self.weights /= S
            return S
        trunc_to = max_weight * sum_untrunc / (1. - max_weight * n_to_trunc)
        max_untrunc = torch.max(self.weights[to_not_trunc])
        ## trunc_to calculation is done so that
        ## after w[to_trunc]=trunc_to
        ## w[to_trunc] / sum(w) all equal max_weight
        ## **But** we don't want to truncate below next smallest weight
        if trunc_to >= max_untrunc:
            self.weights[to_trunc] = trunc_to
            S = sum(self.weights)
            if not S > 0.:
                raise ValueError(f'Sum of weights is {S} but should be positive')
            self.weights /= S
            return S
        else:
            self.weights[to_trunc] = max_untrunc
            return self.truncate_weights(max_weight)



class SimulatorWeightedSample(WeightedSample):
    """A weighted sample of simulator inputs"""
    def __init__(self, particles, log_proposal, log_prior, squared_distances):
        """Arguments are:
        `particles` $n \times d$ tensor whose rows are simulator input vectors
        `log_proposal` length $n$ tensor of log proposal density for each particle
        `log_prior` length $n$ tensor of log prior for each particle
        `squared distances` length $n$ tensor of squared distances between simulator output and observations
        """
        # In next line we basically want:
        # log_weights = log_prior - log_proposal
        # But if prior density == proposal density == 0 this gives nan.
        # So use weight zero in this case.
        log_weights = torch.where(log_prior == -np.inf, log_prior, log_prior - log_proposal)
        # Crude normalisation to avoid overflow/underflow when exponentiating
        log_weights = log_weights - torch.max(log_weights)
        weights = torch.exp(log_weights)
        super().__init__(particles, weights, np.inf) # Includes normalisation
        self.other = {"squared_distances" : squared_distances}


    def get_alternate_weights(self, epsilon):
        """Return weights appropriate to another `epsilon` value"""
        sqd = self.other["squared_distances"]
        new_eps = epsilon
        old_eps = self.epsilon

        # A simple version of generic reweighting code
        # w = self.weights
        # w /= torch.exp(-0.5*sqd / old_eps**2.)
        # w *= torch.exp(-0.5*sqd / new_eps**2.)
        # w /= sum(w)

        if new_eps == 0:
            w = self.weights
            # Remove existing distance-based weight contribution
            w /= torch.exp(-0.5 * sqd / old_eps**2.)
            # Replace with a indicator function weight contribution
            w = torch.where(
                sqd==0.,
                w,
                torch.zeros_like(w)
            )
        else:
            # TODO Avoid need to normalise by always using log weights?
            # Normalising sqd part 1
            # Ignore distances if weight already zero
            # (to avoid rare possibility of setting all weights to zero)
            sqd_pos_weight = torch.where(
                self.weights > 0,
                sqd,
                torch.full_like(sqd, sqd.max())
            )
            # Normalising sqd part 2
            # Reduce chance of exponentiation giving zero weights
            sqd_norm = sqd_pos_weight - sqd_pos_weight.min()
            # A more efficient way to do the generic case
            a = 0.5 * (old_eps**-2. - new_eps**-2.)
            w = self.weights * torch.exp(sqd_norm*a)

        wsum = w.sum()
        if not wsum > 0.:
            raise ValueError(f'Sum of weights is {wsum} but should be positive')

        return w / wsum


class LikelihoodWeightedSample(WeightedSample):
    """A weighted sample of likelihood-based model inputs"""
    def __init__(self, particles, log_proposal, log_prior, log_likelihood):
        """Arguments are:
        `particles` $n \times d$ tensor whose rows are model input vectors
        `log_proposal` length $n$ tensor of log proposal density for each particle
        `log_prior` length $n$ tensor of log prior for each particle
        `log_likelihood` length $n$ tensor of log likelihood for each particle
        """
        log_weights = log_prior - log_proposal
        # Crude normalisation to avoid overflow/underflow
        log_weights = log_weights - log_weights.max()
        weights = torch.exp(log_weights)
        super().__init__(particles, weights, 1.) # Includes normalisation
        self.other = {"log_likelihood" : log_likelihood}

    def get_alternate_weights(self, epsilon):
        """Return weights appropriate to another `epsilon` value"""
        # An easy-to-read version of the reweighting code is:
        # logw = torch.log(self.weights)
        # logw -= (1. - self.epsilon) * self.log_likelihood
        # logw += (1. - epsilon) * self.log_likelihood
        # w = torch.exp(logw)
        # self.weights = w / sum(w)

        delta = self.epsilon - epsilon
        if delta == 0.:
            return self.weights

        ll = self.other["log_likelihood"]
        # Normalising ll part 1
        # Ignore likelihood if weight already zero
        # (to avoid rare possibility of setting all weights to zero)
        ll_pos_weight = torch.where(
            self.weights > 0.,
            ll,
            torch.full_like(ll, -np.inf)
        )
        # Normalising ll part 2
        # Reduce chance of exponentiation giving zero weights
        ll_norm = ll_pos_weight - ll_pos_weight.max()
        w = self.weights * torch.exp(delta * ll_norm)
        wsum = w.sum()
        if not wsum > 0.:
            raise ValueError(f'Sum of weights is {wsum} but should be positive')
        return w / wsum


class InitialFinalWeightedSample(WeightedSample):
    """A sample of model inputs with weights interpolating between initial and final target densities"""
    def __init__(self, particles, log_proposal, log_initial, log_final):
        """Arguments are:
        `particles` $n \times d$ tensor whose rows are model input vectors
        `log_proposal` length $n$ tensor of log proposal density for each particle
        `log_initial` length $n$ tensor of log initial densities for each particle
        `log_final` length $n$ tensor of log final densities for each particle
        `log_likelihood` length $n$ tensor of log likelihood for each particle
        """
        log_weights = log_initial - log_proposal
        # Crude normalisation to avoid overflow/underflow
        log_weights = log_weights - torch.max(log_weights)
        weights = torch.exp(log_weights)
        super().__init__(particles, weights, 1.) # Includes normalisation
        self.other = {"log_initial" : log_initial, "log_final" : log_final}

    def get_alternate_weights(self, epsilon):
        """Return weights appropriate to another `epsilon` value"""
        # An easy-to-read version of the reweighting code is:
        # logw = torch.log(self.weights)
        # logw -= self.epsilon * self.log_initial + (1. - self.epsilon) * self.log_final
        # logw += epsilon * self.log_initial + (1. - epsilon) * self.log_final
        # w = torch.exp(logw)
        # self.weights = w / sum(w)

        w = self.weights.detach().clone()
        delta = self.epsilon - epsilon
        li = self.other["log_initial"]
        lf = self.other["log_final"]
        if delta > 0.:
            w *= torch.exp(delta * (lf - li))
            w /= torch.sum(w)
        return w
