import torch
from models.weighted_sample import SimulatorWeightedSample, LikelihoodWeightedSample

# This file defines a class to encapsulate models and related details
# (e.g. priors, tempering)
# Particular applications should create use subclasses of this.
# Several subclasses are at the end of the file.

class Model:
    def __init__():
        self.max_eps = 1. ## Value of epsilon for initial tempering
                          ## Some subclasses will need to override this

    """Encapsulates model and prior"""
    def run(self, inputs, log_proposal):
        """Perform several model samples and return results as `WeightedSample` object
        `inputs` $n \times d$ tensor whose rows are model input vectors
        `log_proposal` length $n$ tensor of log proposal density for each input
        """
        raise NotImplementedError

    def log_prior(self, inputs):
        """Calculate vector of log prior densities of `inputs`

        Defaults to uniform"""
        n = inputs.shape[0]
        return torch.zeros(n)

class SimulatorModel(Model):
    """A model based on a simulator"""
    def __init__(self, observations, observations_are_data=True):
        """`observations_are_data` should be `True` if `observations` is
        direct simulator output, or `False` if it is summaries"""
        ## Subclasses can override this method to hardcode observations if desired
        if observations_are_data:
            self.obs_summaries = self.summaries(observations)
        else:
            self.obs_summaries = observations

    def simulator(self, inputs):
        """Perform simulations"""
        ## It can be useful to directly access this
        ## e.g. to simulate datasets which will be analysed.
        ## In this case subclasses could add more user-friendly access methods,
        ## with named arguments which are concatenated into `inputs`.
        raise NotImplementedError

    def summaries(self, data):
        """Converts data to summary statistics

        The output should be a tensor of shape
        (number of simuations, number of summaries)

        This function defaults to the identity transformation."""
        return data

    def run(self, inputs, log_proposal):
        sim_data = self.simulator(inputs)
        sim_summaries = self.summaries(sim_data)
        distances = self.obs_summaries.reshape((1,-1)) - sim_summaries
        squared_distances = torch.sum(distances ** 2., 1)
        return SimulatorWeightedSample(
            inputs,
            log_proposal,
            self.log_prior(inputs),
            squared_distances
        )

class LikelihoodModel(Model):
    """A model based on likelihood calculations"""

    def log_likelihood(self, inputs):
        """Calculate log likelihood values"""
        raise NotImplementedError

    def run(self, inputs, log_proposal):
        return LikelihoodWeightedSample(
            inputs,
            log_proposal,
            self.log_prior(inputs),
            self.log_likelihood(inputs)
        )
