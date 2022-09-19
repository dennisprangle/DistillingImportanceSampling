import torch
from models.weighted_sample import WeightedSample

class SimulatorModel():
    """Encapsulates a prior and a model based on a simulator

    Particular applications should subclass this"""
    def __init__(self, observations, observations_are_data=True):
        """`observations_are_data` should be `True` if `observations` is
        direct simulator output, or `False` if it is summaries"""
        ## Subclasses can override this method to hardcode observations if desired
        if observations_are_data:
            self.obs_summaries = self.summaries(observations)
        else:
            self.obs_summaries = observations

    def log_prior(self, inputs):
        """Calculate vector of log prior densities of `inputs`

        Defaults to uniform"""
        n = inputs.shape[0]
        return torch.zeros(n)

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
        """Perform several model samples and return results as `WeightedSample` object
        `inputs` $n \times d$ tensor whose rows are model input vectors
        `log_proposal` length $n$ tensor of log proposal density for each input
        """
        sim_data = self.simulator(inputs)
        sim_summaries = self.summaries(sim_data)
        distances = self.obs_summaries.reshape((1,-1)) - sim_summaries
        squared_distances = torch.sum(distances ** 2., 1)
        return WeightedSample(
            inputs,
            log_proposal,
            self.log_prior(inputs),
            squared_distances
        )

