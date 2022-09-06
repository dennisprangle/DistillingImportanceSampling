import torch
from torch.nn.functional import relu
from torch.distributions import MultivariateNormal
from models.models import SimulatorModel

class MG1Model(SimulatorModel):
    """A MG1 queueing model

    Arguments:
    `observations` should be a 1-dim tensor of observed inter-departure times
    `nobs` is number of observations (set automatically for `observations` not `None`)
    `max_arrival` inter-arrival times are clipped to this threshold (avoids numerical problems in simulation)

    The use case of `observations = None` (the default) is when this object is being used to simulate data rather than perform inference e.g.
    ```
    m = MG1Model(nobs = 5)
    theta = torch.zeros([1,3])
    obs = m.simulate_from_parameters(theta)
    m.observations = obs[0,:]
    ```
    The object `m` can now be used for inference on the simulated dataset.
    """
    def __init__(self, observations=None, nobs=10, max_arrival=1E6):
        if observations is not None:
            nobs = observations.shape[0]
        self.nobs = nobs
        self.max_arrival = max_arrival
        self.standard_normal = torch.distributions.Normal(0., 1.)
        n_inputs = 3+2*nobs
        self.prior = MultivariateNormal(torch.zeros(n_inputs), torch.eye(n_inputs))
        super().__init__(observations, observations_are_data=True)

    def log_prior(self, inputs):
        """Calculate vector of log prior densities of `inputs`

        For this model they are independent N(0,1)"""
        return torch.sum(self.standard_normal.log_prob(inputs), dim=1)

    def convert_inputs(self, inputs):
        """Converts inputs from normal scale to standard parameters + latents

        Here `inputs` is a 2 dimensional tensor.
        Each row is input for a simulation.
        Columns `0:3` control the parameters.
        Columns `3:3+self.nobs` control arrival times.
        Columns `3+self.nobs:3+2*self.nobs` control service times.
        Using Normal(0,1) samples everywhere gives samples from the prior.

        Returns tuple of arrival rates, minimum service times, service time width, arrivals, services. The first 3 of these are 1-dim tensor of parameters. The remaining 2 are 2-dim tensors with `self.nobs` columns of latent variables.

        Alternatively, if `inputs` has 3 columns, only parameters are calculated, with `None` returned for the latent variables.
        """
        inputs_u = self.standard_normal.cdf(inputs) # TODO: update to use utils.norm_to_unif instead
        ## Get standard form of parameters
        arrival_rate = inputs_u[:,0] / 3.
        min_service = inputs_u[:,1] * 10.
        service_width = inputs_u[:,2] * 10.

        if inputs.shape[1] == 3:
            return arrival_rate, min_service, service_width, None, None

        ## Get inter-arrival and inter-service times
        arrivals_u = inputs_u[:,3:3+self.nobs]
        services_u = inputs_u[:,3+self.nobs:3+2*self.nobs]
        ## Adding 1e-20 avoids rare numerical errors (0/0, giving nan)
        arrivals = (1e-20 - torch.log(arrivals_u)) / arrival_rate.reshape([-1, 1])
        arrivals.clamp_(0., self.max_arrival)
        services = min_service.reshape([-1, 1]) + \
                   services_u * service_width.reshape([-1, 1])

        return arrival_rate, min_service, service_width, arrivals, services

    def simulator(self, inputs):
        """Perform simulations

        Here `inputs` is a 2 dimensional tensor.
        Each row is input for a simulation.
        Columns `0:3` control the parameters.
        Columns `3:3+self.nobs` control arrival times.
        Columns `3+self.nobs:3+2*self.nobs` control service times.
        Using Normal(0,1) samples everywhere gives samples from the prior.

        Returns a tensor of shape `inputs.shape[0], self.nobs`.
        Each row gives inter-departure times from a queue simulation.
        """
        n = inputs.shape[0] ## Number of simulations to perform
        _, _, _, arrivals, services = self.convert_inputs(inputs)
        # Compute interdeparture times in `departures`
        # n.b. interdeparture time = service time + time queue empty pre-arrival
        departures = services
        current_arrival = torch.zeros(n) # Arrival times of customer i
        last_departure = torch.zeros(n) # Departure times of customer i-1
        for i in range(self.nobs):
            current_arrival += arrivals[:,i]
            departures[:,i] += relu(current_arrival - last_departure)
            last_departure += departures[:,i]

        return departures

    def simulate_from_parameters(self, theta):
        """Perform simulations from tensor of parameters only"""
        n = theta.shape[0] ## Number of simulations to perform
        latents = torch.randn([n, 2*self.nobs])
        inputs = torch.cat([theta, latents], dim=1)
        return self.simulator(inputs)
