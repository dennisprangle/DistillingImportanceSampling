# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:06:38 2021

@author: Cecilia
"""

import networkx as nx
import copy
from models.models import SimulatorModel
import torch
from torch.distributions import MultivariateNormal

   

class SInetworkModel(SimulatorModel):
    
    """A SI model on an Erdős–Rényi random network
    Arguments:
    `observations` should be list containing a list of lists.
    `infection_start_point` is a tensor containing the indexes infective nodes at time 0 (default is 0) 
    `n_timestep` number of simulated steps
    `n_nodes` number of nodes (size of the population)
    `n_inputs` is the number of parameters and latent variables
    The use case of `observations = None` (the default) is when this object is being used to simulate data rather than perform inference e.g.
    ```
    m = SInetworkModel( n_timestep=5, n_nodes=5, n_inputs=17, observations_are_data=False) 
    theta = torch.zeros([1,17])
    obs = m.simulate_from_parameters(theta.view(1,17))
    ```

    """

    def __init__(self, observations=None, infection_start_point=torch.zeros([1], dtype=torch.int32), 
               n_timestep=10, n_nodes=10, n_inputs=57,  observations_are_data=True):
        time_observed = list(range(n_timestep))
        if observations is not None:
            infection_start_point = torch.Tensor(observations[0][0])
            time_observed = list(range(len(observations[0])))
            n_timestep = len(time_observed)  

        self.infection_start_point = infection_start_point
        self.time_observed = time_observed
        self.standard_normal = torch.distributions.Normal(0., 1.)
        self.n_timestep = n_timestep
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.prior = MultivariateNormal(torch.zeros(self.n_inputs), torch.eye(self.n_inputs))
        super().__init__(observations, observations_are_data)

    def log_prior(self, inputs):
        """calculate vector of log prior densities of `inputs`

        for this model they are independent Unif(0,1)"""
        return torch.sum(self.standard_normal.log_prob(inputs), dim=1)
    
    def convert_inputs(self, inputs):
        """Converts inputs from normal scale to standard parameters 

        Each row is input for a simulation.
        Columns `0` is the per-contact probability of infection
        Columns `1` is the probability of contact
        Columns `(n(n-1))/2` are latent variables governing the edges creation
        Other columns are latent variables governing the infection for each node
        """
        inputs_u = self.standard_normal.cdf(inputs)
        p_infection = inputs_u[:,0]
        p_contact = inputs_u[:,1]
        individual_probabilities = inputs_u[:,-self.n_nodes:]
        edges_probabilities = inputs_u[:,2:-self.n_nodes]
        g_list = []
        for h in range(inputs_u.shape[0]):
            g = nx.Graph()
            g.add_nodes_from(range(0, self.n_nodes))
            #z = 2
            z = 0
            for i in g.nodes():
                for j in g.nodes():
                    if (i < j): 
                        if (edges_probabilities[h,z] < p_contact[h]):
                        #if (inputs_contact[z]< p_contact[h]):
                            g.add_edge(i, j)
                        z+=1 
            g_list.append(g)
        return p_infection, p_contact, individual_probabilities, g_list, edges_probabilities
        


    def simulator(self, inputs):
        p_infection, _ ,individual_probabilities, g_list,_ = self.convert_inputs(inputs)
        nsim = individual_probabilities.shape[0]
        diffusionstate_array = [None]*nsim
        for k in range(nsim):        
            network = g_list[k]
            # Initialize the time-series
            diffusionstate_array_tmp = []
            infected_nodes = self.infection_start_point.tolist()
            present_infected_nodes = copy.deepcopy(infected_nodes)
            if 0 in self.time_observed:
                diffusionstate_array_tmp.append(present_infected_nodes)
            for ind_t in range(1, self.n_timestep):
                for ind_l in present_infected_nodes:
                #chosen_node_for_infection = np.random.choice([x for x in network.neighbors(ind_l)], 1)[0]
                    node_for_infection = [x for x in network.neighbors(ind_l)]
                    for n in node_for_infection:
                        if (individual_probabilities[k,n]<p_infection[k])*(n not in infected_nodes)==1:
                            infected_nodes.append(n)
                present_infected_nodes = copy.deepcopy(infected_nodes)
                if ind_t in self.time_observed:
                    diffusionstate_array_tmp.append(present_infected_nodes)
            diffusionstate_array[k] = diffusionstate_array_tmp
        # return an array of objects of list containing infected_nodes at the observed time-points
        return (diffusionstate_array)
    
    # def simulator(self, inputs):
    #     p_infection, _ ,individual_probabilities, g_list,_ = self.convert_inputs(inputs)
    #     nsim = individual_probabilities.shape[0]
    #     diffusionstate_array = [None]*nsim
    #     for k in range(nsim):        
    #         network = g_list[k]
    #         # Initialize the time-series
    #         diffusionstate_array_tmp = []
    #         infected_nodes = self.infection_start_point.tolist()
    #         present_infected_nodes = copy.deepcopy(infected_nodes)
    #         if 0 in self.time_observed:
    #             diffusionstate_array_tmp.append(present_infected_nodes)
    #         for ind_t in range(1, self.n_timestep):
    #             for ind_l in present_infected_nodes:
    #             #chosen_node_for_infection = np.random.choice([x for x in network.neighbors(ind_l)], 1)[0]
    #                 node_for_infection = [x for x in network.neighbors(ind_l)]
    #                 for n in node_for_infection:
    #                     if (individual_probabilities[k,n]<p_infection[k])*(n not in infected_nodes)==1:
    #                         infected_nodes.append(n)
    #             present_infected_nodes = copy.deepcopy(infected_nodes)
    #             if ind_t in self.time_observed:
    #                 diffusionstate_array_tmp.append(present_infected_nodes)
    #         diffusionstate_array[k] = diffusionstate_array_tmp
    #     # return an array of objects of list containing infected_nodes at the observed time-points
    #     return (diffusionstate_array)
    

    def simulate_from_parameters(self, theta):
        """Perform simulations from tensor of parameters only"""
        inputs = theta
        return self.simulator(inputs)

    def summaries(self, data):
        """Converts data to summary statistics

        The output should be a tensor of shape
        (number of simulations, number of summaries)
        
        ```
        Here we compare the whole obserevde data.
        This function allows to get a different form for the data.
        For each simulation the output is a tensor of length n_obs*n_timestep.
        Each elemnt of the tensor can be 1 (node infective at the correspondig time step)
        or 0 (node susceptible at the corresponding time step)
            
        """
        summaries = torch.zeros([len(data),self.n_timestep*self.n_nodes])
        for d in range(len(data)):
            I = torch.zeros([self.n_timestep,self.n_nodes])
            for t in range(self.n_timestep):
                I[t, data[d][t]]=1
            summaries[d,] = I.reshape(1,-1)
        return summaries
    
    
    # def summaries(self, data):
    #     """Converts data to summary statistics

    #     The output should be a tensor of shape
    #     (number of simulations, number of summaries)
    #     In this case we don't summarize data but the output for each simulation 
    #     is """
    #     summaries = torch.zeros([len(data),self.n_timestep+self.n_nodes])
    #     for d in range(len(data)):
    # #        summaries[d,] = torch.tensor([len(t)/self.n_nodes for t in data[d]])
    #         for t in range(len(data[d])):
    #             summaries[d,data[d][t]]+=1
    #             summaries[d,self.n_nodes+t]=len(data[d][t])/self.n_nodes
    #     return summaries
