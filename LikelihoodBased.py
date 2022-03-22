# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:37:36 2022

@author: Cecilia
"""
import time
import numpy as np

class ImpSamp:
    """Class to perform likelihood-based importance sampling analysis

       `model` a `Model` object encapsulating likelihood and importance distributions
       `S` is number of samples from the importance distributions
       `size` is number of samples from the the weighted importance sample
       """
    def __init__(self, model, S, size):
        
        self.model = model
        self.S = S
        self.size = size
    
    def get_proposals(self):
        self.model.get_parameters(self.S)
        return(self.model.particles)

    
    def compute_likelihood(self):
        return(self.model.likelihood())
    
    def get_weights(self):
        "default uniform prior"
        lik = self.compute_likelihood()
        self.weights = lik/self.model.ImpProb
        return(self.weights)
    
    def sample(self):
        start = time.time()
        proposals = self.get_proposals()
        w = self.get_weights()
        end = time.time()
        indexes = np.random.choice(list(range(self.S)), self.size, p=w/sum(w), replace=True)
        self.ess = sum(self.weights)**2/sum(self.weights**2)
        self.elapsed_time = end-start
        print(
                f"Effective sample size {self.ess}, "
                f"elapsed sec {self.elapsed_time:.1f}"
            )
        return(proposals[indexes,:])
        
    


