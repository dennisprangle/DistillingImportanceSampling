# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:57:41 2022

@author: Cecilia
"""


import numpy as np
import networkx as nx
import itertools


class SInetworkLikelihood:
    
    def __init__(self, obs, n_nodes, pIDist=None, pCDist= None, particles=None):
        if particles != None:
            self.p_I, self.p_C= particles
        self.obs = obs
        self.n_nodes = n_nodes
        self.n_timestep = len(obs)
        self.M = int(n_nodes*(n_nodes-1)/2)
        self.pIDist = pIDist
        self.pCDist = pCDist

    def get_parameters(self, S):
        self.p_I = self.pIDist.rvs(S)
        self.p_C = self.pCDist.rvs(S)
        self.ImpProb = self.pIDist.pdf(self.p_I)* self.pCDist.pdf(self.p_C)
        self.particles= np.vstack((self.p_I, self.p_C)).T

         
        
    def prob_network(self,G):
        
        """
    

        Parameters
        ----------
        G : nx object
        n_nodes : int- number of nodes in the network
        p_C : float- probability of and adges between each pair of nodes

        Returns
        -------
        float - probability of observing the given the network G guven 
        the number of nodes and the probabilty of contact

        """
        D = G.number_of_edges()
        prob_N = self.p_C**D*(1-self.p_C)**(self.M-D)
        return(prob_N)
 
        
    def prob_conditional_to_N (self, G):
              
        """
    

        Parameters
        ----------
        obs : a list of lists of infectious node at each point in time.
        n_nodes : the number (int) of nodes in the graoph.
        n_timestep : nr (int) of points in time (nr of lists in the list).
        G : an nx object corresponding to the assumed graph structure.
        p_I : (float) probability that an infectious node has to infect a neighbourhoods.

        Returns
        -------
        the probability of the sequence given the network structure
       """
        N = nx.convert_matrix.to_numpy_array(G)
        I = np.zeros([self.n_timestep,self.n_nodes])
        I[0,0] = 1
        sizeA = 0
        bar_I = [[],[1]+[0]*(self.n_nodes-1)]
        C01 = 0
        susceptible = np.full(self.n_nodes, True)
        susceptible[0] = False
        for t in range(1,self.n_timestep):
            I[t, self.obs[t]]=1     
            C01 += len(self.obs[t])-len(self.obs[t-1])
            if t>1:
                bar_I.append(I[t-1]*(1-I[t-2]))
            p1 = np.dot(bar_I[t],N)
            for n in range(self.n_nodes):
                if susceptible[n] and p1[n]>0 and I[t-1,n]==0:
                    sizeA += 1
                    susceptible[n] = False
                else:
                    if I[t-1,n]!=I[t,n]:
                        return(0)
        C00 = sizeA-C01
        return(self.p_I**C01*(1-self.p_I)**(C00))
        
    def likelihood(self):
        """
    

        Parameters
        ----------
        obs : list of lists of infectcious nodes
        n_nodes : int - number of nodes
        n_timestep : int - nr of point in time
        p_I :float - probability of infection
        p_C : float - probability of contact

        Returns
        -------
        float - probability od the observavtions given the probability of contact and infection

        """
        likelihood = 0
   
        for p in itertools.product([0,1], repeat=self.M):
            adjacency = np.zeros([self.n_nodes,self.n_nodes])
            h=0
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if j>i:
                        adjacency[i,j]=adjacency[j,i]=p[h]
                        h+=1
            G = nx.from_numpy_matrix(adjacency)
            likelihood += self.prob_conditional_to_N(G)*self.prob_network( G)
        return(likelihood)    

