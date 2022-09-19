# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:09:46 2022

@author: Cecilia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 10:31:19 2022

@author: Cecilia
"""




import torch
from nflows import transforms, distributions, flows, nn
from DIS import DIS
from models.SInetwork import SInetworkModel
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal
from SInetworkLikelihood import SInetworkLikelihood
from LikelihoodBased import ImpSamp
from scipy import stats
plt.ion()

torch.manual_seed(111)



ninputs =  17   #2 + 5 + 5(4)/2


"Syntetic observations"
obs= [[[0], [0, 1, 3], [0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]] 


p_inf = torch.distributions.Normal(0., 1.).cdf(torch.tensor(-0.1000))
p_con = torch.distributions.Normal(0., 1.).cdf(torch.tensor(-0.4151))
"Model for analysis"
model = SInetworkModel( observations=obs,  n_nodes=5, n_inputs=ninputs)

" Setting up normalising flows "
base_dist = distributions.StandardNormal(shape=[ninputs])

transform = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
    features = ninputs,
    hidden_features = 20,
    num_bins = 5,
    tails = "linear",
    tail_bound = 10.,
    num_blocks = 3
)

approx_dist = flows.Flow(transform, base_dist)

optimizer = torch.optim.Adam(
    approx_dist.parameters()
)




"Run the analysis"

dis2 = DIS(model, approx_dist, optimizer,
          importance_sample_size=5000, ess_target=250, max_weight=0.1)
dis2.pretrain(initial_target=model.prior, goal=0.75, report_every=10)

while dis2.eps > 0. or dis2.ess < 250.:
    dis2.train(iterations=1)

nsamp = 10000
with torch.no_grad():
    weighted_params = dis2.get_sample(10*nsamp)
weighted_params.update_epsilon(0.0)
params = weighted_params.sample(nsamp).detach()
sel_infection, sel_contact = model.convert_inputs(params)[0:2]

"Run the likelihood-based analysis"

modlik = SInetworkLikelihood(obs[0],5, stats.beta(1,1), stats.beta(1,1))
IS = ImpSamp(modlik, S=10*nsamp, size=nsamp)
sample = IS.sample()

plt.hist(sample[:,0], density=True, alpha=0.6)
plt.hist(sel_infection.numpy(), density=True, alpha=0.6)
plt.title('Probability of infection')
plt.show()

plt.hist(sample[:,1], density=True, alpha=0.6)
plt.hist(sel_contact.numpy(), density=True, alpha=0.6)
plt.title('Probability of contact')
plt.show()

_, ax= plt.subplots(2,2)
ax[0,0].hist(sample[:,0], density=True, edgecolor='C0',alpha=0.5)
ax[0,0].set_ylabel('IS')
ax[0,1].hist(sample[:,1], density=True,edgecolor='C0', alpha=0.5)
ax[1,0].hist(sel_infection.numpy(), density=True,color='red', edgecolor='red', alpha=0.5)
ax[1,1].hist(sel_contact.numpy(), density=True,color='red', edgecolor='red', alpha=0.5)
ax[1,0].set_ylabel('DIS')
ax[1,0].set_xlabel('prob. infection')
ax[1,1].set_xlabel('prob. contact')
plt.savefig("SI_ex1_post.pdf")
