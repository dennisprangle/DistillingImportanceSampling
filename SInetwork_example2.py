# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:17:20 2022

@author: Cecilia
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:20:18 2022

@author: Cecilia
"""

import torch
from nflows import transforms, distributions, flows, nn
from DIS import DIS
from models.SInetwork import SInetworkModel
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
import statistics as st
plt.ion()

torch.manual_seed(111)

"Synthetic observations"
theta=torch.tensor([ 0.5833,  0.2042,  0.9461,  0.4926, -0.7683,  0.8084,  0.6812, -0.6109,
        -1.2449, -0.5074,  0.7361,  0.7243,  0.0851,  0.5613, -0.8666,  0.5357,
          2.7929,  0.0363,  0.0465,  0.5178, -1.3461, -0.0538,  1.2639, -1.4479,
          1.3920,  0.2770,  1.6843, -0.2228,  1.0129,  1.1868, -1.5221, -0.7176,
        -1.1272,  0.4316,  0.6116, -0.5346, -0.7434,  0.4664, -2.3321, -0.7594,
          0.4919,  0.9134, -0.5375,  0.9160, -0.8127,  0.4387,  0.1966, -0.7628,
          0.9750,  0.2971,  0.4657,  0.2370, -0.3403,  0.0146,  1.2364, -1.6066,
          0.2635])
p_inf = torch.distributions.Normal(0., 1.).cdf(theta[0])
p_con = torch.distributions.Normal(0., 1.).cdf(theta[1])

obs= [[[0],
  [0, 3, 6, 8],
  [0, 3, 6, 8, 5, 9, 4],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2],
  [0, 3, 6, 8, 5, 9, 4, 2]]]


ninputs = 57

"Model for analysis"
model = SInetworkModel( observations=obs,  n_nodes=10, n_inputs=ninputs)
model.max_eps = 50


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

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.)

approx_dist.apply(init_weights)

optimizer = torch.optim.Adam(
    approx_dist.parameters()
)

"Run the analysis"

dis12 = DIS(model, approx_dist, optimizer,
          importance_sample_size=5000, ess_target=250, max_weight=0.1)
dis12.pretrain(initial_target=model.initial_target, goal=0.5, report_every=10)

while dis12.eps > 0. or dis12.ess < 250.:
    dis12.train(iterations=20)
    plt.close('all')

    """
    Results visualization
    """
    nsamp = 1000
    proposals = dis12.train_sample.particles[0:nsamp]
    params = dis12.train_sample.sample(nsamp)
    prop_infection, prop_contact = model.convert_inputs(proposals)[0:2]
    sel_infection, sel_contact = model.convert_inputs(params)[0:2]

    "plot posterior distributions"
    _, ax= plt.subplots(1,2)
    ax[0].hist(sel_infection.numpy(), density=True,color='red', edgecolor='red', alpha=0.5)
    ax[0].vlines(p_inf, 0, 3.5, linestyle='-',color='black')
    ax[1].hist(sel_contact.numpy(), density=True,color='red', edgecolor='red', alpha=0.5)
    ax[1].vlines(p_con, 0, 3.5, linestyle='-',color='black')
    ax[0].set_xlabel('prob. infection')
    ax[1].set_xlabel('prob. contact')
    ax[0].set_ylabel('DIS')
    plt.savefig('SI_ex2_post.pdf')

    "Bivariate posterior plot"
    plt.figure()
    plt.plot(sel_infection.numpy(), sel_contact.numpy(), 'o')
    plt.xlabel('infection probability')
    plt.ylabel('contact probability')
    plt.savefig('SI_ex2_bivariate.pdf')

    "Posterior  of the network"
    G = model.convert_inputs(theta.view(1,ninputs))[-2][0]
    nxs = model.convert_inputs(params)[-2]
    n_nodes = 10
    n_edges = n_nodes*(n_nodes-1)/2

    posterior_mx  = np.zeros([n_nodes,n_nodes])
    posterior_adj = []
    list_infnodes =model.convert_inputs(params)[2]<model.convert_inputs(params)[0].view(nsamp,1)
    post_infnodes=torch.sum(list_infnodes,0)/nsamp
    i = 0
    for g in nxs:
        posterior_adj.append(str(np.vstack([nx.to_numpy_matrix(g),list_infnodes[i]])))
        i += 1

    for g in nxs:
        posterior_mx += nx.to_numpy_matrix(g)

    posterior_mx /= nsamp

    post_w=[]
    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            post_w.append(posterior_mx[i,j])
    
    mode = Counter(posterior_adj).most_common(1)[0][0]
    mode_m = np.matrix(mode).reshape(n_nodes+1,n_nodes)
    mode_graph = nx.from_numpy_matrix(mode_m[0:n_nodes,:])
    mode_inf = mode_m[-1,:]

    CG=nx.complete_graph(n_nodes)
    cmap=plt.cm.Blues
    label_e=dict(zip(CG.edges(), post_w))
    pos = nx.circular_layout(CG)  # Seed layout for reproducibility
    options = {
        #   "node_color": mode_inf.tolist()[0],
        "node_color": post_infnodes.tolist(),
        "edge_color": post_w,
        "width": 5,
        "cmap":cmap,
        "with_labels": False,
        "edge_cmap": plt.cm.Blues,
        #   "alpha":0.7,
        "node_size":800
    }
    options_true = {
        "node_color": ["r","w","r","r","r","r","r","w","r","r"],
        "edge_color": 'r',
        "width": 2,
        "with_labels": True,
        "font_size":14,
        #   "alpha":.8,
        "style":'--'    }
    plt.figure(figsize=[7,5])
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    plt.colorbar(sm)
    nx.draw(CG, pos, **options)
    nx.draw(G, pos, **options_true)
    #plt.title("Posterior modal network - epsilon={:.3f}, nr.layers={:d}".format(dis10.eps,15))
    plt.savefig('SI_ex2_net.pdf')

    nocon_ind = np.where(model.convert_inputs(theta.view(1,ninputs))[-1]>model.convert_inputs(theta.view(1,ninputs))[1])[1]
    con_ind = np.where(model.convert_inputs(theta.view(1,ninputs))[-1]<model.convert_inputs(theta.view(1,ninputs))[1])[1]
    model.convert_inputs(params)[-1]
    q1 = np.quantile(model.convert_inputs(params)[-1], 0.25, axis=0)
    q2 = np.quantile(model.convert_inputs(params)[-1], 0.75, axis=0)
    ax = plt.figure(figsize=[12,6])
    plt.scatter(np.arange(0,n_edges),torch.mean(model.convert_inputs(params)[-1],0), marker= "_", color='blue')
    plt.scatter(con_ind,torch.mean(model.convert_inputs(params)[-1][:,con_ind],0), marker= "_", color='red')
    plt.hlines(st.mean(sel_contact.numpy()), 0, n_edges, linestyle='-', linewidth=1, color='grey')
    plt.xticks(np.arange(0,n_edges), CG.edges(), rotation=100)
    plt.vlines(con_ind,q1[con_ind],q2[con_ind], color='red')
    plt.vlines(nocon_ind,q1[nocon_ind],q2[nocon_ind], color='blue')
    plt.hlines(np.quantile(sel_contact.numpy(),0.75), 0, n_edges, linestyle='--', linewidth=1, color='grey')
    plt.hlines(np.quantile(sel_contact.numpy(),0.25), 0, n_edges, linestyle='--',linewidth=1, color='grey')
    plt.Line2D([0,0], [0,0], color='blue')
    plt.Line2D([0,0], [0,0], color='red')
    plt.legend( labels=['no contact', 'contact','posterior prob. contact'])
    plt.savefig('SI_ex2_contact.pdf')

    "Correlation plot"
    plt.figure()
    edge_pars = model.convert_inputs(params)[-1]
    edge_pars = np.transpose(edge_pars)
    corr_matrix = np.corrcoef(edge_pars)
    plt.imshow(corr_matrix, cmap="hot")
    plt.colorbar()
    plt.savefig('SI_ex2_corr.pdf')
    plt.pause(0.1)

