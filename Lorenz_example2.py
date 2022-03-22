from DIS import DIS
from time import time
from models.sde import NeuralLorenzSDE
from models.lorenz import LorenzModel
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
plt.ion()

torch.manual_seed(111)

with open('Lorenz_data.pkl', 'rb') as infile:
    lorenz_data = pickle.load(infile)

pars0, T, dt, x0, true_path, obs_data, obs_indices = lorenz_data.values()

true_path = torch.cat([x0.unsqueeze(1), true_path], dim=1)

obs_indices1 = [i+1 for i in obs_indices] #adjust to 1-based indexing for plots

lorenz_model = LorenzModel(x0=x0, T=T, dt=dt, obs_indices=obs_indices,
                           obs_data=obs_data, obs_scale=0.2)

## Approximating family
dis_approx = NeuralLorenzSDE(x0=x0, T=T, dt=dt,
                             obs_indices=obs_indices, obs_data=obs_data,
                             hidden_size_x=(80,80,80),
                             nlayers_theta=12)

optimizer = torch.optim.Adam(dis_approx.parameters())

dis = DIS(lorenz_model, dis_approx, optimizer,
          importance_sample_size=5000, ess_target=250, max_weight=0.1)

paths_toplot = 30
paths0 = np.zeros((paths_toplot,3,1))
for j in range(3):
    paths0[:,j,:] = x0[j]

while dis.eps > 0. or dis.ess < 250.:
    dis.train(iterations=10)

    # Plot some IS output
    plt.close('all')
    samp = dis.train_sample.sample(1000)
    pars_sample = samp[:,0:3]
    df = pd.DataFrame(pars_sample.detach().numpy(),
                      columns=['theta1', 'theta2', 'theta3'])

    scat = pd.plotting.scatter_matrix(df)
    # Show true parameter values
    # and extend axis to include them
    for j in range(3):
        scat[j,j].axvline(x=pars0[j], c='k')
        lims = list(scat[j,j].get_xlim())
        lims[0] = np.min((lims[0], pars0[j]))
        lims[1] = np.max((lims[1], pars0[j]))
        scat[j,j].set_xlim(lims)
    plt.savefig('Lorenz_ex2_pars.pdf')

    plt.figure()
    paths = np.reshape(samp[0:paths_toplot, 4:], (paths_toplot, 3, -1),
                       order='F')
    paths = np.concatenate((paths0, paths), 2)
    for j in range(paths_toplot):
        plt.plot(range(T+1), paths[j,0,:], 'r-.', alpha=0.3)
        plt.plot(range(T+1), paths[j,1,:], 'b:', alpha=0.3)
        plt.plot(range(T+1), paths[j,2,:], 'g--', alpha=0.3)
    plt.plot(range(T+1), true_path[0,:], 'r-')
    plt.plot(range(T+1), true_path[1,:], 'b-')
    plt.plot(range(T+1), true_path[2,:], 'g-')
    plt.plot(obs_indices1, obs_data[:,0], 'ro', ms=10, alpha=0.8)
    plt.plot(obs_indices1, obs_data[:,1], 'bo', ms=10, alpha=0.8)
    plt.plot(obs_indices1, obs_data[:,2], 'go', ms=10, alpha=0.8)
    plt.xlabel('i')
    plt.savefig('Lorenz_ex2_paths.pdf')

np.save('Lorenz_example2_pars', pars_sample.detach().numpy())
wait = input('Press enter to terminate')
