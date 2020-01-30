from DIS import DIS
from Lorenz_functions import Lorenz_model
from time import time
from SDE import LorenzSDE, NeuralLorenzSDE
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfe = tf.contrib.eager
tf.enable_eager_execution()
import matplotlib.pyplot as plt
plt.ion()

pars0, T, dt, x0, true_path, obs_data, obs_indices = np.load('Lorenz_data.npy',
                                                             allow_pickle=True)
true_path = np.hstack((np.expand_dims(x0, axis=1), true_path))

obs_indices1 = [i+1 for i in obs_indices] #adjust to 1-based indexing for plots

##Prior is Exp(0.1)
base = tfd.Independent(
    distribution = tfd.Uniform(low = np.zeros(4, dtype=np.float32),
                               high = np.ones(4, dtype=np.float32)),
    reinterpreted_batch_ndims=1)
unif2exp = [tfb.AffineScalar(scale=-10.), tfb.Invert(tfb.Exp())]
bijector = tfb.Chain(unif2exp)
prior = tfd.TransformedDistribution(base, bijector)

initial_target = LorenzSDE(x0=x0, T=T, dt=dt,
                           theta_dist=prior,
                           replace_rejected_samples=True)

lorenz_model = Lorenz_model(x0=x0, T=T, dt=dt, obs_indices=obs_indices,
                            obs_data=obs_data, prior=prior,
                            initial_target=initial_target)

## Approximating family
dis_approx = NeuralLorenzSDE(x0=x0, T=T, dt=dt,
                              obs_indices=obs_indices, obs_data=obs_data,
                              hidden_size_x=(80,80,80),
                              hidden_size_theta=(30,30,30), nlayers_theta=8)

dis_opt = tf.train.AdamOptimizer()
dis = DIS(model=lorenz_model, q=dis_approx, optimiser=dis_opt,
          importance_size=50000, ess_target=2500, max_weight=0.1)

start_time = time()

paths_toplot = 30
paths0 = np.zeros((paths_toplot,3,1))
for j in range(3):
    paths0[:,j,:] = x0[j]

i = 0
while dis.eps > 0.:
    dis.train(iterations=10)
    elapsed = time() - start_time
    print('Elapsed time (mins) {:.1f}'.format(elapsed/60.))
    # Plot some IS output
    output_sample = [b.numpy() for b in dis.batches]
    output_sample = np.vstack(output_sample)
    pars_sample = output_sample[:,0:4]
    df = pd.DataFrame(pars_sample,
                      columns=['theta1', 'theta2', 'theta3', 'obs noise scale'])
    plt.close(1)
    plt.close(2)
    scat = pd.plotting.scatter_matrix(df)
    # Show true parameter values
    # and extend axis to include them
    for j in range(4):
        scat[j,j].axvline(x=pars0[j], c='k')
        lims = list(scat[j,j].get_xlim())
        lims[0] = np.min((lims[0], pars0[j]))
        lims[1] = np.max((lims[1], pars0[j]))
        scat[j,j].set_xlim(lims)
    plt.savefig('Lorenz_ex1_pars{:d}.pdf'.format(i))
    plt.figure()
    paths = np.reshape(output_sample[0:paths_toplot, 4:], (paths_toplot, 3, -1),
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
    plt.savefig('Lorenz_ex1_paths{:d}.pdf'.format(i))

    plt.pause(0.1)
    i += 1

np.save('Lorenz_example1_pars', pars_sample)
wait = input('Press enter to terminate')
