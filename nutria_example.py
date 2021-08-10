from DIS import DIS
from nutria_functions import RickerModel, RickerApprox
from time import time
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

## Data taken from `particles` package
obs = [0.55, 0.55, 0.6 , 0.55, 0.5 , 0.5 , 0.55, 0.6 , 0.7 , 0.8 , 0.7 ,
       0.75, 0.75, 0.7 , 0.7 , 0.7 , 0.65, 0.7 , 0.9 , 0.9 , 1.  , 1.1 ,
       1.  , 0.9 , 0.9 , 0.95, 0.95, 0.9 , 0.95, 1.05, 1.15, 1.2 , 1.6 ,
       1.7 , 1.7 , 1.85, 1.95, 2.  , 2.2 , 2.45, 2.7 , 2.9 , 3.1 , 3.05,
       3.4 , 3.9 , 3.5 , 3.35, 3.3 , 2.7 , 2.5 , 2.55, 2.55, 2.5 , 2.7 ,
       2.75, 2.9 , 3.45, 3.2 , 3.05, 3.2 , 3.05, 2.9 , 3.45, 3.5 , 4.  ,
       4.3 , 4.3 , 5.  , 5.55, 4.85, 4.5 , 4.3 , 4.2 , 4.  , 4.15, 4.2 ,
       3.85, 3.55, 3.6 , 3.7 , 3.65, 3.  , 2.9 , 3.  , 2.95, 3.05, 3.35,
       3.5 , 3.6 , 4.  , 3.9 , 4.1 , 4.1 , 3.35, 3.15, 3.35, 3.25, 3.25,
       3.3 , 3.15, 3.1 , 3.6 , 3.7 , 4.05, 4.  , 3.8 , 2.3 , 2.15, 2.1 ,
       2.1 , 2.1 , 2.  , 2.1 , 2.2 , 2.3 , 2.95, 3.2 , 2.6 , 2.65]
T = len(obs)
x0 = obs[0]
obs = np.array(obs) * 1000.

##Prior for theta is 2 independent N(0,1)s
prior = tfd.Independent(
    tfd.Normal(tf.zeros(2), tf.ones(2)),
    reinterpreted_batch_ndims=1)

ricker_model = RickerModel(x0=x0, T=T, obs=obs, prior=prior)

## Approximating family
dis_approx = RickerApprox(x0=x0, T=T, obs=obs,
                          hidden_size_x=(30,30,30),
                          hidden_size_theta=(20,20,20), flow_layers=8)

## Next line has version for testing during development
##dis = DIS(model=ricker_model, q=dis_approx,
##          importance_size=2000, ess_target=400, max_weight=0.1,
##          reset_optimizer=True)
## Next line has tuning suggested from paper
dis = DIS(model=ricker_model, q=dis_approx,
          importance_size=50000, ess_target=2500, max_weight=0.1,
          reset_optimizer=True)

start_time = time()

paths_toplot = 30

i = 0
while dis.eps > 0.:
    dis.train(iterations=5)
    elapsed = time() - start_time
    print('Elapsed time (mins) {:.1f}'.format(elapsed/60.))
    # Plot some IS output
    output_sample = [b.numpy() for b in dis.batches]
    output_sample = np.vstack(output_sample)
    pars_sample = output_sample[:,0:2]
    df = pd.DataFrame(pars_sample,
                      columns=['theta1', 'theta2'])
    plt.close(1)
    plt.close(2)
    scat = pd.plotting.scatter_matrix(df)
    plt.savefig('nutria_ex_pars{:d}.pdf'.format(i))
    plt.figure()
    paths = output_sample[0:paths_toplot, 2:] * 1000.

    for j in range(paths_toplot):
        plt.plot(range(T), paths[j,:], '-', alpha=0.3)
    plt.plot(range(T), obs[:], 'o')
    plt.xlabel('t')
    plt.savefig('nutria_example_paths{:d}.pdf'.format(i))

    plt.pause(0.1)
    i += 1

np.save('nutria_example_pars', pars_sample)
wait = input('Press enter to terminate')
