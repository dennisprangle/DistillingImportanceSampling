from SDE import LorenzSDE
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfe = tf.contrib.eager
tf.enable_eager_execution()

## Generate a synthetic dataset
x0 = [-30., 0., 30.]
x0_tf = tf.constant(x0)
pars0 = np.array((10., 28., 8./3., 2.), dtype='float32')
theta = tfd.MultivariateNormalDiag(loc=pars0,
                                   scale_diag=[0., 0., 0., 0.])
T = 100
dt = 0.02
obs_indices = range(19,100,20) #i.e. 20,40,...,100 in 1-based indexing
nobs = len(obs_indices)
lorenz = LorenzSDE(x0_tf, T, dt, theta)
lorenz_samp = lorenz.sample(5)

x = np.reshape(lorenz_samp[0,4:], (3,-1), order='F')
plt.figure()
plt.plot(range(T), x[0,:], "r-")
plt.plot(range(T), x[1,:], "g-")
plt.plot(range(T), x[2,:], "b-")

y = x[:,obs_indices].transpose()
y += np.random.normal(scale=pars0[3], size=(nobs,3))
y = y.round(2)

np.save('Lorenz_data', (pars0, T, dt, x0, x, y, obs_indices))
