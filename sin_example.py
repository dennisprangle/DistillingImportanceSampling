from DIS import DIS
from itertools import chain
from scipy.stats import norm
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

class SinModel:
    """Class to encapsulate sin model and how it's tempered"""
    def __init__(self):
        """Initialise function - no parameters required!
        """
        self.initial_target = tfd.Independent(
            tfd.Normal(loc=[0.,0.], scale=[2.,2.]),
            reinterpreted_batch_ndims=1)
        self.max_eps = 1.

    def likelihood_prelims(self, inputs):
        """Store preliminary calculations required for evaluating likelihood

        `inputs` is a [:,2] tensor of parameters
        """
        nbatches = inputs.shape[0]
        th1, th2 = tf.split(inputs, [1,1], -1)
        th1 = tf.squeeze(th1)
        th2 = tf.squeeze(th2)        
        self.initial_log_density = self.initial_target.log_prob(inputs)
        self.log_target = -100. * tf.pow((th2 - tf.sin(th1)), 2)
        self.log_target += tf.where(tf.abs(th1) < np.pi, tf.zeros(nbatches), tf.fill([nbatches,], -np.infty))

    def log_tempered_target(self, eps):
        """Calculate log of unnormalised tempered target density

        Requires likelihood_prelim to have already been run.
        """
        return eps * self.initial_log_density + (1. - eps) * self.log_target

model = SinModel()
## Approximating family
hidden_size = (10, 10, 10)

bichain=list(chain.from_iterable([
    tfb.Permute(np.flipud(np.arange(2))), # reverse permutation
    tfb.RealNVP(1, shift_and_log_scale_fn =
        tfb.real_nvp_default_template(hidden_size, activation=tf.nn.elu,
            kernel_initializer=tf.initializers.truncated_normal(stddev=0.001)
        ))
    ] for _ in range(4)))

bichain = bichain[1:] # remove final permutation
bijector = tfb.Chain(bichain)
base = tfd.MultivariateNormalDiag(loc=tf.zeros(2))
dis_approx = tfd.TransformedDistribution(base, bijector)
dis_approx.sample() # force variable creation

dis_opt = tf.train.AdamOptimizer()
dis = DIS(model=model, q=dis_approx, optimiser=dis_opt,
          importance_size=4000, ess_target=2000, max_weight=0.1)

dis.pretrain(0.5, report_every=10)

for i in range(24):
    dis.train(iterations=5)
    # Plot some samples
    nplot = 300
    q_samp = dis_approx.sample(nplot).numpy()
    log_q = dis_approx.log_prob(q_samp).numpy()
    model.likelihood_prelims(q_samp)
    log_w = model.log_tempered_target(dis.eps) - log_q
    w = np.exp(log_w)
    w /= sum(w)
    selected = np.full(nplot, False)
    ind = np.random.choice(nplot, size=nplot//2, p=w) # samples with replacement
    selected[ind] = True
    unselected = np.logical_not(selected)
    plt.figure()
    plt.scatter(x=q_samp[unselected,0], y=q_samp[unselected,1], c="blue",
                alpha=0.6, marker="o", edgecolors="none")
    plt.scatter(x=q_samp[selected,0], y=q_samp[selected,1], c="red",
                marker="+", edgecolors="none")
    plt.xlim((-5.,5.))
    plt.ylim((-2.,2.))
    plt.title("Iteration {:d}, epsilon={:.3f}".format(dis.t, dis.eps))
    plt.savefig("sin{:d}.pdf".format(dis.t))
    plt.pause(0.1)

wait = input("Press enter to terminate")
