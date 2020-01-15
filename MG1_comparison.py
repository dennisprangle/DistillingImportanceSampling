from DIS import DIS
from itertools import chain
from scipy.stats import norm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfe = tf.contrib.eager
tf.enable_eager_execution()

class MG1:
    """Class to encapsulate MG1 queue model and how it's tempered"""
    def __init__(self, interdeparture, nobs, nlatent):
        """
        `interdeparture` is observed interdeparture times
        `nobs` is number of observations
        """

        self.interdeparture = interdeparture
        self.nobs = nobs
        self.normCDF = tfb.NormalCDF()
        self.initial_target = tfd.Independent(
            tfd.Normal(loc=tf.zeros(nlatent), scale=tf.ones(nlatent)),
            reinterpreted_batch_ndims=1)
        self.max_eps = 10.

    def likelihood_prelims(self, inputs):
        """Store preliminary calculations required for evaluating likelihood

        `inputs` is a tensor of random variables
        `inputs[:,0:3]` control parameters
        `inputs[:,3:3+self.nobs]` control arrival times
        `inputs[:,3+T:3+2*self.nobs]` control service times
        """
        nbatches = inputs.shape[0]
        self.log_prior = self.initial_target.log_prob(inputs)
        ## Raw underlying U(0,1) variables
        inputs_u = self.normCDF.forward(inputs)
        ## Parameters
        arrival_rate = inputs_u[:,0] / 3.
        min_service = inputs_u[:,1] * 10.
        service_width = inputs_u[:,2] * 10.
        ## Arrival and service variables
        arrivals_u = inputs_u[:,3:3+self.nobs]
        services_u = inputs_u[:,3+self.nobs:3+2*self.nobs]
        arrivals = -tf.log(arrivals_u) / tf.reshape(arrival_rate, (nbatches, 1))
        services = tf.reshape(min_service, (nbatches, 1)) + \
                   services_u * tf.reshape(service_width, (nbatches, 1))
        # Compute interdeparture times
        departures = [] # Inter-departure times
        current_arrival = tf.zeros(nbatches) # Will be arrival time
                                             # for current customer
        last_departure = tf.zeros(nbatches)  # Will be departure time
                                             # for previous customer
                                             # (or zero if none)
        for i in range(self.nobs):
            current_arrival += arrivals[:,i]
            departures += [services[:,i] + \
                  tf.maximum(0., current_arrival - last_departure)]
            last_departure += departures[i]

        departures = tf.stack(departures)
        self.sq_dist = tf.math.reduce_sum(
            tf.pow(tf.reshape(self.interdeparture, (-1, 1)) - departures, 2.),
            0)

    def log_tempered_target(self, eps):
        """Calculate log of unnormalised tempered target density

        Requires likelihood_prelim to have already been run.
        """
        nbatches = self.sq_dist.shape[0]
        if (eps == 0.):
            log_obs = tf.where(self.sq_dist == 0., tf.zeros(nbatches),
                               tf.fill(nbatches, -np.infty))
        else:
            log_obs = -self.sq_dist / (2. * eps ** 2.)

        return self.log_prior + log_obs


def normals2queue(inputs, nobs):
    """Convert inputs to queue output

    If the inputs are N(0,1) draws, this outputs samples from prior + model.
    This is a numpy version of similar tensorflow code in `likelihood_prelims`.
        
    `inputs` is a matrix (rows are batches)
    `inputs[:,0:3]` control parameters
    `inputs[:,3:3+nobs]` control arrival times
    `inputs[:,3+T:3+2*nobs]` control service times

    Returns:

    `pars` a matrix, rows are batches, cols are parameters
    `departures` a matrix, rows are batches, cols are interdeparture times
    """

    nbatches = inputs.shape[0]
    ## Raw underlying U(0,1) variables
    inputs_u = norm.cdf(inputs)
    ## Parameters
    arrival_rate = inputs_u[:,0] / 3.
    min_service = inputs_u[:,1] * 10.
    service_width = inputs_u[:,2] * 10.
    pars = np.stack((arrival_rate, min_service, service_width), axis=1)
    ## Arrival and service variables
    arrivals_u = inputs_u[:,3:3+nobs]
    services_u = inputs_u[:,3+nobs:3+2*nobs]
    arrivals = -np.log(arrivals_u) / np.reshape(arrival_rate, (nbatches, 1))
    services = np.reshape(min_service, (nbatches, 1)) + \
               services_u * np.reshape(service_width, (nbatches, 1))
    # Compute interdeparture times
    departures = np.zeros((nbatches, nobs)) # Inter-departure times
    current_arrival = np.zeros(nbatches) # Will be arrival time
                                         # for current customer
    last_departure = np.zeros(nbatches)  # Will be departure time
                                         # for previous customer
                                         # (or zero if none)
    for i in range(nobs):
        current_arrival += arrivals[:,i]
        departures[:,i] = services[:,i] + \
                          np.maximum(0., current_arrival - last_departure)
        last_departure += departures[:,i]

    return pars, departures


def run_sim(is_size, ess_frac):
    ## See MG1_example.py for code to generate this synthetic data
    deps_obs = np.array(
    [4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
     19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
     17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
     11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566],
    dtype='float32')
    nobs = 20
    nlatent = 43

    ar_model = MG1(deps_obs, nobs, nlatent=nlatent)

    ## Approximating family
    hidden_size = (100, 100, 50)

    bichain=list(chain.from_iterable([
        tfb.Permute(np.random.permutation(nlatent)), # random permutation
        tfb.RealNVP(nlatent//2, shift_and_log_scale_fn =
            tfb.real_nvp_default_template(hidden_size, activation=tf.nn.elu,
              kernel_initializer=tf.initializers.truncated_normal(stddev=0.001)
            )),
    ] for _ in range(16)))

    bichain = bichain[1:] # remove final permutation
    bijector = tfb.Chain(bichain)

    base = tfd.MultivariateNormalDiag(loc=tf.zeros(nlatent))

    dis_approx = tfd.TransformedDistribution(base, bijector)
    dis_approx.sample() # Ensure variables created
    dis_opt = tf.train.AdamOptimizer()
    dis = DIS(model=ar_model, q=dis_approx, optimiser=dis_opt,
              importance_size=is_size, ess_target=is_size*ess_frac,
              max_weight=0.1, nbatches=10)

    while dis.elapsed < 60. * 180.: #stops shortly after 180 mins
        dis.train(iterations=50)

    # Save samples from approx posterior to a file
    x_sample = [b.numpy() for b in dis.batches]
    x_sample = np.vstack(x_sample)
    pars_samp, deps_samp = normals2queue(x_sample, nobs)
    pars_samp[:,2] += pars_samp[:,1]
    np.save('MG1_pars_N{:.0f}_frac{:.2f}'.format(is_size, ess_frac), pars_samp)

    # Return some summaries
    results = np.column_stack((np.array(dis.time_list), np.array(dis.eps_list),
                               np.array(dis.it_list)))
    results = np.insert(results, 3, is_size, 1)
    results = np.insert(results, 4, ess_frac, 1)
    results = pd.DataFrame(results,
                           columns=['time', 'eps', 'iteration',
                                    'is samples', 'ess frac'])
    return results
    

output = []
for is_size in (50000, 20000, 10000, 5000):
    for ess_frac in (0.05, 0.1, 0.2):
        try: # run_sim can fail due to numerical instability
            output += [run_sim(is_size, ess_frac)]
        except ValueError: 
            pass

output = pd.concat(output)
output['ess frac'] = output['ess frac'].astype('category')
output['is samples'] = output['is samples'].astype('int')
pd.to_pickle(output, "mg1_comparison.pkl")

## Need to specify palette explicitly for categorical variable
## See https://github.com/mwaskom/seaborn/issues/1515
pl1 = sns.relplot(x="time", y="eps", style="is samples", hue="ess frac",
                  kind="line", data=output, palette=['r','b','g'])
pl1.set(yscale="log")
pl1.set(xlabel="time (seconds)")
pl1.set(ylabel="epsilon")
pl1._legend.texts[0].set_text("M/N")
pl1._legend.texts[4].set_text("N")
plt.show(pl1)
pl1.savefig("MG1_comp.pdf")

pl2 = sns.relplot(x="iteration", y="eps", style="is samples", hue="ess frac",
                  kind="line", data=output, palette=['r','b','g'])

wait = input("Press enter to terminate")
