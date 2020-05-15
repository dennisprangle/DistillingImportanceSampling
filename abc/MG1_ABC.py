import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import sys

plt.ion()

def sample_prior(n):
    """Return matrix of `n` samples from the prior"""
    theta = np.random.uniform(size=(n,3))
    theta[:,0] /= 3.
    theta[:,1] *= 10.
    theta[:,2] *= 10.
    theta[:,2] += theta[:,1]
    return theta

def simulate_sq_dist(deps_obs, theta, use_summaries=False):
    """Simulate interdeparture times for M/G/1 queues
    and return squared distances to `deps_obs`

    `deps_obs` - observed inter-departure times
    `theta` - matrix of parameters
    `use_summaries` - if true use quartile summaries
    """
    nobs = len(deps_obs)
    nsims = theta.shape[0]

    arrival_rate = theta[:,0]
    min_service = theta[:,1]
    service_width = theta[:,2] - theta[:,1]
    
    arrivals = np.random.exponential(scale=1., size=(nsims, nobs))
    arrivals = arrivals / np.reshape(arrival_rate, (nsims, 1))
    services = np.random.uniform(size=(nsims, nobs))
    services = np.reshape(min_service, (nsims, 1)) + \
               services * np.reshape(service_width, (nsims, 1))

    departures = np.zeros((nsims, nobs)) # Inter-departure times
    current_arrival = np.zeros(nsims)    # Will be arrival time
                                           # for current customer
    last_departure = np.zeros(nsims)     # Will be departure time
                                           # for previous customer
                                           # (or zero if none)
    for i in range(nobs):
        current_arrival += arrivals[:,i]
        departures[:,i] = services[:,i] + \
                          np.maximum(0., current_arrival - last_departure)
        last_departure += departures[:,i]

    if use_summaries:
        q = np.linspace(0.,4.,5) / 4.
        obs_quantiles = np.quantile(deps_obs, q)
        sim_quantiles = np.quantile(departures, q, axis=1).T
        diff = sim_quantiles - np.reshape(obs_quantiles, (1,-1))
    else:
        diff = departures - np.reshape(deps_obs, (1,-1))

    sq_dist = np.sum(diff**2., axis=1)

    return sq_dist


# def get_ess_factor(sq_dist, w, eps_old, eps_new):
#     """Relative change in ESS from reducing eps"""
#     ess_old = (sum(w)**2.)/sum(w**2.)
#     abc_w_old = np.exp(-0.5*sq_dist/(eps_old**2.))
#     abc_w_new = np.exp(-0.5*sq_dist/(eps_new**2.))
#     w_new = w * abc_w_new / abc_w_old
#     ess_new = (sum(w_new)**2.)/sum(w_new**2.)
#     return ess_new / ess_old


# def get_prob_factor(sq_dist, w, eps_old, eps_new):
#     """Relative probability of acceptance for a representative simulation"""
#     med_sq_dist = np.quantile(sq_dist, 0.5)
#     acc_prob_old = np.exp(-0.5*med_sq_dist/(eps_old**2.))
#     acc_prob_new = np.exp(-0.5*med_sq_dist/(eps_new**2.))
#     return acc_prob_new / acc_prob_old

 
# def get_eps(sq_dist, w, eps_old, prob_fraction=0.8, lower=0.001, upper=1E6,
#             max_iterations=100):
#     """Find epsilon value below current giving required reduction in acceptance probability"""
#     upper = np.minimum(eps_old, upper)
#     eps = (lower + upper) / 2.
#     for i in range(max_iterations):
#         factor = get_prob_factor(sq_dist, w, eps_old, eps)
#         if factor > prob_fraction:
#             upper = eps
#         else:
#             lower = eps
#         eps = (lower + upper) / 2.

#     return eps


def get_eps(sq_dist, w, eps_old, prob_fraction=0.8):
    """Find epsilon value giving required typical reduction in acceptance probability"""
    med_sq_dist = np.quantile(sq_dist, 0.5)
    eps = eps_old ** -2.
    eps -= 2. * np.log(prob_fraction) / med_sq_dist
    eps = eps ** -0.5
    return eps


def proposal_density(theta, last_theta, last_w, cov_matrix):
    """Calculate proposal densities

    `theta` - proposed parameters
    `last_theta` - parameters used to generated proposals
    `last_w` - weights used to generate proposals
    `cov_matrix` - noise covariance used to generate proposals
    """
    n_new = theta.shape[0]
    n_old = last_theta.shape[0]
    var = multivariate_normal(mean=np.zeros(3), cov=cov_matrix)
    d = [[last_w[j]*var.pdf(theta[i,:]-last_theta[j,:])
         for i in range(n_new)] for j in range(n_old)]
    d = np.array(d)
    return np.sum(d, axis=0)


def is_feasible(theta):
    """Returns logical vector of which rows of `theta` are supported by the prior
    """
    service_width = theta[:,2] - theta[:,1]
    feasible = (theta[:,0] >= 0.)
    feasible = np.logical_and(feasible, theta[:,1] >= 0.)
    feasible = np.logical_and(feasible, service_width >= 0.)    
    feasible = np.logical_and(feasible, theta[:,0] <= 1./3.)
    feasible = np.logical_and(feasible, theta[:,1] <= 10.)
    feasible = np.logical_and(feasible, service_width <= 10.)
    return feasible

def sample_abc_posterior(n, eps, deps_obs, batch_size=1000, use_prior=True,
                         last_theta=None, last_w=None, use_summaries=False):
    """Returns weighted samples from the ABC posterior

    `n` - how many samples to return
    `eps` - ABC bandwidth
    `dep_obs` - observed inter-departure times
    `batch_size` - batch size for vectorised simulations
    `use_prior` - if True sample parameters from the prior
    `last_theta` - previous theta sample, used when `use_prior==False`
    `last_w` - previous weights, used when `use_prior==False`
    """
    sims_count = 0
    if use_prior==False:
        cov_matrix = 2. * np.cov(last_theta, rowvar=False, aweights=last_w)

    ##Sample until we have n acceptances
    theta_accepted = np.zeros((n,3))
    sq_dist_accepted = np.zeros(n)
    naccepted = 0
    while(naccepted < n):
        ##Sample a batch
        if use_prior:
            theta = sample_prior(batch_size)
            new_sims = batch_size
        else:
            ancestors = np.random.choice(n, size=batch_size, replace=True,
                                         p=last_w)
            theta = last_theta[ancestors,:]
            theta += np.random.multivariate_normal(mean=np.zeros(3),
                                                   cov=cov_matrix,
                                                   size=batch_size)
            feasible = is_feasible(theta)
            theta = theta[feasible == True,:]
            new_sims = sum(feasible)
            if new_sims == 0:
                continue

        ##Accept/reject (using normal kernel to get probabilities)
        sq_dist = simulate_sq_dist(deps_obs, theta, use_summaries)
        sims_count += new_sims
        w = np.exp(-0.5*sq_dist/(eps**2.))
        acc = np.random.binomial(1,w)

        ##Record acceptances, discarding any excess
        nacc = sum(acc)
        if sum(acc) == 0:
            continue
        theta = theta[acc==1,:]
        sq_dist = sq_dist[acc==1]
        toadd = np.minimum(nacc, n - naccepted)
        theta_accepted[naccepted:(naccepted+toadd),:] = theta[0:toadd,:]
        sq_dist_accepted[naccepted:(naccepted+toadd)] = sq_dist[0:toadd]
        naccepted += toadd

    ##Calculate normalised weights
    if use_prior:
        w = np.ones(n)
    else:
        ## Unnormalised weight should be proportional to
        ## prior density / proposal density
        ## and our prior is uniform
        w = 1. / proposal_density(theta_accepted, last_theta, last_w, cov_matrix)

    w /= np.sum(w)
        
    return(theta_accepted, sq_dist_accepted, w, sims_count)


# def weighted_quantile(x, w, q):
#     """Returns quantile q of values x under weights w

#     (based on https://stackoverflow.com/a/29677616)"""
#     sorter = np.argsort(x)
#     x = x[sorter]
#     w = w[sorter]

#     wq = np.cumsum(w) - 0.5 * w
#     return np.interp(q, wq, x)



def abc_pmc(deps_obs, n_to_accept, prob_fraction, max_sims=10**6,
            batch_size=1000, use_summaries=False):
    """Run ABC PMC
    `dep_obs` - observed inter-departure times
    `n_to_accept` - how many ABC acceptances are required in each iteration
    `prob_fraction` - used in updating epsilon
    `max_sims` - maximum number of ABC simulations to perform
    `batch_size` - batch size for vectorised simulations

    Returns a data frame of a sample from the final iteration
    (based on importance resampling)
    """
    start_time = time()
    iteration = 1
    eps = np.inf
    first_iteration = True
    total_sims = 0
    theta = None
    w = None
    while total_sims < max_sims:
        print('Iteration {:d}, epsilon {:.2f}'.format(iteration, eps))
        (theta, sq_dist, w, sims) \
            = sample_abc_posterior(n_to_accept, eps, deps_obs,
                                   batch_size=batch_size,
                                   use_prior=first_iteration,
                                   last_theta=theta,
                                   last_w=w,
                                   use_summaries=use_summaries)
        total_sims += sims
        ess = (sum(w)**2.)/sum(w**2.)
        print('ESS {:.0f}, simulations {:d}, progress {:.1f}%, minutes {:.1f}'.\
              format(ess, total_sims, 100.*total_sims/max_sims,
                     (time()-start_time)/60.))
        sys.stdout.flush()
        eps = get_eps(sq_dist, w, eps, prob_fraction)
        iteration += 1
        first_iteration = False

        ancestors = np.random.choice(theta.shape[0], size=200, replace=True, p=w)
        reparam = theta[ancestors,:]
        reparam[:,2] += reparam[:,1]
        plt.close()
        df = pd.DataFrame(reparam,
                          columns=['arrival rate', 'min service', 'max service'])
        f, axes = plt.subplots(1, 3)
        sns.distplot(df['arrival rate'], kde=False, norm_hist=True, ax=axes[0])
        sns.distplot(df['min service'], kde=False, norm_hist=True, ax=axes[1])
        sns.distplot(df['max service'], kde=False, norm_hist=True, ax=axes[2])
        axes[0].axvline(x=0.1, c='k')
        axes[1].axvline(x=4., c='k')
        axes[2].axvline(x=5., c='k')
        f.tight_layout()
        plt.pause(1)

    return df

deps_obs = np.array(
    [4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
    19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
    17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
    11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566])

nsims = 1.65*(10**8)
df_no_summaries = abc_pmc(deps_obs, n_to_accept=500, prob_fraction=0.7,
                          max_sims=nsims, batch_size=10**5,
                          use_summaries=False)
df_no_summaries.to_pickle("MG1_ABC_sample_no_summaries.pkl")
df_summaries = abc_pmc(deps_obs, n_to_accept=500, prob_fraction=0.7,
                       max_sims=nsims, batch_size=10**5,
                       use_summaries=True)
df_summaries.to_pickle("MG1_ABC_sample_summaries.pkl")
