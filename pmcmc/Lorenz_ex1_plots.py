import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

pars0 = [10, 28, 8/3, 2]
parnames = ['theta1', 'theta2', 'theta3', 'sigma']

mcmc = pd.read_csv("Lorenz63_ex1_mcmc.csv", header=0, names=parnames)
mcmc = np.exp(mcmc)

# Trace plots
ax = mcmc.plot(subplots=True, layout=(2,2), legend=False, rasterized=True)
ax[0,0].set_ylabel(parnames[0])
ax[0,1].set_ylabel(parnames[1])
ax[1,0].set_ylabel(parnames[2])
ax[1,1].set_ylabel(parnames[3])
plt.tight_layout()

plt.savefig('Lorenz_ex1_PMCMC_trace.pdf')

## Zooming suggests no burn-in necessary
## But I'll thin output
mcmc = mcmc[0:80000:20]

# Pairs plots
scat = pd.plotting.scatter_matrix(mcmc, rasterized=True)
# Show true parameter values
# and extend axis to include them
for j in range(4):
    scat[j,j].axvline(x=pars0[j], c='k')

plt.tight_layout()

plt.savefig('Lorenz_ex1_PMCMC_pairs.pdf')

## Marginal histograms compared vis DIS
q = np.load('Lorenz_example1_pars.npy')
dis = pd.DataFrame(q, columns=parnames)

bins1 = np.arange(5., 15., 1.)
bins2 = np.arange(22., 32., 1.)
bins3 = np.arange(1.5, 4.2, 0.3)
bins4 = np.arange(0., 5., 0.5)

f, axes = plt.subplots(2, 4)
sns.distplot(mcmc['theta1'], bins=bins1, kde=False, norm_hist=True,
             ax=axes[0, 0], axlabel=False)
sns.distplot(dis['theta1'], bins=bins1, kde=False, norm_hist=True,
             color='red', ax=axes[1, 0])
sns.distplot(mcmc['theta2'], bins=bins2, kde=False, norm_hist=True,
             ax=axes[0, 1], axlabel=False)
sns.distplot(dis['theta2'], bins=bins2, kde=False, norm_hist=True,
             color='red', ax=axes[1, 1])
sns.distplot(mcmc['theta3'], bins=bins3, kde=False, norm_hist=True,
             ax=axes[0, 2], axlabel=False)
sns.distplot(dis['theta3'], bins=bins3, kde=False, norm_hist=True,
             color='red', ax=axes[1, 2])
sns.distplot(mcmc['sigma'], bins=bins4, kde=False, norm_hist=True,
             ax=axes[0, 3], axlabel=False)
sns.distplot(dis['sigma'], bins=bins4, kde=False, norm_hist=True,
             color='red', ax=axes[1, 3])
axes[0,0].set_ylim([0.,0.45])
axes[1,0].set_ylim([0.,0.45])
axes[0,1].set_ylim([0.,0.45])
axes[1,1].set_ylim([0.,0.45])
axes[0,2].set_ylim([0.,1.7])
axes[1,2].set_ylim([0.,1.7])
axes[0,3].set_ylim([0.,0.9])
axes[1,3].set_ylim([0.,0.9])
axes[0,0].set_ylabel('MCMC')
axes[1,0].set_ylabel('DIS')
for j in range(4):
    axes[0,j].axvline(x=pars0[j], c='k')
    axes[1,j].axvline(x=pars0[j], c='k')

f.tight_layout()

plt.savefig('Lorenz_ex1_marginals.pdf')
