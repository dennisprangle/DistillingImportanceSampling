import scipy.io
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

q = np.load('MG1_pars_N50000_frac0.05.npy')
dis = pd.DataFrame(q, columns=['arrival rate', 'min service', 'max service'])

d = scipy.io.loadmat('paper_1_1_1_16_1.mat')
p = d['par_mat']

p[:,1] += p[:,0]
p[:,2] = np.exp(p[:,2])

mcmc = pd.DataFrame(p, columns=['min service', 'max service', 'arrival rate'])
mcmc['iteration'] = range(100000)

# Trace plots
mcmc.plot(x='iteration', y='min service', kind='line')
mcmc.plot(x='iteration', y='max service', kind='line')
mcmc.plot(x='iteration', y='arrival rate', kind='line')

# Zooming in suggests we only need to discard about 1000 as burn-in
# (Also, longer chains might be needed to capture the tails)
mcmc = mcmc[1000:100000:10]

bins1 = np.arange(0., 0.2, 0.01)
bins2 = np.arange(2.01, 5.01, 0.2)
bins3 = np.arange(3.5, 8., 0.3)

f, axes = plt.subplots(2, 3)
sns.distplot(mcmc['arrival rate'], bins=bins1, kde=False, norm_hist=True,
             ax=axes[0, 0], axlabel=False)
sns.distplot(dis['arrival rate'], bins=bins1, kde=False, norm_hist=True,
             color='red', ax=axes[1, 0])
sns.distplot(mcmc['min service'], bins=bins2, kde=False, norm_hist=True,
             ax=axes[0, 1], axlabel=False)
sns.distplot(dis['min service'], bins=bins2, kde=False, norm_hist=True,
             color='red', ax=axes[1, 1])
sns.distplot(mcmc['max service'], bins=bins3, kde=False, norm_hist=True,
             ax=axes[0, 2], axlabel=False)
sns.distplot(dis['max service'], bins=bins3, kde=False, norm_hist=True,
             color='red', ax=axes[1, 2])
axes[0,0].set_ylim([0.,25.])
axes[1,0].set_ylim([0.,25.])
axes[0,1].set_ylim([0.,3.])
axes[1,1].set_ylim([0.,3.])
axes[0,2].set_ylim([0.,1.4])
axes[1,2].set_ylim([0.,1.4])
axes[0,0].set_ylabel('MCMC')
axes[1,0].set_ylabel('DIS')
axes[0,0].axvline(x=0.1, c='k')
axes[1,0].axvline(x=0.1, c='k')
axes[0,1].axvline(x=4., c='k')
axes[1,1].axvline(x=4., c='k')
axes[0,2].axvline(x=5., c='k')
axes[1,2].axvline(x=5., c='k')

f.tight_layout()

plt.savefig('MG1_post.pdf')
