from DIS import DIS
import torch
from models.MG1 import MG1Model
import scipy.io
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
plt.ion()

# Prepare MCMC output
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

# Get fitted DIS proposal and use it for importance sampling
with open(f'MG1_dist_N5000_frac0.05.pkl', 'rb') as infile:
    approx_dist = pickle.load(infile)

comparison_summary = pd.read_pickle('mg1_comparison.pkl')
subset = (comparison_summary["is samples"] == 5000) & (comparison_summary["ess frac"] == 0.05)
eps = comparison_summary[subset]["eps"].min()

obs = torch.tensor(
    [4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
     19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
     17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
     11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566])
ninputs = 43

model = MG1Model(obs)

dis_obj = DIS(model, approx_dist, None,
          importance_sample_size=5000,
          ess_target=250, max_weight=0.1)

torch.manual_seed(1)
with torch.no_grad():
    weighted_params = dis_obj.get_sample(750000) # Limited by my PC's memory
weighted_params.update_epsilon(eps)
params = weighted_params.sample(10000).detach()
arrival_rate, min_service, service_width, _, _ = model.convert_inputs(params)
max_service = min_service + service_width
pars_samp = torch.stack([arrival_rate, min_service, max_service], axis=1)
dis = pd.DataFrame(pars_samp, columns=['arrival rate', 'min service', 'max service'])
dis.to_pickle('MG1_DIS_sample.pkl')

# Plot posterior histograms
bins1 = np.linspace(0., 0.2, 20)
bins2 = np.linspace(0., 6., 20)
bins3 = np.linspace(0., 10., 20)

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
