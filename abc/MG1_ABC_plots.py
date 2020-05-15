import scipy.io
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

abc_no_summaries = pd.read_pickle("MG1_ABC_sample_no_summaries.pkl")
abc_summaries = pd.read_pickle("MG1_ABC_sample_summaries.pkl")

d = scipy.io.loadmat('paper_1_1_1_16_1.mat')
p = d['par_mat']

p[:,1] += p[:,0]
p[:,2] = np.exp(p[:,2])

mcmc = pd.DataFrame(p, columns=['min service', 'max service', 'arrival rate'])
mcmc['iteration'] = range(100000)

#bins1 = np.arange(0., 0.2, 0.01)
#bins2 = np.arange(2.01, 5.01, 0.2)
#bins3 = np.arange(3.5, 8., 0.3)
bins1 = np.arange(0., 0.34, 0.02)
bins2 = np.arange(0., 10., 0.5)
bins3 = np.arange(0., 20., 1,)

f, axes = plt.subplots(3, 3)
sns.distplot(mcmc['arrival rate'], bins=bins1, kde=False, norm_hist=True,
             ax=axes[0, 0], axlabel=False)
sns.distplot(abc_no_summaries['arrival rate'], bins=bins1, kde=False,
             norm_hist=True, color='red', ax=axes[1, 0])
sns.distplot(abc_summaries['arrival rate'], bins=bins1, kde=False,
             norm_hist=True, color='green', ax=axes[2, 0])
sns.distplot(mcmc['min service'], bins=bins2, kde=False, norm_hist=True,
             ax=axes[0, 1], axlabel=False)
sns.distplot(abc_no_summaries['min service'], bins=bins2, kde=False,
             norm_hist=True, color='red', ax=axes[1, 1])
sns.distplot(abc_summaries['min service'], bins=bins2, kde=False,
             norm_hist=True, color='green', ax=axes[2, 1])
sns.distplot(mcmc['max service'], bins=bins3, kde=False, norm_hist=True,
             ax=axes[0, 2], axlabel=False)
sns.distplot(abc_no_summaries['max service'], bins=bins3, kde=False,
             norm_hist=True, color='red', ax=axes[1, 2])
sns.distplot(abc_summaries['max service'], bins=bins3, kde=False,
             norm_hist=True, color='green', ax=axes[2, 2])
ylims = [20., 2.0, 0.6]
pars_true = [0.1, 4., 5.]
for row in range(3):
    for col in range(3):
        axes[row,col].set_ylim([0., ylims[col]])
        axes[row,col].axvline(x=pars_true[col], c='k')

        
axes[0,0].set_ylabel('MCMC')
axes[1,0].set_ylabel('ABC no summaries')
axes[2,0].set_ylabel('ABC summaries')

f.tight_layout()

plt.savefig('ABC_post.pdf')
