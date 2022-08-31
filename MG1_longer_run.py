## Run analysis for the MG1 model
import torch
from nflows import distributions, flows, nn
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms import RandomPermutation, CompositeTransform
from nflows_mods import AffineCouplingTransform
from DIS import DIS
from utils import resample
from models.MG1 import MG1Model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()

torch.manual_seed(111)

is_size = 5000
ess_frac = 0.05

## Synthetic observations used in paper
## (originally generated using earlier tensorflow code)
obs = torch.tensor(
    [4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
     19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
     17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
     11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566])
ninputs = 43

model = MG1Model(obs)

## Set up flow for approximate distribution
base_dist = distributions.StandardNormal(shape=[ninputs])

## Spline flow
transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
    features = ninputs,
    hidden_features = 20,
    num_bins = 5,
    tails = "linear",
    tail_bound = 10.,
    num_blocks = 3
)
## End of spline flow

## Real NVP flow
# mask = torch.cat([torch.ones(21), torch.zeros(22)], dim=0)
# num_layers = 16
# transform = []

# def create_net(in_features, out_features):
#     return nn.nets.ResidualNet(
#         in_features, out_features, hidden_features=20, num_blocks=10
#     )
    
# for _ in range(num_layers):
#     transform.append(RandomPermutation(features=ninputs))
#     tf = AffineCouplingTransform(
#         mask=mask,
#         transform_net_create_fn=create_net,
#         scale_activation=lambda x : (torch.exp(x) + 1e-3).clamp(0, 3)
#     )
#     transform.append(tf)

# transform = CompositeTransform(transform)
## End of real NVP flow


approx_dist = flows.Flow(transform, base_dist)

optimizer = torch.optim.Adam(approx_dist.parameters())

dis = DIS(model, approx_dist, optimizer,
          importance_sample_size=is_size,
          ess_target=is_size*ess_frac, max_weight=0.1)

dis.pretrain(initial_target=model.prior, goal=0.5, report_every=10)

mins_to_run = 60.
while dis.elapsed_time < 60. * mins_to_run: #stop shortly after specified time
    dis.train(iterations=1)

# Plot results
params = dis.is_sample
arrival_rate, min_service, service_width, _, _ = model.convert_inputs(params)
max_service = min_service + service_width
pars_samp = torch.stack([arrival_rate, min_service, max_service], axis=1)
dis2 = pd.DataFrame(pars_samp, columns=['arrival rate', 'min service', 'max service'])

sns.pairplot(dis2)
sns.pairplot(pd.DataFrame(params[:,0:6]))
# _, axes = plt.subplots(1, 3)
# sns.distplot(dis2['arrival rate'], kde=False, norm_hist=True,
#              color='red', ax=axes[0])
# sns.distplot(dis2['min service'], kde=False, norm_hist=True,
#              color='red', ax=axes[1])
# sns.distplot(dis2['max service'], kde=False, norm_hist=True,
#             color='red', ax=axes[2])

# axes[0].axvline(x=0.1, c='k')
# axes[1].axvline(x=4., c='k')
# axes[2].axvline(x=5., c='k')

plt.show()
plt.pause(0.1)

wait = input("Press enter to terminate")