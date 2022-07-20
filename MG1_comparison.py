## Run analysis for the MG1 model
import torch
from nflows import distributions, flows, nn
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

## Initialise weights, following https://stackoverflow.com/a/49433937
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.)

def run_sim(is_size, ess_frac):
    ## Synthetic observations used in paper
    ## (originally generated using earlier tensorflow code)
    obs = torch.tensor(
        [4.67931388, 33.32367159, 16.1354178 ,  4.26184914, 21.51870177,
         19.26768645, 17.41684327,  4.39394293,  4.98717158,  4.00745068,
         17.13184198,  4.64447435, 12.10859597,  6.86436748,  4.199275  ,
         11.70312317,  7.06592802, 16.28106949,  8.66159665,  4.33875566])
    ninputs = 43
    mask = torch.cat([torch.ones(21), torch.zeros(22)], dim=0)

    model = MG1Model(obs)

    ## Set up flow for approximate distribution
    num_layers = 16
    base_dist = distributions.StandardNormal(shape=[ninputs])
    transform = []

    def create_net(in_features, out_features):
        return nn.nets.ResidualNet(
            in_features, out_features, hidden_features=20, num_blocks=10
        )
    
    for _ in range(num_layers):
        transform.append(RandomPermutation(features=ninputs))
        tf = AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_net,
            scale_activation=lambda x : (torch.exp(x) + 1e-3).clamp(0, 3)
        )
        transform.append(tf)

    transform = CompositeTransform(transform)
    approx_dist = flows.Flow(transform, base_dist)

    approx_dist.apply(init_weights)

    optimizer = torch.optim.Adam(approx_dist.parameters())

    dis = DIS(model, approx_dist, optimizer,
              importance_sample_size=is_size,
              ess_target=is_size*ess_frac, max_weight=0.01)

    _, axes = plt.subplots(1,3)

    while dis.elapsed_time < 60. * 60.: #stop shortly after 60 mins
       dis.train(iterations=50)

    ## Save samples from approx posterior to a file
    params = dis.train_sample.sample(1000).detach()
    arrival_rate, min_service, service_width, _, _ = model.convert_inputs(params)
    max_service = min_service + service_width
    pars_samp = torch.stack([arrival_rate, min_service, max_service], axis=1)

    np.save(f'MG1_pars_N{is_size:.0f}_frac{ess_frac:.2f}', pars_samp.numpy())

    # Return some summaries    
    results = dis.get_history_array()
    results = np.insert(results, 3, is_size, 1)
    results = np.insert(results, 4, ess_frac, 1)
    results = pd.DataFrame(results,
                           columns=['time', 'eps', 'iteration',
                                    'is samples', 'ess frac'])
    return results

torch.manual_seed(111)
output = []
# Values for main comparison
for is_size in (50000, 20000, 10000, 5000):
    for ess_frac in (0.05, 0.1, 0.2):
        try: # run_sim can fail due to numerical instability
            output.append(run_sim(is_size, ess_frac))
        except:
            pass 

output = pd.concat(output, ignore_index=True)
output['ess frac'] = output['ess frac'].astype('category')
output['is samples'] = output['is samples'].astype('int')
pd.to_pickle(output, "mg1_comparison.pkl")

pl1 = sns.relplot(x="time", y="eps", style="is samples", hue="ess frac",
                  kind="line",
                  palette=sns.color_palette("rocket", n_colors=3), # A palette that's ok in black-and-white
                  data=output)

pl1.set(yscale="log")
pl1.set(xlabel="time (seconds)")
pl1.set(ylabel="epsilon")
pl1._legend.texts[0].set_text("M/N")
pl1._legend.texts[4].set_text("N")
plt.show()
pl1.savefig("MG1_comp.pdf")

#pl2 = sns.relplot(x="iteration", y="eps", style="is samples", hue="ess frac",
#                  data=output)
