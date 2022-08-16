## Run analysis for the sin model
import torch
from nflows import distributions, flows, nn
from nflows.transforms import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from DIS import DIS
from models.sin import SinModel
from utils import resample
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.rc('font', size=16)

torch.manual_seed(111)

model = SinModel()

## Set up flow for approximate distribution
num_layers = 4
base_dist = distributions.StandardNormal(shape=[2])
# transform = []

# ## This is taken from nflows examples
# def create_net(in_features, out_features):
#     return nn.nets.ResidualNet(
#         in_features, out_features, hidden_features=10, num_blocks=10
#    )

# for _ in range(num_layers):
#     transform.append(ReversePermutation(features=2))
#     ## Next line uses an alternative flow layer
#     ##transform.append(MaskedAffineAutoregressiveTransform(features=2,
#     ##  hidden_features=6))
#     tf = AffineCouplingTransform(
#         mask=[1,0],
#         transform_net_create_fn=create_net,
#         scale_activation=AffineCouplingTransform.GENERAL_SCALE_ACTIVATION
#     )
#     transform.append(tf)

# transform = CompositeTransform(transform)

transform = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
    features = 2,
    hidden_features = 20,
    num_bins = 5,
    tails = "linear",
    tail_bound = 10.,
    num_blocks = 3
)

approx_dist = flows.Flow(transform, base_dist)
optimizer = torch.optim.Adam(approx_dist.parameters())

dis = DIS(model, approx_dist, optimizer,
          importance_sample_size=4000, ess_target=2000, max_weight=0.1)

dis.pretrain(initial_target=model.prior, goal=0.5, report_every=10)

for i in range(6):
    ## Do some training
    dis.train(iterations=5)
    ## Plot some samples
    nplot = 300
    proposals = dis.train_sample.particles[0:nplot,:]
    selected = dis.is_sample[0:nplot,:]
    plt.figure()
    plt.scatter(x=proposals[:,0], y=proposals[:,1], c="blue",
                alpha=0.6, marker="o", edgecolors="none", label="proposal")
    plt.scatter(x=selected[:,0], y=selected[:,1], c="red", marker="+", label="target")
    plt.xlim((-4.5, 4.5))
    plt.ylim((-2.5, 2.5))
    plt.title(f"Iteration {dis.iterations_done:d}, epsilon={dis.eps:.3f}")
    if dis.iterations_done == 12: # Only need legend in one plot for final figure
        plt.legend(loc="lower right")
    plt.savefig("sin{:d}.pdf".format(dis.iterations_done))
    plt.pause(0.1)

wait = input("Press enter to terminate")
