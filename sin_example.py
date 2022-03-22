## Run analysis for the sin model
import torch
from nflows import distributions, flows, nn
from nflows.transforms import ReversePermutation, CompositeTransform
from nflows_mods import AffineCouplingTransform
from DIS import DIS
from models.sin import SinModel
from utils import resample
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

torch.manual_seed(111)

model = SinModel()

## Set up flow for approximate distribution
num_layers = 4
base_dist = distributions.StandardNormal(shape=[2])
transform = []

## This is taken from nflows examples
def create_net(in_features, out_features):
    return nn.nets.ResidualNet(
        in_features, out_features, hidden_features=10, num_blocks=10
   )

for _ in range(num_layers):
    transform.append(ReversePermutation(features=2))
    ## Next line uses an alternative flow layer
    ##transform.append(MaskedAffineAutoregressiveTransform(features=2,
    ##  hidden_features=6))
    tf = AffineCouplingTransform(
        mask=[1,0],
        transform_net_create_fn=create_net,
        scale_activation=AffineCouplingTransform.GENERAL_SCALE_ACTIVATION
    )
    transform.append(tf)

transform = CompositeTransform(transform)
approx_dist = flows.Flow(transform, base_dist)
optimizer = torch.optim.Adam(approx_dist.parameters())

dis = DIS(model, approx_dist, optimizer,
          importance_sample_size=4000, ess_target=2000, max_weight=0.01)

dis.pretrain(initial_target=model.initial_target, goal=0.5, report_every=10)

for i in range(20):
    ## Do some training
    dis.train(iterations=1)
    ## Plot some samples
    nplot = 300
    proposals = dis.train_sample.particles[0:nplot,:]
    selected = dis.train_sample.sample(nplot)
    plt.figure()
    plt.scatter(x=proposals[:,0], y=proposals[:,1], c="blue",
                alpha=0.6, marker="o", edgecolors="none")
    plt.scatter(x=selected[:,0], y=selected[:,1], c="red", marker="+")
    plt.xlim((-5.,5.))
    plt.ylim((-2.,2.))
    plt.title("Iteration {:d}, epsilon={:.3f}".format(dis.iterations_done,
                                                      dis.eps))
    plt.savefig("sin{:d}.pdf".format(dis.iterations_done))
    plt.pause(0.1)

wait = input("Press enter to terminate")
