import torch
from models.sde import LorenzSDE
import pickle
import matplotlib.pyplot as plt
plt.ion()

torch.manual_seed(0)

class Delta:
    def __init__(self, x):
        self.x = x
    def sample(self, nreps):
        return torch.tile(self.x, [nreps, 1])
    def log_prob(self, x):
        if len(x.shape) == 1:        
            return float(torch.equal(x, self.x))
        else:
            return float(torch.eq(x, self.x.unsqueeze(1)).numpy())

## Generate a synthetic dataset
x0 = torch.tensor([-30., 0., 30.])
pars0 = torch.tensor([10., 28., 8./3., 2.])
theta_dist = Delta(pars0)
T = 100
dt = 0.02
obs_indices = range(19,100,20) #i.e. 20,40,...,100 in 1-based indexing
nobs = len(obs_indices)
lorenz = LorenzSDE(x0, T, dt, theta_dist)
lorenz_samp = lorenz.sample(1)
xflat = lorenz_samp[0,4:]
xcols = [ xflat[i::3] for i in range(3) ]
x = torch.stack(xcols, dim=0)

# Plot data to check it's sensible (when run interactively)
plt.figure()
plt.plot(range(T), x[0,:], "r-")
plt.plot(range(T), x[1,:], "g-")
plt.plot(range(T), x[2,:], "b-")

y = x[:,obs_indices].transpose(0,1)
y = y + torch.randn([nobs,3])
y = (100*y).round() / 100

output = {
    'pars0' : pars0,
    'T' : T,
    'dt' : dt,
    'x0' : x0,
    'x' : x,
    'y' : y,
    'obs_indices' : obs_indices
}

with open('Lorenz_data.pkl', 'wb') as outfile:
    pickle.dump(output, outfile)
