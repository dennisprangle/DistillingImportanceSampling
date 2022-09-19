import torch
import numpy as np
from utils import resample, effective_sample_size
from models.models import SimulatorModel
from random import shuffle
from time import time
from copy import deepcopy

class DIS:
    def __init__(self, model, approx_dist, optimizer, importance_sample_size,
                 ess_target, max_bisection_its=50, batch_size=100,
                 max_weight=1., nbatches=None):
       """Class to perform a distilled importance sampling analysis

       `model` a `SimulatorModel` object encapsulating model and prior
       `approx_dist` is a trainable distribution which will approximate the target. Any object with `sample` and `log_prob` methods can be used. This allows the user to choose between various libraries for distributions.
       `optimizer` is a torch `Optimizer` object for the variables used in `approx_dist`
       `importance_sample_size` is number of importance samples (`N` in paper)
       `ess_target` is target effective sample size (`M` in paper)
       `max_bisection_its` controls the ESS selection algorithm
       `batch_size` is training batch size (`n` in paper)
       `max_weight` is the maximum normalised weight allowed after clipping (omega in paper)
       `nbatches` is how many training batches to create (`B` in paper). Defaults to `ess_target` / `batch_size` (rounded).
       """
       self.start_time = time()
       self.elapsed_time = 0.
       self.model = model
       self.approx_dist = approx_dist
       self.optimizer = optimizer
       self.importance_sample_size = importance_sample_size
       self.ess_target = ess_target
       self.max_bisection_its = max_bisection_its
       self.batch_size = batch_size
       self.max_weight = max_weight
       if nbatches is None:
           self.nbatches = np.ceil(ess_target / batch_size).astype('int')
       else:
           self.nbatches = nbatches
       self.iterations_done = 0
       self.eps = np.infty
       self.history = {
           'elapsed_time':[],
           'epsilon':[],
           'iterations_done':[]
       }
       self.ess = 0.

    def get_history_array(self):
        # Uses same format as code for earlier paper versions
        return np.column_stack(
            [np.array(self.history['elapsed_time']),
             np.array(self.history['epsilon']),
             np.array(self.history['iterations_done'])]
        )

    def get_sample(self, size=None):
        """Get a `WeightedSample` from the model using the current proposal"""
        if size is None:
            size = self.importance_sample_size
        particles = self.approx_dist.sample(size).detach()
        log_proposal = self.approx_dist.log_prob(particles).detach()
        return self.model.run(particles, log_proposal)

    def get_loss(self, params):
        """Calculate loss under parameters from current target distribution"""
        return -torch.sum(self.approx_dist.log_prob(params))

    def pretrain(self, initial_target, goal=0.75, report_every=100):
        """Train approximation to match an initial target

        `initial_target` distribution. Any object with `log_prob` and `sample` method can be used. This allows the user to choose between various libraries for distributions.
        `goal` Target value for ESS / actual sample size. Pretraining will continue until this is reached.
        `report_every` Report training progress on every multiple of this many iterations."""
        ess = 0.
        pretraining_iterations_done = 0
        while ess < self.batch_size * goal:
            for i in range(report_every):
                params = initial_target.sample([self.batch_size,])
                loss = self.get_loss(params)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            pretraining_iterations_done += report_every
            params = self.approx_dist.sample(self.batch_size).detach()
            logw = initial_target.log_prob(params)
            logw -= self.approx_dist.log_prob(params)
            ess = effective_sample_size(logw, log_input=True).item()
            print(
                f"Pretraining iterations {pretraining_iterations_done:2d}, "
                f"loss {loss:.1f}, "
                f"ESS (from 100 samples) {ess:.1f}"
            )

    def train(self, iterations=10):
        """Train approximation using DIS algorithm from original paper

        `iterations` is how many iterations to perform.
        Training can then be continued by calling `train` again."""
        for i in range(iterations):
            with torch.no_grad():
                self.train_sample = self.get_sample()
                new_eps = self.train_sample.find_eps(self.ess_target, self.eps, self.max_bisection_its)
                self.train_sample.update_epsilon(new_eps)
                self.ess = effective_sample_size(self.train_sample.weights)
                self.eps = new_eps
                self.is_sample = self.train_sample.sample(1000).detach() # Useful for plots
                S = self.train_sample.truncate_weights(self.max_weight)
                total_loss = 0.
            for _ in range(self.nbatches):
                batch = self.train_sample.sample(self.batch_size).detach()
                loss = S * self.get_loss(batch)
                total_loss += loss.detach()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.iterations_done += 1
            self.elapsed_time = time() - self.start_time
            self.history['elapsed_time'].append(self.elapsed_time)
            self.history['epsilon'].append(self.eps)
            self.history['iterations_done'].append(self.iterations_done)

            # Report status
            print(
                f"Iteration {self.iterations_done:2d}, "
                f"epsilon {self.eps:.3f}, "
                f"ESS (untruncated) {self.ess:.1f}, "
                f"elapsed mins {self.elapsed_time/60:.1f}"
            )
