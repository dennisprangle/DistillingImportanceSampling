import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfe = tf.contrib.eager
import numpy as np
import matplotlib.pyplot as plt
from time import time

class DIS:
    def __init__(self, model, q, optimiser, importance_size,
                 ess_target, max_bisection_its=50, batch_size=100,
                 max_weight=1., nbatches=None, log_dir=None):
       """Class to perform a distilled importance sampling analysis

       `model` encapsulates model and prior
       `q` is approximate posterior distribution
       `optimiser` is a tensorflow optimiser object
       `importance_size` is number of importance samples (N in paper)
       `ess_target` is target effective sample size (M in paper)
       `max_bisection_its` controls the ESS selection algorithm
       `batch_size` is training batch size (n in paper)
       `max_weight` is the maximum normalised weight allowed after clipping (omega in paper)
       `nbatches` is how many training batches to create (B in paper)
       `log_dir` tensorboard log directory (None for no logging)
       """
       self.start_time = time()
       self.elapsed = 0.
       self.model = model
       self.q = q
       self.optimiser = optimiser
       self.importance_size = importance_size
       self.ess_target = ess_target
       self.max_bisection_its = max_bisection_its
       self.batch_size = batch_size
       self.max_weight = max_weight
       if nbatches is None:
           self.nbatches = np.ceil(ess_target / batch_size).astype('int')
       else:
           self.nbatches = nbatches
       self.t = 0 # Number of iterations performed so far
       self.eps = model.max_eps # Current eps value
       self.time_list = [0.] # Elapsed time
       self.eps_list = [self.eps]
       self.it_list = [0] # Number of iterations done
       self.to_train = tf.trainable_variables()
       if log_dir is None:
           self.tensorboard = False
       else:
           self.tensorboard = True
           self.writer = tf.contrib.summary.create_file_writer(log_dir)
           self.writer.set_as_default()
           self.steps = 0


    def get_weights(self, eps):
       """Calculates normalised importance sampling weights"""
       logw = self.model.log_tempered_target(eps) - self.log_q
       max_logw = np.max(logw)
       if max_logw == -np.inf:
           raise ValueError('All weights zero! ' \
              + 'Suggests overflow in importance density.')

       w = np.exp(logw - max_logw)
       return w


    def clip_weights(self, w):
        """Clip weights to `self.max_weight`
        Other weights are scaled up proportionately to keep sum equal to 1"""
        S = sum(w)
        to_clip = (w > S*self.max_weight)
        n_to_clip = sum(to_clip)
        if n_to_clip == 0:
            return w

        print("Truncating {:d} weights".format(n_to_clip))
        to_not_clip = np.logical_not(to_clip)
        sum_unclipped = sum(w[to_not_clip])
        if sum_unclipped == 0:
            # Impossible to clip further!
            return w
        clip_to = self.max_weight * sum_unclipped \
                  / (1. - self.max_weight * n_to_clip)
        max_unclipped = np.max(w[to_not_clip])
        ## clip_to calculation is done so that
        ## after w[to_clip]=clip_to
        ## w[to_clip] / sum(w) all equal max_weight
        ## **But** we don't want to clip below next smallest weight
        if clip_to >= max_unclipped:
            w[to_clip] = clip_to
            return w
        else:
            w[to_clip] = max_unclipped
            return clip_weights(w)

    def get_ess(self, w):
        """Calculates effective sample size of normalised importance sampling weights"""
        ess = (np.sum(w) ** 2.0) / np.sum(w ** 2.0)
        return ess


    def get_eps_and_weights(self, eps_guess):
        """Find new epsilon value

        Uses bisection to find epsilon < eps_guess giving required ESS. If none exists, returns eps_guess.

        Returns new epsilon value and corresponding ESS and normalised importance sampling weights.
        """
        
        lower = 0.
        upper = eps_guess
        eps_guess = (lower + upper) / 2.
        for i in range(self.max_bisection_its):
            w = self.get_weights(eps_guess)
            ess = self.get_ess(w)
            if ess > self.ess_target:
                upper = eps_guess
            else:
                lower = eps_guess
            eps_guess = (lower + upper) / 2.

        # Consider returning eps=0 if it's still an endpoint
        if lower == 0.:
            w = self.get_weights(0.)
            ess = self.get_ess(w)
            if ess > self.ess_target:
                return 0., ess, w

        # Be conservative by returning upper end of remaining range
        w = self.get_weights(upper)
        ess = self.get_ess(w)
        return upper, ess, w


    def loss(self, theta):
        return -tf.reduce_sum(self.q.log_prob(theta))


    def pretrain(self, goal=0.9, report_every=100):
        """Train approximation to match initial target"""
        ess = 0.
        its_done = 0
        while ess < self.batch_size * goal:
            for i in range(report_every):
                theta = self.model.initial_target.sample(self.batch_size)
                with tf.GradientTape() as tape:
                    l = self.loss(theta)
                g = tape.gradient(l, self.to_train)
                self.optimiser.apply_gradients(zip(g, self.to_train))
                if self.tensorboard:
                    with self.writer.as_default(), \
                         tf.contrib.summary.always_record_summaries():
                        grad_norm = tf.global_norm(g)
                        tf.contrib.summary.scalar('loss', l, step=self.steps)
                        tf.contrib.summary.scalar('grad_norm', grad_norm,
                                                  step=self.steps)
                        self.steps += 1

            its_done += report_every
            theta = self.q.sample(self.batch_size)
            logw = self.model.initial_target.log_prob(theta)
            logw -= self.q.log_prob(theta)
            max_logw = np.max(logw)
            if max_logw > -np.inf:
                w = np.exp(logw - np.max(logw))
                w /= sum(w)
                ess = self.get_ess(w)
            else:
                ess = 0.
            print(("Pretraining iterations {:2d}, loss {:.1f}, " +
                "ESS (from 100 samples) {:.1f}").format(its_done, l.numpy(), ess))


    def train(self, iterations=10):
        for i in range(iterations):
            theta = self.q.sample(self.importance_size)
            nsampled = theta.shape[0] # May be less than importance size if sample allows
                                      # instant rejection
            self.log_q = self.q.log_prob(theta).numpy() ## Needed for importance
                                                        ## sampling weights

            ## Fix nans/infs - can occur due to numeric overflow
            ## TO DO - ideally alter code (SDE? real NVP?) to prevent this from occuring
            log_q_is_nan = np.isnan(self.log_q)
            theta_is_nan = (tf.reduce_any(tf.is_nan(theta), 1)).numpy()
            theta_is_inf = (tf.reduce_any(tf.is_inf(theta), 1)).numpy()
            bad_rows = np.any(
                np.vstack((log_q_is_nan, theta_is_nan, theta_is_inf)),
                axis=0)
            if sum(bad_rows) > 0:
                print("nans or infs in sample or density!")
                ## Next line effectively sets weight to zero
                self.log_q[bad_rows] = np.inf
                ## Set nan/infinite thetas to zero (an arbitrary safe value)
                theta = tf.where(bad_rows, tf.zeros_like(theta), theta)
                theta = tf.where(bad_rows, tf.zeros_like(theta), theta)
                                                        
            self.model.likelihood_prelims(theta) # Store summaries weights
            self.eps, ess, w = self.get_eps_and_weights(self.eps)
            w = self.clip_weights(w)
            S = sum(w)
            w /= S

            # Create training batches
            indices = [np.random.choice(nsampled, size=self.batch_size, p=w)
                       for _ in range(self.nbatches)]
            self.batches = [tf.gather(theta, i) for i in indices]

            # Train on batches
            total_loss = 0.
            for theta_batch in self.batches:
                with tf.GradientTape() as tape:
                    l = S * self.loss(theta_batch)

                total_loss += l
                g = tape.gradient(l, self.to_train)
                self.optimiser.apply_gradients(zip(g, self.to_train))

                if self.tensorboard:
                    with self.writer.as_default(), \
                         tf.contrib.summary.always_record_summaries():
                        grad_norm = tf.global_norm(g)
                        tf.contrib.summary.scalar('loss', l, step=self.steps)
                        tf.contrib.summary.scalar('grad_norm', grad_norm,
                                                  step=self.steps)
                        self.steps += 1

            self.t += 1
            self.elapsed = time() - self.start_time
            self.time_list += [self.elapsed]
            self.eps_list += [self.eps]
            self.it_list += [self.t]

            # Report status
            message = "Iteration {:2d}, epsilon {:.3f}, ESS {:.1f} " + \
                      "elapsed mins {:.1f}"
            print(message.format(self.t, self.eps, ess, self.elapsed/60.))
