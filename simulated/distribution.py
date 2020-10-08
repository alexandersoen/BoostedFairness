#%%
import torch
import numpy as np

import torch.distributions as D
from scipy.stats import skewnorm, bernoulli, norm

class SimulatedDistribution(D.Distribution):
    def __init__(self, mu_1, mu_2, std_1, std_2, skew_weight):
        self.norms = D.Normal(loc=torch.Tensor([mu_1, mu_2]), scale=torch.Tensor([std_1, std_2]))
        self.mix = D.Categorical(torch.Tensor([skew_weight, 1-skew_weight]))

    def rsample(self, sample_shape=torch.Size([])):
        a_samples = self.mix.sample(sample_shape)

        pre_x_samples = self.norms.sample(sample_shape)
        x_samples = torch.sum(pre_x_samples * torch.stack([a_samples, 1 - a_samples]).T, axis=1)

        return x_samples, a_samples

    def log_prob(self, x, a):
        a = torch.Tensor(a)
        raw_x_log_probs = self.norms.log_prob(torch.stack([x, x]).T)
        x_log_probs = torch.sum(raw_x_log_probs * torch.stack([a, 1 - a]).T, axis=1)
        a_log_probs = self.mix.log_prob(a)

        return x_log_probs + a_log_probs