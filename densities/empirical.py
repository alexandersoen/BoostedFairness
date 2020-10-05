import math
import torch

from torch.distributions.distribution import Distribution
from torch.distributions.uniform import Uniform

class EmpiricalDistribution(Distribution):

    def __init__(self, raw_samples, domain=None, baseline=0):

        samples = list(tuple(int(x) for x in v) for v in raw_samples)

        self.domain = set(samples) if domain is None else domain
        self.probs = {k: 0 for k in self.domain}
        for s in samples:
            self.probs[s] += 1

        for s in self.domain:
            self.probs[s] = (self.probs[s] + baseline) / (len(samples) + len(self.domain) * baseline)

    def log_prob(self, value):
        t_value = tuple(int(v) for v in value)
        try:
            log_prob = math.log(self.probs[t_value])
        except (KeyError, ValueError):
            log_prob = math.log(1e-9)
        return log_prob

    def rsample(self, sample_shape=torch.Size([])):
        val_vec, pdf_vec = zip(*self.probs.items())

        cdf_vector = torch.cumsum(torch.Tensor(pdf_vec), 0)

        unif = Uniform(0, 1)
        u_samples = unif.rsample(sample_shape).reshape(-1)
        n_samples = len(u_samples)

        ids = torch.sum(cdf_vector.repeat(n_samples, 1) < u_samples.repeat(len(cdf_vector), 1).T, axis=1)

        return(torch.Tensor([val_vec[i] for i in ids]).reshape(*sample_shape, -1))