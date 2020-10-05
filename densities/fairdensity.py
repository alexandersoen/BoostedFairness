#%%
import math
import torch
import itertools

from scipy.integrate import nquad
from torch.distributions.multinomial import Multinomial

class FairDensity:

    def __init__(self, q_init_xa_conds, x_support, a_domain):
        self.q_init_xa_conds = q_init_xa_conds
        self.x_support = x_support  # Tuple of lower and upper bound pairs
        self.a_domain = a_domain  # Enumeration of possible a's
        self.models = []
        self.thetas = []
        self.logzs = []

        self.a_pdf = Multinomial(probs=torch.ones(len(a_domain)))

    def normalise(self, abstol=1e-3):
        z = 0
        for a in self.a_domain:
            z_a = 0
            for x in self.x_support:
                z_a += math.exp(self._unnorm_log_prob(torch.Tensor(x), a))

            #func = lambda *x: math.exp(self._unnorm_log_prob(x, a))

            #opts = {
            #    'epsabs': abstol,
            #}
            #with torch.no_grad():
            #    z_a, _ = nquad(func, self.x_support, opts=opts)

            z += z_a

        return math.log(z)

    def log_prob(self, x, a):
        norm = 0
        if len(self.logzs) > 0:
            norm = self.logzs[-1]

        return self._unnorm_log_prob(x, a) - norm

    def _unnorm_log_prob(self, x, a):
        ai = self.a_domain.index(a)
        dens = self.q_init_xa_conds[ai].log_prob(x) - math.log(len(self.a_domain))

        for (m, theta) in zip(self.models, self.thetas):
            dens += theta * m(x)

        return dens

    def append(self, m, theta):
        self.models.append(m)
        self.thetas.append(theta)
        self.logzs.append(self.normalise())

    def gen_loss_function(self):
        def loss(classifier, samples):
            p_samples = [tuple([s[0], s[1]]) for s in samples if s[2] == 1]
            q_samples = [tuple([s[0], s[1]]) for s in samples if s[2] == 0]

            p_x_samples, _ = zip(*p_samples)
            q_x_samples, _ = zip(*q_samples)

            p_x_samples = torch.stack(p_x_samples)
            q_x_samples = torch.stack(q_x_samples)


            weights = 1
            for (m, theta, logz) in zip(self.models, self.thetas, self.logzs):
                weights *= torch.exp(theta * m(q_x_samples) - logz)

            p_expectation = torch.mean(torch.log(torch.sigmoid(classifier(p_x_samples))))
            q_expectation = torch.mean(torch.log(1 - torch.sigmoid(classifier(q_x_samples))) * weights)

            return -(p_expectation + q_expectation)

        return loss
    
    def representation_rate(self):
        rr_list = []

        a_probs = {}
        for a in self.a_domain:
            a_prob = 0
            for x in self.x_support:
                a_prob += math.exp(self.log_prob(torch.Tensor(x), a))
            a_probs[a] = a_prob

        for (ai, aj) in itertools.product(self.a_domain, repeat=2):
            rr_list.append(a_probs[ai] / a_probs[aj])

        return min(rr_list)


    def sample_q_init(self, n):

        a_samples = self.a_pdf.sample([int(n)])
        a_counts = torch.sum(a_samples, axis=0)

        sample_list = []
        for i in range(len(a_counts)):

            x_samples = self.q_init_xa_conds[i].sample([int(a_counts[i])])
            xa_samples = [(x, self.a_domain[i]) for x in x_samples]

            sample_list += xa_samples

        return sample_list