#%%
import math
import torch

from scipy.integrate import nquad

class QDensity:

    def __init__(self, q0, support):
        self.q0 = q0
        self.support = support  # Tuple of lower and upper bound pairs
        self.models = []
        self.thetas = []
        self.logz = None
        self.normalised = True

    def __len__(self):
        return self.q0.event_shape[0]

    def isuntrained(self):
        return (len(self.models) == 0)

    def isnormalised(self):
        return self.normalised

    def normalise(self, abstol=1e-3):
        func = lambda *x: torch.exp(self.log_prob(torch.Tensor(x))).numpy()
        opts = {
            'epsabs': abstol,
        }

        with torch.no_grad():
            z, _ = nquad(func, self.support, opts=opts)
        self.logz = math.log(z)
        self.normalised = True

    def log_prob(self, x):
        dens = self.q0.log_prob(x)
        if self.isuntrained():
            return dens

        for (m, theta) in zip(self.models, self.thetas):
            dens += theta * m(x)[0]

        if not self.isnormalised():
            return dens

        return dens - self.logz

    def log_prob_grad(self, x):
        g = torch.Tensor(x)
        g.requires_grad = True

        f = self.q0.log_prob(g)
        for (m, theta) in zip(self.models, self.thetas):
            f += theta * m(g)[0]

        f.backward()
        return f, g.grad

    def append(self, m, theta):
        self.models.append(m)
        self.thetas.append(theta)
        self.normalised = False


# %%
q0 = torch.distributions.MultivariateNormal(torch.Tensor([0, 0]), torch.eye(2))
q = QDensity(q0, [(-20, 20), (-20, 20)])

c1 = torch.nn.Linear(2, 1)
q.append(c1, 0.7)

q.normalise()
# %%
