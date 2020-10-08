#%%
import json
import itertools

import numpy as np
import torch.distributions as D
import concurrent.futures as futures

from tqdm import tqdm
from scipy.stats import norm
from classifiers.utils import KL
from simulated.distribution import SimulatedDistribution
from densities.fairdensitycts import FairDensityCts

NUM_SAMPLES = 5_000

mu_1_vals = np.linspace(-1, 0, 11)
mu_2_vals = np.linspace(0, 1, 11)
std_vals = np.linspace(0, 2, 11)[1:]
skew_vals = np.linspace(0.7, 0.9, 6)

total_settings = len(mu_1_vals) * len(mu_1_vals) * len(std_vals) * len(std_vals) * len(skew_vals)

payload = itertools.product(mu_1_vals, mu_2_vals, std_vals, std_vals, skew_vals)

#%%
def worker(setting):

    mu_1, mu_2, std_1, std_2, skew = setting

    d = SimulatedDistribution(mu_1, mu_2, std_1, std_2, skew)

    samples_p = d.sample([NUM_SAMPLES])
    cond_0 = [a.item() for (a, b) in zip(*samples_p) if b.item() == 0]
    cond_1 = [a.item() for (a, b) in zip(*samples_p) if b.item() == 1]

    cond_dists = [D.Normal(*norm.fit(cond_0)), D.Normal(*norm.fit(cond_1))]

    q = FairDensityCts(cond_dists, [(-20, 20)], [[0], [1]])

    cur_kl = KL(d, q, q.x_support, q.a_domain)

    settings = {
        'mu_1': mu_1,
        'mu_2': mu_2,
        'std_1': std_1,
        'std_2': std_2,
        'skew': skew,
    }

    return {
        'setting': settings,
        'kl': cur_kl,
    }

results = []

#%%

with futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(worker, payload), total=total_settings))

with open('kl_settings.json', 'w') as f:
    json.dump(results, f)

# %%
