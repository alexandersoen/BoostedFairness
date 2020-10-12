
#%%
import sys
import time
import math
import json
import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import concurrent.futures as futures

from tqdm import tqdm
from scipy.stats import norm
from classifiers.utils import KL
from booster.booster import BoostDensity
from simulated.distribution import SimulatedDistribution
from densities.fairdensitycts import FairDensityCts

from sklearn.model_selection import KFold

#%%

TAU = float(sys.argv[1])
SINGLE_SIM = None

#%%

NUM_SAMPLES = 5_000
NUM_SIMS = 48
NUM_ITERS = 10
MAX_CPUS_PER_WORKER = 1

with open('kl_settings.json', 'r') as f:
    simulated_settings = json.load(f)

if SINGLE_SIM:
    payload = [sorted(simulated_settings, key=lambda x: -float(x['kl']))[19]]
else:
    payload = sorted(simulated_settings, key=lambda x: -float(x['kl']))[:NUM_SIMS]

total_settings = len(payload)

#%%
def worker(setting):

    torch.set_num_threads(MAX_CPUS_PER_WORKER)

    mu_1 = setting['setting']['mu_1']
    mu_2 = setting['setting']['mu_2']
    std_1 = setting['setting']['std_1']
    std_2 = setting['setting']['std_2']
    skew = setting['setting']['skew']

    d = SimulatedDistribution(mu_1, mu_2, std_1, std_2, skew)

    samples = d.sample([NUM_SAMPLES])

    cur_fold_res = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    for f_idx, (train_index, test_index) in enumerate(cv.split(samples[0])):

        #train_len = int(len(dataset_df) * TRAIN_SPLIT)
        train_x = samples[0][train_index].reshape(-1, 1)
        train_a = samples[1][train_index].reshape(-1, 1)
        train_sample = (train_x, train_a)

        test_x = samples[0][test_index].reshape(-1, 1)
        test_a = samples[1][test_index].reshape(-1, 1)
        test_sample = (test_x, test_a)


        #%%
        # Models
        model = nn.Sequential(
            nn.Linear(train_x.shape[1], 20),
            nn.ReLU(),
            #nn.Linear(20, 200),
            #nn.ReLU(),
            #nn.Linear(200, 200),
            #nn.ReLU(),
            #nn.Linear(200, 20),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Hardtanh(min_val=-math.log(2), max_val=math.log(2))
        )

        #%%
        # Fair Density setup
        x_support = [(-20, 20)]
        a_domain = [[0], [1]]


        cond_0 = [a.item() for (a, b) in zip(*train_sample) if b.item() == 0]
        cond_1 = [a.item() for (a, b) in zip(*train_sample) if b.item() == 1]
        train_x_cond_a = [D.Normal(*norm.fit(cond_0)), D.Normal(*norm.fit(cond_1))]
        #train_x_cond_a = [D.Normal(0, 1), D.Normal(0, 1)]

        train_sample = list(zip(*train_sample))
        test_sample = list(zip(*test_sample))

        #%%
        boost = BoostDensity(TAU, train_sample, test_sample, train_x_cond_a, x_support, a_domain, model, true_dist=d)

        #%%
        boost.init_boost(optimiser_gen=optim.Adam, batch_size=128, num_iter=NUM_ITERS, num_epochs=200, calc_pdf=True)

        #%%
        start = time.time()
        #boost.boost(verbose=False)
        boost.boost(verbose=True)
        final = time.time()

        cur_fold_res.append({
            'fold_idx': f_idx,
            'train_history': boost.train_history,
            'boost_history': boost.boost_history,
            'runtime': final - start,
        })

    return {
        'setting': setting,
        'fold_res': cur_fold_res,
    }

results = []

#%%
with futures.ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(worker, payload), total=total_settings))

#%%
if SINGLE_SIM:
    with open('simulated_res_{}_single_{}.json'.format(TAU, SINGLE_SIM), 'w') as f:
        json.dump(results, f)
else:
    with open('simulated_res_{}.json'.format(TAU), 'w') as f:
        json.dump(results, f)

# %%

