
#%%
# Imports
import json
import time
import math
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

from booster.booster import BoostDensity
from densities.fairdensitycts import FairDensityCts
from simulated.distribution import SimulatedDistribution

#%%
# Specify the dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

#%%
TAU = 0.9
TRAIN_SPLIT = 0.7

NUM_SAMPLES = 5_000

#%%
mu_1 = 0.6
mu_2 = -0.3
std_1 = 1
std_2 = 1.1
skew = 0.9

#%%
d = SimulatedDistribution(mu_1, mu_2, std_1, std_2, skew)

samples = d.sample([NUM_SAMPLES])

#%%

from sklearn.model_selection import KFold

results = {}

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
        nn.Hardtanh(min_val=-math.log10(2), max_val=math.log10(2))
    )

    #%%
    # Fair Density setup
    x_support = [(-20, 20)]
    a_domain = [[0], [1]]


    cond_0 = [a.item() for (a, b) in zip(*train_sample) if b.item() == 0]
    cond_1 = [a.item() for (a, b) in zip(*train_sample) if b.item() == 1]
    train_x_cond_a = [D.Normal(*norm.fit(cond_0)), D.Normal(*norm.fit(cond_1))]

    train_sample = list(zip(*train_sample))
    test_sample = list(zip(*test_sample))

    #%%
    boost = BoostDensity(TAU, train_sample, test_sample, train_x_cond_a, x_support, a_domain, model, true_dist=d)

    #%%
    boost.init_boost(optimiser_gen=optim.Adam, batch_size=128)

    #%%
    start = time.time()
    boost.boost()
    final = time.time()

    results[f_idx] = {
        'fold': f_idx,
        'train_history': boost.train_history,
        'boost_history': boost.boost_history,
        'runtime': final - start,
    }

#%%
#with open(NAME, 'w') as f:
#    json.dump(results, f)