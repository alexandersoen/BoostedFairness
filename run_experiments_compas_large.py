#%%
# Imports
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from booster.booster import BoostDensity
from densities.empirical import EmpiricalDistribution
from classifiers.utils import all_binaries

#%%
# Specify the dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets import CompasDataset

#%%
TAU = 0.9
TRAIN_SPLIT = 0.7
SENSITIVE_ATTRIBUTE = 'sex'
DOMAIN = [1, 3, 3, 2, 1]

NAME = 'compas_small_{}'.format(SENSITIVE_ATTRIBUTE)

dataset = CompasDataset()
raw_dataset_df = dataset.convert_to_dataframe()[0]
raw_dataset_df['age1'] = (raw_dataset_df['age'] <= 25).astype(int)
raw_dataset_df['age2'] = ((raw_dataset_df['age'] > 25) & (raw_dataset_df['age'] <= 65)).astype(int)
raw_dataset_df['age3'] = (raw_dataset_df['age'] > 65).astype(int)
raw_dataset_df['pior1'] = (raw_dataset_df['prior_count'] <= 0).astype(int)
raw_dataset_df['pior2'] = ((raw_dataset_df['prior_count'] > 0) & (raw_dataset_df['prior_count'] <= 10)).astype(int)
raw_dataset_df['pior3'] = ((raw_dataset_df['prior_count'] > 10) & (raw_dataset_df['prior_count'] <= 20)).astype(int)
raw_dataset_df['pior4'] = ((raw_dataset_df['prior_count'] > 20) & (raw_dataset_df['prior_count'] <= 30)).astype(int)
raw_dataset_df['pior5'] = ((raw_dataset_df['prior_count'] > 30) & (raw_dataset_df['prior_count'] <= 40)).astype(int)
raw_dataset_df['pior6'] = (raw_dataset_df['prior_count'] > 40).astype(int)
raw_dataset_df['jail1'] = (raw_dataset_df['days_in_jail']/12 <= 0).astype(int)
raw_dataset_df['jail2'] = ((raw_dataset_df['days_in_jail']/12 > 0) & (raw_dataset_df['days_in_jail']/12 <= 3)).astype(int)
raw_dataset_df['jail3'] = ((raw_dataset_df['days_in_jail']/12 > 3) & (raw_dataset_df['days_in_jail']/12 <= 6)).astype(int)
raw_dataset_df['jail4'] = ((raw_dataset_df['days_in_jail']/12 > 6) & (raw_dataset_df['days_in_jail']/12 <= 12)).astype(int)
raw_dataset_df['jail5'] = ((raw_dataset_df['days_in_jail']/12 > 12) & (raw_dataset_df['days_in_jail']/12 <= 24)).astype(int)
raw_dataset_df['jail6'] = ((raw_dataset_df['days_in_jail']/12 > 24) & (raw_dataset_df['days_in_jail']/12 <= 48)).astype(int)
raw_dataset_df['jail7'] = ((raw_dataset_df['days_in_jail']/12 > 48) & (raw_dataset_df['days_in_jail']/12 <= 60)).astype(int)
raw_dataset_df['jail8'] = (raw_dataset_df['days_in_jail']/12 > 60).astype(int)

dataset_df 
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

from sklearn.model_selection import KFold

results = {}

cv = KFold(n_splits=5, random_state=42, shuffle=True)
for f_idx, (train_index, test_index) in enumerate(cv.split(dataset_df)):

    #train_len = int(len(dataset_df) * TRAIN_SPLIT)

    train_df = dataset_df.iloc[train_index]
    test_df = dataset_df.iloc[test_index]

    sensitive = [c for c in train_df.columns if c == SENSITIVE_ATTRIBUTE]
    nonsensitive = [c for c in train_df.columns if c != SENSITIVE_ATTRIBUTE]

    train_x = torch.Tensor(train_df.loc[:, nonsensitive].values)
    train_a = torch.Tensor(train_df.loc[:, sensitive].values)
    train_a = torch.cat([train_a, 1 - train_a], axis=1)
    train_sample = list(zip(train_x, train_a))

    test_x = torch.Tensor(test_df.loc[:, nonsensitive].values)
    test_a = torch.Tensor(test_df.loc[:, sensitive].values)
    test_a = torch.cat([test_a, 1 - test_a], axis=1)
    test_sample = list(zip(test_x, test_a))


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
    x_support = all_binaries(DOMAIN)
    a_domain = [tuple(int(v) for v in t) for t in torch.eye(train_a.shape[1])]

    train_x_cond_a = []
    for i in range(train_a.shape[1]):
        a_mask = train_a[:, i] > 0

        train_x_cond_a.append(EmpiricalDistribution(train_x[a_mask], domain=x_support, baseline=1))
        #train_x_cond_a.append(EmpiricalDistribution([], domain=x_support, baseline=1))

    #%%
    boost = BoostDensity(TAU, train_sample, test_sample, train_x_cond_a, x_support, a_domain, model)

    #%%
    boost.init_boost(optimiser_gen=optim.Adam, batch_size=128)

    #%%
    start = time.time()
    boost.boost()
    final = time.time()

    results[i] = {
        'fold': f_idx,
        'train_history': boost.train_history,
        'boost_history': boost.boost_history,
        'runtime': final - start,
    }

#%%
with open(NAME, 'w') as f:
    json.dump(results, f)