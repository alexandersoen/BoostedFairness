#%%
# Imports
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
TRAIN_SPLIT = 0.7
SENSITIVE_ATTRIBUTE = 'sex'
DOMAIN = [1, 3, 3, 2, 1]

dataset = load_preproc_data_compas(['sex'])
dataset_df = dataset.convert_to_dataframe()[0]
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

train_len = int(len(dataset_df) * TRAIN_SPLIT)

train_df = dataset_df[:train_len]
test_df = dataset_df[train_len:]

train_x = torch.Tensor(train_df.loc[:, 'race':].values)
train_a = torch.Tensor(train_df.loc[:, 'sex':'sex'].values)
train_a = torch.cat([train_a, 1 - train_a], axis=1)
train_sample = list(zip(train_x, train_a))

test_x = torch.Tensor(test_df.loc[:, 'race':].values)
test_a = torch.Tensor(test_df.loc[:, 'sex':'sex'].values)
test_a = torch.cat([test_a, 1 - test_a], axis=1)
test_sample = list(zip(test_x, test_a))


#%%
# Models
model = nn.Sequential(
    nn.Linear(train_x.shape[1], 20),
    nn.ReLU(),
    nn.Linear(20, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.Hardtanh()
)

#%%
# Fair Density setup
x_support = all_binaries(DOMAIN)
a_domain = [tuple(int(v) for v in t) for t in torch.eye(train_a.shape[1])]

train_x_cond_a = []
for i in range(train_a.shape[1]):
    a_mask = train_a[:, i] > 0

    #train_x_cond_a.append(EmpiricalDistribution(train_x[a_mask], domain=x_support, baseline=1))
    train_x_cond_a.append(EmpiricalDistribution([], domain=x_support, baseline=1))

#%%
boost = BoostDensity(0.5, train_sample, test_sample, train_x_cond_a, x_support, a_domain, model)

#%%
boost.init_boost(optimiser_gen=optim.Adam, batch_size=128)

#%%
boost.boost()
# %%
