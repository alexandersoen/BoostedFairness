
#%%
# Imports
import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from booster.booster_dt import BoostDensityDT
from densities.empirical import EmpiricalDistribution
from classifiers.utils import all_binaries
from classifiers.embedding import EmbLayer

#%%
# Specify the dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets import CompasDataset

#%%
def interval(row):
    i_loc = np.where(row == 1)[0]
    if len(i_loc) > 0:
        for i in range(i_loc[0]):
            row[i] = 1
    return row

#%%
TAU = float(sys.argv[1])  # 0.9
SENSITIVE_ATTRIBUTE = str(sys.argv[2])  # 'sex'

#%%
#DOMAIN = [2, 3, 3, 2, 2]
DOMAIN = [1, 3, 3, 1, 1]
EMB_SIZE = 5

#NAME = 'dt_trick2_compas_small_{}_{}.json'.format(SENSITIVE_ATTRIBUTE, TAU)
NAME = 'dt_compas_small_{}_{}.json'.format(SENSITIVE_ATTRIBUTE, TAU)

dataset = load_preproc_data_compas(['sex'])
dataset_df = dataset.convert_to_dataframe()[0]
# 
dataset_df = dataset_df.reindex(columns=['sex', 'race', 'age_cat=Less than 25', 'age_cat=25 to 45', 'age_cat=Greater than 45', 'priors_count=0', 'priors_count=1 to 3', 'priors_count=More than 3', 'c_charge_degree=F', 'c_charge_degree=M','two_year_recid'])

# Change to interval threholds
dataset_df.iloc[:, 2:5] = np.apply_along_axis(interval, 1, dataset_df.iloc[:, 2:5].values)
dataset_df.iloc[:, 5:8] = np.apply_along_axis(interval, 1, dataset_df.iloc[:, 5:8].values)

del dataset_df['c_charge_degree=M']

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
    # Fair Density setup
    x_support = all_binaries(DOMAIN, threshold=[1,2])
    a_domain = [tuple(int(v) for v in t) for t in torch.eye(train_a.shape[1])]

    train_x_cond_a = []
    for i in range(train_a.shape[1]):
        a_mask = train_a[:, i] > 0

        train_x_cond_a.append(EmpiricalDistribution(train_x[a_mask], domain=x_support, baseline=0))
        #train_x_cond_a.append(EmpiricalDistribution([], domain=x_support, baseline=1))

    save_dir = 'dt_compas_small_{}_{}/fold_{}'.format(SENSITIVE_ATTRIBUTE, TAU, f_idx)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #%%
    boost = BoostDensityDT(TAU, train_sample, test_sample, train_x_cond_a, x_support, a_domain)

    #%%
    boost.init_boost(num_iter=50, max_depth=8)

    #%%
    start = time.time()
    boost.boost(save_classifiers=save_dir)
    final = time.time()

    results[f_idx] = {
        'fold': f_idx,
        'train_history': boost.train_history,
        'boost_history': boost.boost_history,
        'runtime': final - start,
    }

#%%
results[0]['boost_history']

#%%
with open(NAME, 'w') as f:
    json.dump(results, f)