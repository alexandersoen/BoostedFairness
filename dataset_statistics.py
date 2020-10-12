#%%
# Specify the dataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas, load_preproc_data_adult

#%%
attributes = ['sex', 'race']

dataset = load_preproc_data_compas(['sex'])
dataset_df = dataset.convert_to_dataframe()[0]

#%%
for a in attributes:
    positive = sum(dataset_df[a])
    negative = len(dataset_df) - sum(dataset_df[a])

    print(a, min(positive/negative, negative/positive))

#%%
attributes = ['sex', 'race']

dataset = load_preproc_data_adult(['sex'])
dataset_df = dataset.convert_to_dataframe()[0]

#%%
for a in attributes:
    positive = sum(dataset_df[a])
    negative = len(dataset_df) - sum(dataset_df[a])

    print(a, min(positive/negative, negative/positive))

#%%
attributes = ['sex', 'race']

dataset = load_preproc_data_compas(['sex'])
dataset_df = dataset.convert_to_dataframe()[0]

#%%
def str_add(x, y):
    return str(x) + str(y)

for a in attributes:
    atts = ['a']
    positive = sum(dataset_df[a])
    negative = len(dataset_df) - sum(dataset_df[a])

    print(a, min(positive/negative, negative/positive))