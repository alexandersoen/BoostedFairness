#%%
import dill
import numpy as np
from classifiers.utils import all_binaries

#%%
TRICK = False
MAX_I_VALS = 10 #49
SAVE = True

# %%

def top_feature(i, tau, sattr, fold, dataset):
    if TRICK:
        with open(f'dt_trick_{dataset}_{sattr}_{tau}/fold_{fold}/classifier_iter_{i}.p', 'rb') as f:
            tree = dill.load(f)
    else:
        with open(f'dt_{dataset}_{sattr}_{tau}/fold_{fold}/classifier_iter_{i}.p', 'rb') as f:
            tree = dill.load(f)


    if dataset == 'compas_small':
        DOMAIN = [1, 3, 3, 1, 1]
    else:
        DOMAIN = [1, 7, 9, 1]

    x_support = all_binaries(DOMAIN, threshold=[1,2])
    #%%
    split_range = list(reversed(range(sum(DOMAIN))))
    n = tree.root

    for i in split_range:
        vals_0 = [x for x in x_support if x[i] == 0]
        vals_1 = [x for x in x_support if x[i] == 1]

        if (sum(n.h(x) for x in vals_0) == len(vals_0) and sum(1 - n.h(x) for x in vals_1) == len(vals_1)) or (sum(n.h(x) for x in vals_1) == len(vals_1) or sum(1 - n.h(x) for x in vals_0) == len(vals_0)):
            split_idx = i
            break

    return split_idx

#%%
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
fig = plt.figure(figsize=(9, 6))
plt.subplots_adjust(hspace=0.7, top=0.9)
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

i = 1
for sattr in ['sex', 'race']:

    COL_NAMES = [
        'sex',
        'race',
        'age_cat in [0, 25)',
        'age_cat in [0, 45)',
        'age_cat=>45',
        'priors_count>=0',
        'priors_count in [0, 3]',
        'priors_count> 3',
        'c_charge_degree=F',
        'two_year_recid'
        ]

    COL_NAMES.remove(sattr)

    for tau in [0.7, 0.9]:
        ax = fig.add_subplot(2, 2, i)

        #fig = plt.figure(figsize=(10, 5))
        #ax = fig.add_subplot(1, 1, 1)

        counts = []

        i_vals = [i+1 for i in range(MAX_I_VALS)]

        for fold in range(5):
            #i_vals = [i+1 for i in range(49)]
            counts += [1+top_feature(i, tau, sattr, fold, 'compas_small') for i in i_vals]
            #plt.plot(i_vals, [top_feature(i, tau, sattr, fold) for i in i_vals], label=f'Fold {fold+1}')
        bar_vals = [counts.count(i) for i in i_vals]

        i_filter_vals = range(len([i for i in i_vals if counts.count(i) > 0]))
        bar_filter_vals = [counts.count(i) for i in i_vals if counts.count(i) > 0]
        col_filter_names = [COL_NAMES[i-1] for i in i_vals if counts.count(i) > 0]

        ax.bar(i_filter_vals, bar_filter_vals, facecolor=colours[i-1])

        ax.set_xticks(i_filter_vals)
        ax.set_xticklabels(col_filter_names, rotation=35, ha='right')

        plt.title(f'tau = {tau}; sattr = {sattr}')
        i += 1

if TRICK:
    plt.suptitle(r'COMPAS Split at Root Node w/ $\tilde{\theta_{t}}$', fontsize=14)
    if SAVE:
        plt.savefig('dt_trick_split_analysis_compas.eps', bbox_inches='tight', format='eps')
        plt.savefig('dt_trick_split_analysis_compas.png', bbox_inches='tight')
else:
    plt.suptitle('COMPAS Split at Root Node', fontsize=14)
    if SAVE:
        plt.savefig('dt_split_analysis_compas.eps', bbox_inches='tight', format='eps')
        plt.savefig('dt_split_analysis_compas.png', bbox_inches='tight')

#%%
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
fig = plt.figure(figsize=(9, 6))
plt.subplots_adjust(hspace=0.7, top=0.9)
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']


i = 1
for sattr in ['sex']:

    COL_NAMES = [
        'race',
        'sex',
        'Age <20',
        'Age <30',
        'Age <40',
        'Age <50',
        'Age <60',
        'Age <70',
        'Age >=70',
        'Education Years<6',
        'Education Years<7',
        'Education Years<8',
        'Education Years<9',
        'Education Years<10',
        'Education Years<11',
        'Education Years<12',
        'Education Years<=12',
        'Education Years>12',
        'Income Binary'
        ]

    COL_NAMES.remove(sattr)

    for tau in [0.7, 0.9]:
        ax = fig.add_subplot(2, 2, i)

        #fig = plt.figure(figsize=(10, 5))
        #ax = fig.add_subplot(1, 1, 1)

        counts = []

        i_vals = [i+1 for i in range(MAX_I_VALS)]

        for fold in range(5):
            #i_vals = [i+1 for i in range(49)]
            counts += [1+top_feature(i, tau, sattr, fold, 'adult') for i in i_vals]
            #plt.plot(i_vals, [top_feature(i, tau, sattr, fold) for i in i_vals], label=f'Fold {fold+1}')

        i_filter_vals = range(len([i for i in i_vals if counts.count(i) > 0]))
        bar_filter_vals = [counts.count(i) for i in i_vals if counts.count(i) > 0]
        col_filter_names = [COL_NAMES[i-1] for i in i_vals if counts.count(i) > 0]

        ax.bar(i_filter_vals, bar_filter_vals, facecolor=colours[i-1+4])

        ax.set_xticks(i_filter_vals)
        ax.set_xticklabels(col_filter_names, rotation=35, ha='right')

        plt.title(f'tau = {tau}; sattr = {sattr}')
        i += 1

if TRICK:
    plt.suptitle(r'Adult DT Split at Root Node w/ $\tilde{\theta_{t}}$', fontsize=14)
    if SAVE:
        plt.savefig('dt_trick_split_analysis_adult.eps', bbox_inches='tight', format='eps')
        plt.savefig('dt_trick_split_analysis_adult.png', bbox_inches='tight')
else:
    plt.suptitle('Adult DT Split at Root Node', fontsize=14)
    if SAVE:
        plt.savefig('dt_split_analysis_adult.eps', bbox_inches='tight', format='eps')
        plt.savefig('dt_split_analysis_adult.png', bbox_inches='tight')