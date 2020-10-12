#%%
import json
import torch
import numpy as np

import matplotlib.pyplot as plt

from simulated.distribution import SimulatedDistribution

#%%
taus = [0.7, 0.9]

values = {}
for tau in taus:

    with open('simulated_res_{}.json'.format(tau), 'r') as f:
        res = json.load(f)

    values[tau] = res

#%%
best_indices = {}

# Get largest RR reduction
for tau in taus:
    rr_values = [min(values[tau][j]['fold_res'][i]['boost_history'][-1]['rr'] for i in range(5)) for j in range(len(values[tau]))]
    cur_idx = sorted(enumerate(rr_values), key=lambda x: x[1])[0][0]

    best_indices[tau] = cur_idx

idx = 35

setting = values[taus[0]][idx]['setting']
print(setting)

# %%
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

# Plot the RR
for tau in taus:

    rrs = [[values[tau][idx]['fold_res'][i]['boost_history'][j]['rr'] for i in range(5)] for j in range(11)]

    error = np.array([1.96 * np.std(rrs[i]) / (5**0.5) for i in range(11)])
    vals = np.array([np.mean(rrs[i]) for i in range(11)])

    plt.errorbar(range(11), vals, yerr=error, capsize=3, label=r'$\tau={}$'.format(tau))

ax.set_title('Simulated Representation Rate')
ax.set_ylim((0.7, 1.0+0.01))
ax.set_xlim(0-0.2, 10+0.2)
ax.set_ylabel('Representation Rate')
ax.set_xlabel('Boosting Iterations')
plt.legend()

plt.savefig('representation_rate.eps', bbox_inches='tight', format='eps')

# %%
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

# Plot the RR
for tau in taus:
    kls = [[values[tau][idx]['fold_res'][i]['boost_history'][j]['kl'] for i in range(5)] for j in range(11)]

    error = np.array([1.96 * np.std(kls[i]) / (5**0.5) for i in range(11)])
    vals = np.array([np.mean(kls[i]) for i in range(11)])

    plt.errorbar(range(11), vals, yerr=error, capsize=3, label=r'$\tau={}$'.format(tau))

ax.set_title('Simulated KL Divergence')
#ax.set_ylim((0.7, 1.0))
ax.set_xlim(0-0.2, 10+0.2)
ax.set_ylabel('KL Divergence')
ax.set_xlabel('Boosting Iterations')
#plt.legend()

plt.savefig('kl_divergence.eps', bbox_inches='tight', format='eps')

# %%
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

f_idx = 1
cur_tau = 0.7

cur_boost_history = values[cur_tau][idx]['fold_res'][f_idx]['boost_history']
#cur_boost_history = results[0]['fold_res'][0]['boost_history']

q_init_pdf = cur_boost_history[0]['pdf']
q_final_pdf = cur_boost_history[-1]['pdf']

setting = values[cur_tau][idx]['setting']

mu_1 = setting['setting']['mu_1']
mu_2 = setting['setting']['mu_2']
std_1 = setting['setting']['std_1']
std_2 = setting['setting']['std_2']
skew = setting['setting']['skew']

d = SimulatedDistribution(mu_1, mu_2, std_1, std_2, skew)

xs = torch.linspace(-5, 5, 1001)

plt.plot(xs, torch.exp(d.log_prob(xs, torch.ones(xs.shape))) + torch.exp(d.log_prob(xs, torch.zeros(xs.shape))), c='blue', label=r'P')
plt.plot(xs, torch.Tensor(q_init_pdf[0]) + torch.Tensor(q_init_pdf[1]), c='orange', label=r'$Q_{0}$')
plt.plot(xs, torch.Tensor(q_final_pdf[0]) + torch.Tensor(q_final_pdf[1]), c='red', label=r'$Q_{T}$')

ax.set_xlim(-2.5, 2.5)
ax.set_ylabel(r'Marginal Probability of $x$')
ax.set_xlabel(r'$x$')
ax.set_title('Marginal Distribution')
plt.legend()

plt.savefig('distribution_compare.eps', bbox_inches='tight', format='eps')

#%%
stat_res = {t: {} for t in taus}
for cur_tau in taus:
    print('\ntau = {}'.format(cur_tau))
    cur_vals = values[cur_tau][idx]['fold_res']

    cur_init_rrs = [cur_vals[j]['boost_history'][0]['rr'] for j in range(5)]
    cur_init_kls = [cur_vals[j]['boost_history'][0]['kl'] for j in range(5)]
    cur_final_rrs = [cur_vals[j]['boost_history'][-1]['rr'] for j in range(5)]
    cur_final_kls = [cur_vals[j]['boost_history'][-1]['kl'] for j in range(5)]
    cur_runtime = [cur_vals[j]['runtime'] / 60 for j in range(5)]

    stat_res[cur_tau]['init_rr'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_init_rrs), np.std(cur_init_rrs)))
    stat_res[cur_tau]['init_kl'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_init_kls), np.std(cur_init_kls)))
    stat_res[cur_tau]['final_rr'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_final_rrs), np.std(cur_final_rrs)))
    stat_res[cur_tau]['final_kl'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_final_kls), np.std(cur_final_kls)))
    stat_res[cur_tau]['runtime'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_runtime), np.std(cur_runtime)))

    print(stat_res[cur_tau]['runtime'])

print('{} & {} & {}'.format(stat_res[taus[0]]['init_rr'], stat_res[taus[0]]['final_rr'], stat_res[taus[1]]['final_rr']))
print('{} & {} & {}'.format(stat_res[taus[0]]['init_kl'], stat_res[taus[0]]['final_kl'], stat_res[taus[1]]['final_kl']))


# %%
dataset_str = '{}_{}_{}.json'

dataset_names = ['compas_small', 'adult']
sensitive_names = ['sex', 'race']

for cur_dname in dataset_names:
    for cur_sname in sensitive_names:
        stat_res = {t: {} for t in taus}
        for cur_tau in taus:

            cur_dataset = dataset_str.format(cur_dname, cur_sname, cur_tau)
            print('\n{}'.format(cur_dataset))

            with open(cur_dataset, 'r') as f:
                cur_vals = json.load(f)

            cur_init_rrs = [cur_vals[str(j)]['boost_history'][0]['rr'] for j in range(5)]
            cur_init_kls_train = [cur_vals[str(j)]['boost_history'][0]['kl'][0] for j in range(5)]
            cur_init_kls = [cur_vals[str(j)]['boost_history'][0]['kl'][1] for j in range(5)]
            cur_final_rrs = [cur_vals[str(j)]['boost_history'][-1]['rr'] for j in range(5)]
            cur_final_kls_train = [cur_vals[str(j)]['boost_history'][-1]['kl'][0] for j in range(5)]
            cur_final_kls = [cur_vals[str(j)]['boost_history'][-1]['kl'][1] for j in range(5)]
            cur_runtime = [cur_vals[str(j)]['runtime'] / 60 for j in range(5)]

            stat_res[cur_tau]['init_rr'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_init_rrs), np.std(cur_init_rrs)))
            stat_res[cur_tau]['init_kl'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_init_kls), np.std(cur_init_kls)))
            stat_res[cur_tau]['init_kl_train'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_init_kls_train), np.std(cur_init_kls_train)))
            stat_res[cur_tau]['final_rr'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_final_rrs), np.std(cur_final_rrs)))
            stat_res[cur_tau]['final_kl'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_final_kls), np.std(cur_final_kls)))
            stat_res[cur_tau]['final_kl_train'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_final_kls_train), np.std(cur_final_kls_train)))
            stat_res[cur_tau]['runtime'] = ('{:.3f} ({:.3f})'.format(np.mean(cur_runtime), np.std(cur_runtime)))
            print(stat_res[cur_tau]['runtime'])

        print('{} & {} & {}'.format(stat_res[taus[0]]['init_rr'], stat_res[taus[0]]['final_rr'], stat_res[taus[1]]['final_rr']))
        print('{} & {} & {}'.format(stat_res[taus[0]]['init_kl'], stat_res[taus[0]]['final_kl'], stat_res[taus[1]]['final_kl']))
        print('{} & {} & {}'.format(stat_res[taus[0]]['init_kl_train'], stat_res[taus[0]]['final_kl_train'], stat_res[taus[1]]['final_kl_train']))
# %%
dataset_str = '{}_{}_{}.json'

dataset_names = ['compas_small', 'adult']
sensitive_names = ['sex', 'race']

for cur_dname in dataset_names:
    for cur_sname in sensitive_names:
        for cur_tau in taus:
            cur_dataset = dataset_str.format(cur_dname, cur_sname, cur_tau)
            print('\n{}'.format(cur_dataset))

            with open(cur_dataset, 'r') as f:
                cur_vals = json.load(f)

            accuracy = np.mean([cur_vals[str(i)]['train_history'][199]['train_acc'] for i in range(5)])

            print(accuracy)
# %%
