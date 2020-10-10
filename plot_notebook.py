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

# %%
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

idx = 19

# Plot the RR
for tau in taus:

    rrs = [[values[tau][idx]['fold_res'][i]['boost_history'][j]['rr'] for i in range(5)] for j in range(31)]

    error = np.array([1.96 * np.std(rrs[i]) / (5**0.5) for i in range(31)])
    vals = np.array([np.mean(rrs[i]) for i in range(31)])

    plt.errorbar(range(31), vals, yerr=error, capsize=3, label=r'$\tau={}$'.format(tau))

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

idx = 19

# Plot the RR
for tau in taus:
    kls = [[values[tau][idx]['fold_res'][i]['boost_history'][j]['kl'] for i in range(5)] for j in range(31)]

    error = np.array([1.96 * np.std(kls[i]) / (5**0.5) for i in range(31)])
    vals = np.array([np.mean(kls[i]) for i in range(31)])

    plt.errorbar(range(31), vals, yerr=error, capsize=3, label=r'$\tau={}$'.format(tau))

ax.set_title('Simulated KL Divergence')
#ax.set_ylim((0.7, 1.0))
ax.set_xlim(0-0.2, 10+0.2)
ax.set_ylabel('KL Divergence')
ax.set_xlabel('Boosting Iterations')
#plt.legend()

plt.savefig('kl_divergence.eps', bbox_inches='tight', format='eps')

# %%
idx = 19
f_idx = 1
cur_tau = 0.7

q_init_pdf = values[cur_tau][idx]['fold_res'][f_idx]['boost_history'][0]['pdf']
q_final_pdf = values[cur_tau][idx]['fold_res'][f_idx]['boost_history'][-1]['pdf']

setting = values[cur_tau][idx]['setting']

mu_1 = setting['setting']['mu_1']
mu_2 = setting['setting']['mu_2']
std_1 = setting['setting']['std_1']
std_2 = setting['setting']['std_2']
skew = setting['setting']['skew']

d = SimulatedDistribution(mu_1, mu_2, std_1, std_2, skew)

xs = torch.linspace(-5, 5, 1001)

plt.plot(xs, torch.exp(d.log_prob(xs, torch.ones(xs.shape))) + torch.exp(d.log_prob(xs, torch.zeros(xs.shape))), c='b')
plt.plot(xs, torch.Tensor(q_init_pdf[0]) + torch.Tensor(q_init_pdf[1]), c='orange')
plt.plot(xs, torch.Tensor(q_final_pdf[0]) + torch.Tensor(q_final_pdf[1]), c='red')
plt.xlim(-2.5, 2.5)
#%%
