import os
import math
import copy
import torch
import dill as pickle
import itertools


from tqdm import tqdm
from scipy.stats import entropy
from torch.utils.data import DataLoader

# Imported stuff
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# ...
from algorithm.topdown import TopDown, score
from algorithm.criterion import gini
from learner.projection import Projection

from densities.fairdensity import FairDensity
from densities.fairdensitycts import FairDensityCts
from classifiers.utils import (accuracy, batch_weights, weighted_mean, KL)


def to_boost_clf(clf, x_support):
    vals = []
    for x in x_support:
        vals.append(abs(clf.predict(torch.Tensor(x).view(1, -1))))

    m_val = max(vals)
    return lambda x: (clf.predict(torch.Tensor(x).view(1, -1))) * math.log(2) / m_val

class BoostDensityDT:

    def __init__(self, representation_rate, train_p_data, test_p_data, init_qs, x_support, a_domain, seed=1337):

        self.seed = seed
        self.init_qs = init_qs

        self.q = FairDensity(init_qs, x_support, a_domain)

        self.train_p_samples = train_p_data
        self.test_p_samples = test_p_data

        self.representation_rate = representation_rate
        self.iter_count = 0

        self.p = None

        self.empirical_train = []
        self.empirical_test = []
        
        train_probs = {}
        for x, a in train_p_data:
            t_x = tuple(int(i) for i in x)
            t_a = tuple(int(i) for i in a)
            try:
                train_probs[(t_x, t_a)] += 1
            except KeyError:
                train_probs[(t_x, t_a)] = 1

        test_probs = {}
        for x, a in test_p_data:
            t_x = tuple(int(i) for i in x)
            t_a = tuple(int(i) for i in a)
            try:
                test_probs[(t_x, t_a)] += 1
            except KeyError:
                test_probs[(t_x, t_a)] = 1

        for x, a in itertools.product(self.q.x_support, self.q.a_domain):
            try:
                self.empirical_train.append(train_probs[(x, a)] / len(train_p_data))
            except:
                self.empirical_train.append(1e-7)
            try:
                self.empirical_test.append(test_probs[(x, a)] / len(test_p_data))
            except:
                self.empirical_test.append(1e-7)

    def init_boost(self, num_iter=10, max_depth=3):

        self.num_q_train = len(self.train_p_samples)
        self.num_q_test = len(self.test_p_samples)
        self.num_iter = num_iter

        self.max_depth = max_depth

        self.train_history = []
        self.boost_history = []

    def train_classifier(self, train_p_samples, train_q_samples, test_p_samples, test_q_samples):

        train_x = torch.cat([train_p_samples[0], train_q_samples[0]]).numpy()
        train_y = torch.cat([torch.ones(train_p_samples[0].shape[0]), -torch.ones(train_q_samples[0].shape[0])]).view(-1, 1).numpy()
        train_y = (train_y + 1) / 2

        test_x = torch.cat([test_p_samples[0], test_q_samples[0]]).numpy()
        test_y = torch.cat([torch.ones(test_p_samples[0].shape[0]), -torch.ones(test_q_samples[0].shape[0])]).view(-1, 1).numpy()
        test_y = (test_y + 1) / 2

        x_dim = train_x.shape[1]

        train_x = [tuple(map(int, r)) for r in train_x]
        train_y = [int(r[0]) for r in train_y]

        test_x = [tuple(map(int, r)) for r in test_x]
        test_y = [int(r[0]) for r in test_y]

        learner = Projection(x_dim)
        topdown = TopDown(learner, gini)
        dt = topdown((train_x, train_y), float('inf'), max_depth=self.max_depth)

        """
        #dt = DecisionTreeRegressor(random_state=self.seed)
        dt = DecisionTreeClassifier(random_state=self.seed, max_depth=5)
        #dt = RandomForestClassifier(random_state=self.seed, max_depth=5, n_estimators=200, max_features=2)
        dt.fit(train_x, train_y)
        """

        # Logging
        self.train_history.append({
            'iter': self.iter_count,
            'train_acc': score((train_x, train_y), dt),
            'test_acc': score((test_x, test_y), dt),
        })

        return dt

    def boost_iter(self, save_classifiers=None):

        train_q_samples = self.q.sample_n(self.num_q_train)
        test_q_samples = self.q.sample_n(self.num_q_test)

        train_p_samples = [torch.stack(v) for v in zip(*self.train_p_samples)]
        test_p_samples = [torch.stack(v) for v in zip(*self.test_p_samples)]

        trained_classifier = self.train_classifier(train_p_samples, train_q_samples, test_p_samples, test_q_samples)

        self.q.append(
            #m = lambda x: trained_classifier.predict(torch.Tensor(x).view(1, -1)),
            m = to_boost_clf(trained_classifier, self.q.x_support),
            #theta = - (1 / 2) ** (self.iter_count + 1) * math.log(self.representation_rate) / math.log(2)
            #theta = - (1 / (2 * self.iter_count)) * math.log(self.representation_rate) / math.log(2)
            theta = - (1 / (3 + self.iter_count)) * math.log(self.representation_rate) / math.log(2)
        )

        if save_classifiers:
            with open(os.path.join(save_classifiers, 'classifier_iter_{}.p'.format(self.iter_count)), 'wb') as f:
                pickle.dump(trained_classifier, f)

    def cur_boost_statistics(self):
        if self.p:
            kl = KL(self.p, self.q, self.q.x_support, self.q.a_domain)
        else:
            q_dist = self.q.get_prob_array()
            train_kl = entropy(self.empirical_train, q_dist)
            test_kl = entropy(self.empirical_test, q_dist)
            kl = (train_kl, test_kl)

        # Current Q Parameters
        if self.iter_count < 1:
            cur_theta = None
            cur_logz = None
        else:
            cur_theta = self.q.thetas[-1]
            cur_logz = self.q.logzs[-1]

        stats = {
            'iter': self.iter_count,
            'rr': self.q.representation_rate(),
            'kl': kl,
            'theta': cur_theta,
            'logz': cur_logz,
        }

        return stats

    def boost(self, verbose=True, save_classifiers=None):

        self.iter_count = 0
        self.boost_history.append(self.cur_boost_statistics())
        self.iter_count += 1

        if verbose:
            iterator = tqdm(range(self.num_iter))
        else:
            iterator = range(self.num_iter)

        for _ in iterator:
            self.boost_iter(save_classifiers=save_classifiers)
            self.boost_history.append(self.cur_boost_statistics())
            self.iter_count += 1

        return self.q, self.train_history