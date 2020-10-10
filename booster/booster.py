import math
import copy
import torch
import itertools

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import entropy

from densities.fairdensity import FairDensity
from densities.fairdensitycts import FairDensityCts
from classifiers.utils import (accuracy, batch_weights, weighted_mean, KL)

def collate(batch):
    return [[torch.Tensor(b[0]), torch.Tensor(b[1]), b[2]] for b in batch]

def collate_cts(batch):
    return [[torch.Tensor(b[0]), torch.LongTensor(b[1]), b[2]] for b in batch]

class BoostDensity:

    def __init__(self, representation_rate, train_p_data, test_p_data, init_qs, x_support, a_domain, model, true_dist=None, seed=1337):

        self.init_qs = init_qs
        self.init_model = model
        self.cur_model = None
        self.seed = seed

        if true_dist:
            self.q = FairDensityCts(init_qs, x_support, a_domain) 
        else:
            self.q = FairDensity(init_qs, x_support, a_domain) 

        self.train_p_samples = train_p_data
        self.test_p_samples = test_p_data

        self.representation_rate = representation_rate
        self.iter_count = 0

        if true_dist:
            self.collate_fn = collate_cts
        else:
            self.collate_fn = collate

        self.p = None
        if true_dist:
            self.p = true_dist
        else:
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

    def init_boost(self, batch_size=16, optimiser_gen=None, optimiser_settings={}, num_iter=10, num_epochs=200, early_stop=0.03, calc_pdf=False): #, num_q_train=3000, num_q_test=1000):

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.optimiser_gen = optimiser_gen
        self.optimiser_settings = optimiser_settings
        self.calc_pdf = calc_pdf

        self.num_q_train = 2 * len(self.train_p_samples)
        self.num_q_test = len(self.test_p_samples)

        self.train_history = []
        self.boost_history = []

        self.cur_train_gamma_p = 0
        self.cur_test_gamma_p = 0
        self.cur_train_gamma_q = 0
        self.cur_test_gamma_q = 0

    def train_classifier(self, optimiser, classifier, train_p_samples, train_q_samples,
                        test_p_samples, test_q_samples):

        ce_loss = self.q.gen_loss_function()

        # Set up train samples and test samples together
        train_samples = [tuple([*s, 1]) for s in train_p_samples] + [tuple([*s, 0]) for s in train_q_samples]
        test_samples = [tuple([*s, 1]) for s in test_p_samples] + [tuple([*s, 0]) for s in test_q_samples]

        self.cur_train_samples = train_samples
        self.cur_test_samples = test_samples

        for e in range(self.num_epochs):

            train_weights = []
            test_weights = []

            train_dataloader = DataLoader(train_samples, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

            # Training
            train_ce_list = []
            train_acc_list = []

            classifier.train()
            for train_batch in train_dataloader:

                optimiser.zero_grad()

                cur_train_ce = ce_loss(classifier, train_batch)#train_p, train_q)
                cur_train_ce.backward()
                optimiser.step()

            classifier.eval()
            with torch.no_grad():
                train_acc = accuracy(classifier, train_samples)
                train_ce = ce_loss(classifier, train_samples)

                test_acc = accuracy(classifier, test_samples)
                test_ce = ce_loss(classifier, test_samples)

            # Logging
            self.train_history.append({
                'iter': self.iter_count,
                'epoch': e,
                'train_ce': train_ce.item(),
                'train_acc': train_acc.item(),
                'test_ce': test_ce.item(),
                'test_acc': test_acc.item(),
            })

            # Early stopping
            early_stop_criterion = max(0, (test_ce - train_ce) / test_ce)
            if early_stop_criterion > self.early_stop and e > 100:
                break

        return classifier

    def boost_iter(self):

        cur_model = copy.deepcopy(self.init_model)
        cur_optim = self.optimiser_gen(cur_model.parameters(), **self.optimiser_settings)

        train_q_samples = self.q.sample_q_init(self.num_q_train)
        test_q_samples = self.q.sample_q_init(self.num_q_test)

        trained_classifier = self.train_classifier(cur_optim, cur_model, self.train_p_samples, train_q_samples, self.test_p_samples, test_q_samples)

        self.q.append(
            m = trained_classifier,
            theta = - (1 / 2) ** (self.iter_count + 1) * math.log(self.representation_rate) / math.log(2)
        )

        self.cur_train_gamma_p = torch.mean(trained_classifier(torch.stack([x for (x, _) in self.train_p_samples]))).item()
        self.cur_test_gamma_p = torch.mean(trained_classifier(torch.stack([x for (x, _) in self.test_p_samples]))).item()
        self.cur_train_gamma_q = torch.mean(-trained_classifier(torch.stack([x for (x, _) in train_q_samples]))).item()
        self.cur_test_gamma_q = torch.mean(-trained_classifier(torch.stack([x for (x, _) in test_q_samples]))).item()

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
            'train_gamma_p': self.cur_train_gamma_p,
            'test_gamma_p': self.cur_test_gamma_p,
            'train_gamma_q': self.cur_train_gamma_q,
            'test_gamma_q': self.cur_test_gamma_q,
        }

        if self.calc_pdf:
            pdf = []
            for i in range(len(self.q.a_domain)):
                x_vals = torch.linspace(-5, 5, 1001)
                a_vals = [self.q.a_domain[i] for _ in range(len(x_vals))]

                probs = torch.Tensor([torch.exp(self.q.log_prob(torch.Tensor([x]), a)) for x, a in zip(x_vals, a_vals)]).tolist()

                pdf.append(probs)

            stats['pdf'] = pdf

        return stats

    def boost(self, verbose=True):

        self.iter_count = 0
        self.boost_history.append(self.cur_boost_statistics())
        self.iter_count += 1

        if verbose:
            iterator = tqdm(range(self.num_iter))
        else:
            iterator = range(self.num_iter)

        for _ in iterator:
            self.boost_iter()
            self.boost_history.append(self.cur_boost_statistics())
            self.iter_count += 1

        return self.q, self.train_history