import math
import copy
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

from densities.fairdensity import FairDensity
from classifiers.utils import (accuracy, batch_weights, weighted_mean)

def collate(batch):
    return [[torch.Tensor(b[0]), torch.Tensor(b[1]), b[2]] for b in batch]

class BoostDensity:

    def __init__(self, representation_rate, train_p_data, test_p_data, init_qs, x_support, a_domain, model, seed=1337):

        self.init_qs = init_qs
        self.init_model = model
        self.cur_model = None
        self.seed = seed

        self.q = FairDensity(init_qs, x_support, a_domain) 

        self.train_p_samples = train_p_data
        self.test_p_samples = test_p_data

        self.representation_rate = representation_rate
        self.iter_count = 0

    def init_boost(self, batch_size=16, optimiser_gen=None, optimiser_settings={}, num_iter=200, num_epochs=200, early_stop=0.03): #, num_q_train=3000, num_q_test=1000):

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.optimiser_gen = optimiser_gen
        self.optimiser_settings = optimiser_settings

        self.num_q_train = len(self.train_p_samples)
        self.num_q_test = len(self.test_p_samples)

        self.train_history = []
        self.boost_history = []

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

            train_dataloader = DataLoader(train_samples, batch_size=self.batch_size, shuffle=True, collate_fn=collate)

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
            theta = - (1 / 2) ** (self.iter_count + 1) * math.log(self.representation_rate)
        )

        self.iter_count += 1

    def cur_boost_statistics(self):
        return {
            'iter': self.iter_count,
            'rr': self.q.representation_rate(),
        }

    def boost(self):

        self.iter_count = 0
        self.boost_history.append(self.cur_boost_statistics())
        self.iter_count += 1

        for _ in tqdm(range(self.num_iter)):
            self.boost_iter()
            self.boost_history.append(self.cur_boost_statistics())

        return self.q, self.train_history