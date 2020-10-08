import torch
from scipy.integrate import nquad

def classifier_wrappers(c):
    return lambda x: c(torch.Tensor(x))[0]

# Expansion of cross entropy as per Hisham's suggestion (In fairdensity.py)
#def cross_entropy(p_samples, q_samples, c, q):
#    exp_p = torch.sum(torch.log(c(p_samples)))
#
#    exp_q = (1- torch.log(c(q_samples)))
#    for (theta, m, logz) in zip(q.thetas, q.models, q.logzs):
#        exp_q *= torch.exp(theta * m(q_samples)) / logz
#
#    return exp_p + exp_q


def accuracy(classifier, samples):
    p_samples = [tuple([s[0], s[1]]) for s in samples if s[2] == 1]
    q_samples = [tuple([s[0], s[1]]) for s in samples if s[2] == 0]

    p_x_samples, _ = zip(*p_samples)
    q_x_samples, _ = zip(*q_samples)

    p_x_samples = torch.stack(p_x_samples)
    q_x_samples = torch.stack(q_x_samples)

    p_predict = classifier(p_x_samples)
    q_predict = classifier(q_x_samples)

    acc = torch.cat([p_predict >= 0, q_predict < 0])

    return torch.sum(acc) / float(len(acc))

#def accuracy(classifier, p_samples, q_samples):
#    p_x_samples, _ = p_samples
#
#    print(len(p_x_samples))
#    print(len(_))
#
#    q_x_samples, _ = q_samples
#
#    p_predict = classifier(p_x_samples)
#    q_predict = classifier(q_x_samples)
#
#    acc = torch.cat([p_predict >= 0, q_predict < 0])
#
#    return torch.sum(acc) / float(len(acc))

def batch_weights(sample_size, batch_size):
    weights = [batch_size / sample_size] * (sample_size // batch_size)
    if (remainder := sample_size % batch_size) > 0:
        weights.append(remainder / sample_size)

    return torch.Tensor(weights)

def weighted_mean(values, weights):
    return float(torch.sum(torch.Tensor(values) * weights))

def all_binaries(domain_spec):
    bins = [[]]
    counter = 0
    for spec in domain_spec:
        new_bins = []
        
        if spec == 1:
            for b in bins:
                new_bins.append(tuple(list(b) + [0]))
                new_bins.append(tuple(list(b) + [1]))
        else:
            vals = [list([int(i) for i in t]) for t in torch.eye(spec)]
            for b in bins:
                for v in vals:
                    new_bins.append(tuple(list(b) + v))

        bins = new_bins
    return bins

def KL(p, q, x_support, a_domain, abstol=1e-3):
    kl = 0
    for a in a_domain:
        def func(x):
            x = torch.Tensor([x])

            pdf_p = torch.exp(p.log_prob(x, a))
            return pdf_p * p.log_prob(x ,a) - pdf_p * q.log_prob(x, a)

        opts = {
            'epsabs': abstol,
        }

        with torch.no_grad():
            cur_kl, _ = nquad(func, x_support, opts=opts)

        kl += cur_kl

    return kl