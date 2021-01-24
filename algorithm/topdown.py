import copy

from tree.dtree import DecisionNode, DecisionTree
from tree.tree import Direction

class BadOrder(Exception):
    pass

class TopDown:

    def __init__(self, learner, criterion):
        self.learner = learner
        self.criterion = criterion

    def __call__(self, samples, t, max_depth=3):
        _, ys = samples

        # Initialise T to be the single-leaf tree, majority labelled
        m_label = max(set(ys), key=ys.count)
        root = DecisionNode(Direction(m_label), None)
        dtree = DecisionTree(root)

        # While T has fewer than t internal nodes
        while dtree.size <= t or dtree.depth <= max_depth:

            # Setup best  
            best_drop = 0
            best_grow = None
            best_subsample = None
            best_proj_v = None
            for l, l_samp in partition(samples, dtree):
                cur_xs, cur_ys = l_samp

                if len(cur_ys) < 1:
                    continue
                #print(l, sum(cur_ys), len(cur_ys))

                # Probability of reaching leaf and having a 1 at leaf
                w = len(cur_ys) / len(ys)
                q = sum(cur_ys) / len(cur_ys)

                # Drop contribution of editing a single leaf
                def cur_drop_func(h):
                    pred_ys = h(cur_xs)
                    tau = sum(pred_ys) / len(cur_xs)

                    p = sum((1 - p_y) * c_y for (p_y, c_y) in zip(pred_ys, cur_ys)) / sum(1 - pred_ys)  # q(l0)
                    r = sum(p_y * c_y for (p_y, c_y) in zip(pred_ys, cur_ys)) / sum(pred_ys)  # q(l1)

                    if not(p < q and q < r):
                        raise BadOrder

                    # Drop will be negative of the decrease in criterion
                    return w * (self.criterion(q) - (1 - tau) * self.criterion(p) - tau * self.criterion(r))

                # Learn the best node splitter for current leaf
                drop, h, log = self.learner.learn(cur_drop_func)
                if drop > best_drop:
                    best_drop = drop
                    best_grow = (l, copy.deepcopy(h))
                    best_subsample = copy.deepcopy(l_samp)
                    best_proj_v = log

            # Grow tree as best
            if score(best_subsample, best_grow[0]) > score(best_subsample, best_grow[1]):
                break

            dtree.grow(*best_grow)

        return dtree

def partition(samples, dtree: DecisionTree):
    partition_x = {l.name: [] for l in dtree.leaves}
    partition_y = {l.name: [] for l in dtree.leaves}
    name_to_leaf = {l.name: l for l in dtree.leaves}

    # Partition data into tree leaves
    for s in zip(*samples):
        s_x, s_y = s
        l_n = dtree(s_x, eval_f=lambda x: x.name)

        partition_x[l_n].append(s_x)
        partition_y[l_n].append(s_y)

    # Return leaf + transformed samples
    return ((name_to_leaf[l_n], (partition_x[l_n], partition_y[l_n])) for l_n in name_to_leaf.keys())

def score(samples, dtree):
    cur_xs, cur_ys = samples
    correct = 0
    for x, y in zip(cur_xs, cur_ys):
        correct += int(dtree(x) == y)

    return correct / len(cur_xs)