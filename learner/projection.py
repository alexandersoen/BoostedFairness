import copy
import numpy as np

from learner.learner import Learner

class Projection(Learner):

    def __init__(self, dim_size):
        self.dim_size = dim_size

    def learn(self, drop):
        best_proj_v = None
        best_drop = 0
        best_flip = None
        log = None

        # Go over each dimension
        for flip in (True, False):
            for i in range(self.dim_size):
                proj_v = np.zeros(self.dim_size)
                proj_v[i] = 1

                # Projection function
                def proj(xs):
                    xs = np.array(xs)
                    if flip:
                        return 1 - xs @ proj_v
                    else:
                        return xs @ proj_v

                try:
                    cur_drop = drop(proj)
                except:
                    continue

                # Check for new best
                if cur_drop > best_drop:
                    best_proj_v = copy.deepcopy(proj_v)
                    best_drop = cur_drop
                    best_flip = copy.deepcopy(flip)

        # Best
        def best_proj(xs):
            xs = np.array(xs)
            if best_flip:
                return 1 - xs @ best_proj_v
            else:
                return xs @ best_proj_v

        return best_drop, best_proj, best_proj_v