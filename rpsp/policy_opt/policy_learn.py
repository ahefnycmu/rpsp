# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Mon Nov 28 09:48:17 2016

@author: ahefny, zmarinho

PolicyUpdate: provide a function that can update the policy using SGD in default.

Note the input structure is DIFFERENT from the classic PSR input structure:
        Here the trajectory of state and trajectory of control is orgnalized as follows:
        Trajectory of state:   s_1, s_2, ...., s_T
        Trajectory of control: a_1, a_2, ...., a_T, WHERE a_t ~ pi(a | s_t) !!!
        The total cost is defined as: \sum_i=1^T c(s_i, a_i).
"""
import numpy as np
import time


class BasePolicyUpdater(object):
    def update(self, trajs):
        raise NotImplementedError

    @property
    def policy(self):
        raise NotImplementedError

    def _load(self):
        raise NotImplementedError

    def _save(self):
        raise NotImplementedError


def learn_policy(policy_updater, model, environment, num_trajs=0, num_samples=0, max_traj_len=100,
                 min_traj_length=0, num_iter=100, logger=None):
    best_avg = -np.inf
    tic_all = time.time()
    for i in xrange(num_iter):
        trajs = environment.run(model, policy_updater.policy, max_traj_len,
                                num_trajs=num_trajs, min_traj_length=min_traj_length,
                                num_samples=num_samples)
        print('iter=', i)
        print('Using %d trajectories with %d total samples' % (len(trajs), sum(t.length for t in trajs)))
        tic = time.time()
        res = policy_updater.update(trajs)

        m = np.mean([np.sum(t.rewards) for t in trajs])
        s = np.std([np.sum(t.rewards) for t in trajs])
        if m > best_avg:
            best_avg = m
        print("iteration {}, avg rwd={:3.4f} (std={:3.4f}, best={:3.4f})".format(i, m, s, best_avg))

        if logger is not None:
            logger(i, trajs, res)
    print('TOTAL LEARNING TIME: ', time.time() - tic_all)
