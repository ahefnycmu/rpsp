# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:48:17 2016

@author: ahefny

PolicyUpdate: provide a function that can update the policy using SGD in default.

Note the input structure is DIFFERENT from the classic PSR input structure:
        Here the trajectory of state and trajectory of control is orgnalized as follows:
        Trajectory of state:   s_1, s_2, ...., s_T
        Trajectory of control: a_1, a_2, ...., a_T, WHERE a_t ~ pi(a | s_t) !!!
        The total cost is defined as: \sum_i=1^T c(s_i, a_i).
"""
import numpy as np
import time
from IPython import embed

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
              
def learn_policy(policy_updater, model, environment, 
    num_trajs = 0, num_samples = 0, max_traj_len = 100, min_traj_length = 0,
    num_iter = 100, logger = None, trainer=None, freeze=True):
    best_avg = -np.inf
    retrain = False
    if not freeze:
        policy_updater._policy._psrnet._rffpsr.unfreeze()
    tic_all = time.time()
    for i in xrange(num_iter):
  
        trajs = environment.run(model, policy_updater.policy, max_traj_len,
                                num_trajs=num_trajs, min_traj_length=min_traj_length,
                                num_samples=num_samples)
        print 'iter=',i
        print 'Using %d trajectories with %d total samples' % (len(trajs), sum(t.length for t in trajs))
        if retrain:
            policy_updater = trainer.update(policy_updater, trajs)
        print 'update model'
        tic = time.time()
        res = policy_updater.update(trajs); 
        print 'done update model', time.time()-tic        

        m = np.mean([np.sum(t.rewards) for t in trajs]); 
        s = np.std([np.sum(t.rewards) for t in trajs]);
        if m > best_avg:
            best_avg = m
        print "iteration {}, avg rwd={:3.4f} (std={:3.4f}, best={:3.4f})".format(i,m,s,best_avg)
        
        if logger is not None:
            retrain = logger(i, trajs, res)
    print 'TOTAL LEARNING TIME: ', time.time()-tic_all
                        
def learn_model_policy(policy_updater, model, environment, num_trajs=10,
    max_traj_len=100, min_traj_length=0,
    num_iter=100, plot=False):
    best_avg = -np.inf
    
    for i in xrange(num_iter):
        print '\niteration ', i
        trajs = environment.run(model, policy_updater.policy, num_trajs,
                                max_traj_len, min_traj_length)
        
        # Restimate the state based in updated model
        for k in xrange(len(trajs)):
            model.filter(trajs[k])
        
        # Update policy
        tic=time.time()
        policy_updater.update(trajs)
    
        # Update the model based on the sampled trajectories
        model.update(trajs)
        n = sum(t.length for t in trajs)
        print 'policy update took: %.1f secs.'%(time.time()-tic)
          
        m = np.mean([np.sum(t.rewards) for t in trajs]); 
        s = np.std([np.sum(t.rewards) for t in trajs]);
        print "at iteration {}, the average reward is {} and std is {}.".format(i,m,s)
        if plot:
            # Restimate the state based in updated model
            for k in xrange(len(trajs)):
                model.filter(trajs[k])
            model.prediction_error(trajs, it=i)
    return 

