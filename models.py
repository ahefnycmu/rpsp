# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:12:31 2016

@author: ahefny, zmarinho
"""

import numpy as np
from IPython import embed

from psr_lite.psr_lite.utils.feats import ind2onehot


class TrajectoryList(list):
    #from itertools import imap, chain
    def __init__(self, a_dim=None, o_dim=None, start_traj=[],dim=0):
        super(TrajectoryList,self).__init__(start_traj)
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.dim = dim
        
    def _add(self, i, obs=None, state=None, action=None, reward=None):
        ''' add onto ith trajectory'''
        ti = self[i]
        self[i] = map(lambda x,y: np.vstack([x,y]) , ti, (obs, state, action, reward))
        return
        
    def _append(self, obs=None, states=None, actions=None, rewards=None):
        ''' append a trajectory'''
        self.append(self.zip(obs, states, actions, rewards))
        return
    
    def set(self, indeces):
        self[:] = self.get(indeces)
        return
        
    def get(self, indeces):
        trajs = TrajectoryList(a_dim=self.a_dim,o_dim=self.o_dim)
        for ind in xrange(len(indeces)):
            try:
                trajs+= [self[indeces[ind]]]
            except IndexError:
                embed()
        return trajs
    
    @property
    def observations(self):
        return map(lambda x: ind2onehot(x.obs.T, self.o_dim), self)
    @property
    def states(self):
        return map(lambda x: x.states.T, self)
    @property
    def actions(self):
        return map(lambda x: ind2onehot(x.act.T, self.a_dim), self)
    @property
    def rewards(self):
        return map(lambda x: x.rewards.T, self)
    @property
    def grads(self):
        return map(lambda x: x.policy_grads.T, self)
    @property 
    def len_max(self):
        lens = map(lambda x: len(x[0]), self)
        lens.append(0)
        return np.max(lens)
    @property
    def cum_reward(self):
        rwds = map(lambda x: np.sum(x.rewards), self)
        return rwds

'''
A trajectory consists of
- Observations
- States
- Actions
- Rewards   
- First observation
- First state
- env_states: full model state [pos,velocity]
- velocity: discrete velocity as in reward function
- bib: best in batch index (reported in the last trajectory)

These first 4 quantities are represented as matrices where each row 
corresponds to a timestep and should be interpreted as follows:
    At time t the agent executes action a_t and hence recieves observation
    and reward o_t and r_t. This information is used to update the belief
    state to q_t.
'''
class Trajectory:
    def __init__(self, obs=None, states=None, act=None, rewards=None, act_probs=None, obs0=None, state0=None,
                 policy_grads=None, env_states=None, rng=None, vel=None):
        self.obs = obs        
        self.states = states
        self.act = act
        self.rewards = rewards
        self.obs0 = obs0
        self.state0 = state0
        self.act_probs = act_probs
        self.policy_grads = policy_grads
        self.env_states = env_states
        self.rng = rng
        self.vel = vel
        self.bib = None #best in batch 

    @property
    def length(self):
        return self.obs.shape[0]

    @property
    def prestates(self):
        return np.vstack([self.state0.reshape((1,-1)), self.states[:-1]])
        
    # Implementation of [] operator so that Trajectory can still be
    # used as a tuple.
    def __getitem__(self, index):
        if index == 0:
            return self.obs
        elif index == 1:
            return self.states
        elif index == 2:
            return self.act
        elif index == 3:
            return self.rewards
        elif index == 4:
            return self.obs0
        elif index == 5:
            return self.state0
            
    def __setitem__(self, index, value):
        if index == 0:
            self.obs = value
        elif index == 1:
            self.states = value
        elif index == 2:
            self.act = value
        elif index == 3:
            self.rewards = value
        elif index == 4:
            self.obs0 = value
        elif index == 5:
            self.state0 = value
            
            

            
'''
Base class for filtering models, which are responsible for tracking the state
of the system.
'''
class FilteringModel(object):    
    def update(self, trajs):
        raise NotImplementedError
        
    def reset(self, first_observation):
        raise NotImplementedError
                    
    def update_state(self, o, a):
        raise NotImplementedError
    
    '''
    Returns post-states
    '''
    def compute_states(self, traj):
        n = traj.obs.shape[0]
        d = self.state_dimension
        
        states = np.empty((n,d))
        
        q = self.reset(traj.obs0)
        q0 = q
        
        for i in xrange(n):
            o = traj.obs[i,:]
            a = traj.act[i,:]
            q = self.update_state(o,a)
            states[i,:] = q

        return q0, states
        
    '''
    Returns pre-states
    '''
    def compute_pre_states(self, traj):
        q0, states = self.compute_states(traj)        
        return np.vstack((q0, states[:-1]))
        
    '''
    Updates the state fields of a trajectory object by filtering observations
    and actions. 
    '''        
    def filter(self, traj):
        q0, states = self.compute_states(traj)
        traj.state0 = q0
        traj.states = states        
                                       
    @property
    def state_dimension(self):
        raise NotImplementedError

'''
Class for generating training sets of trajectories to train models.
This implemetation uses all previous trajectories as a training set.
'''
class TrainSetGenerator(object):
    def __init__(self, max_size=np.inf, start_size=0, rng=None):
        self._trajs = TrajectoryList()
        self._max_size = max_size
        self.start_size = start_size
        self.rng = rng

    def update(self, trajs):
        size_before = len(self._trajs)
        self._trajs += trajs
        if self.total_size > self._max_size:
            delta = self.total_size - self._max_size
            print( 'total size: from %d to %d  (delete from %d to %d)'%(size_before, len(self._trajs), self.start_size, self.start_size+delta, ))
            del self._trajs[self.start_size:self.start_size+delta] #remove last delta trajectories before blind

    def gen_data(self):
        return self._trajs

    @property
    def total_size(self):
        return len(self._trajs)


# '''
# A variant of TrainSetGenerator that generate training sets by bootstrap
# sampling form previous trajectories.
# '''
# class BootstrapTrainSetGenerator(TrainSetGenerator):
#     def __init__(self, batch=None, **kwargs):
#         self._batch = batch if batch is not None else self.total_size
#         super(BootstrapTrainSetGenerator, self).__init__(*kwargs)
#
#     def gen_data(self):
#         n = len(self._trajs)
#         idx = np.random.choice(n,self._batch,replace=True)
#         out = self._trajs.get(idx)
#         return out
#
# class LastTrainSetGenerator(TrainSetGenerator):
#     def __init__(self, batch=None):
#         super(LastTrainSetGenerator, self).__init__()
#
#     def gen_data(self):
#         n = len(self._trajs)
#         idx = self.rng.choice(n,n,replace=True)
#         out = self._trajs.get(idx)
#         return out
#
#     def update(self, trajs):
#         self._trajs = TrajectoryList(start_traj=trajs)
#
#
# ''' Draw samples from exp distribution of previous iterations keep
# more samples on more recent trajectories'''
# class ExpTrainSetGenerator(TrainSetGenerator):
#     def __init__(self, lmbda, batch, **kwargs):
#         self._batch = batch
#         self._lmbda = lmbda
#         self._bins = [0]
#         super(ExpTrainSetGenerator, self).__init__(*kwargs)
#
#     def gen_data(self):
#         bins = len(self._bins)-1
#         ts = np.cumsum(self._bins)
#         if  ts[-1] <= self._batch:
#             return self._trajs
#         idx = np.random.exponential(self._lmbda, np.copy(self._batch))
#         size, _ = np.histogram(idx,bins=bins)
#         ids = []
#         for i in xrange(bins):
#             #print size[bins-i-1], ts[i],ts[i+1]
#             ids.extend(np.random.choice(np.arange(ts[i],ts[i+1]), size[bins-i-1], replace=True))
#         out = self._trajs.get(ids)
#         assert len(out)==self._batch, 'error batch size.'
#         return out
#
#     def update(self, trajs):
#         super(ExpTrainSetGenerator, self).update(trajs)
#         self._bins += [len(trajs)]
#
# ''' Draw samples exponentially from data ordered by cumulative reward
# '''
# class HighRewardTrainSetGenerator(TrainSetGenerator):
#     def __init__(self, lmbda, batch, n=5, **kwargs):
#         super(HighRewardTrainSetGenerator, self).__init__(*kwargs)
#         self._n = n
#         self._batch = batch
#         self._lmbda = lmbda
#         return
#
#     #store samples according to cumulative reward
#     def update(self, trajs):
#         self._trajs += trajs
#         self._trajs.sort( key=lambda x: x.rewards.sum(), reverse=False)
#         return
#
#     def gen_data(self):
#         if len(self._trajs)< self._batch:
#             return self._trajs
#         idx = np.random.exponential(self._lmbda, np.copy(self._batch))
#         size, _ = np.histogram(idx,bins=self._n)
#         #print (size)
#         p = np.round(100.0/float(self._n))
#         ts = [int(p/float(100.0)*len(self._trajs)*i) for i in xrange(1, self._n+1)]
#         ts.insert(0,0)
#         ids = []
#         for i in xrange(self._n-1, -1, -1):
#             ids.extend(np.random.choice(np.arange(ts[i],ts[i+1],1), size[i], replace=True))
#             #print size[i],ts[i],ts[i+1]
#         out = self._trajs.get(ids)
#         return out
'''
Class for filtering models that are trained in batch mode 
(i.e. do not support online updates)
'''
class BatchTrainedFilteringModel(FilteringModel):
    def __init__(self, batch_gen = TrainSetGenerator()):
        self._batch_gen = batch_gen        

    def train(self, trajs):
        raise NotImplementedError

    def update(self, trajs):
        self._batch_gen.update(trajs)
        batch = self._batch_gen.gen_data()
        self.train(batch)

'''
Represents a model of a fully observable system. Where the state at time t
is the observation at time t.
'''
class ObservableModel(FilteringModel):
    def __init__(self, obs_dim):
        self._obs_dim = obs_dim        
        
    def update(self, trajs):
        pass
        
    def reset(self, first_obs):
        return np.copy(first_obs)
        
    def update_state(self, o, a):
        return np.copy(o)
    
    @property
    def state_dimension(self):
        return self._obs_dim
   
'''
Represents a model that keeps a finite history of observations
'''        
class FiniteHistoryModel(FilteringModel):
    def __init__(self, obs_dim, past_window):
        self._obs_dim = obs_dim
        self._window_size = past_window
        
    def update(self, trajs):
        pass
    
    def reset(self, first_obs):
        fo = np.copy(first_obs)
        self._state = np.tile(fo, (self._window_size))
        return np.copy(self._state) 

    def update_state(self, o, a):
        self._state[:-self._obs_dim] = np.copy(self._state[self._obs_dim:])
        self._state[-self._obs_dim:] = np.copy(o)
        return np.copy(self._state)
        
    @property
    def state_dimension(self):
        return self._obs_dim*self._window_size
    
# class FiniteDeltaHistoryModel(FilteringModel):
#     def __init__(self, obs_dim, past_window, init_state=None, dt=1.0):
#         self._obs_dim = 2*obs_dim
#         self._dt = dt
#         self._init_state = init_state.copy() if init_state is not None else np.zeros((self._obs_dim * past_window))
#         assert self._init_state.size == self._obs_dim * past_window
#
#     def update(self, trajs):
#         pass
#
#     def reset(self, first_obs):
#         self._state = self._init_state.copy()
#         self._state[-self._obs_dim:-self._obs_dim/2] = first_obs
#         self._state[-self._obs_dim/2:] = np.zeros(self._obs_dim/2)
#         return self._state
#
#     def update_state(self, o, a):
#         last_obs = np.copy(self._state[-self._obs_dim:-self._obs_dim/2])
#         self._state[:-self._obs_dim] = self._state[self._obs_dim:]
#         self._state[-self._obs_dim:-self._obs_dim/2] = o
#         self._state[-self._obs_dim/2:] = (o-last_obs)/float(self._dt) #same as last_obs imediate velocity
#         return self._state
#
#     #@property
#     #def initial_state(self):
#     #    return self._init_state
#
#     @property
#     def state_dimension(self):
#         return self._init_state.size
    
# '''
# Represents a model of a fully observable system. Where the state at time t
# is the observation at time t.
# '''
# class ConstantModel(FilteringModel):
#     def __init__(self, obs_dim, init_state=None):
#         self._obs_dim = obs_dim
#         self._init_state = init_state.copy() if init_state is not None else np.random.normal(size=(self._obs_dim))
#
#     def update(self, trajs):
#         pass
#
#     def reset(self, first_obs):
#         return first_obs
#
#     def update_state(self, o, a):
#         return self.initial_state
#
#     @property
#     def initial_state(self):
#         return self._init_state
#
#     @property
#     def state_dimension(self):
#         return self._obs_dim
#
#     def train(self):
#         pass
    
    
    
# class ZeroModel(FilteringModel):
#     def __init__(self, obs_dim, init_state=None):
#         self._obs_dim = obs_dim
#         self._init_state = np.zeros(shape=(self._obs_dim))
#
#     def reset(self, first_obs):
#         return np.zeros(shape=(self._obs_dim))
#
#     def update_state(self, o, a):
#         return np.zeros(shape=(self._obs_dim))
#
#     @property
#     def state_dimension(self):
#         return self._obs_dim
#