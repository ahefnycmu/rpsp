from __future__ import print_function
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.sparse as ssp
import scipy.spatial
import matplotlib.pyplot as plt
from psr_models.utils import linalg as lg
from IPython import embed
from psr_models.utils.svdot import rbf_dot
from psr_models.utils.utils import *
from psr_models.utils.plot_utils import plot_predictions
from distutils.dir_util import mkpath
import cPickle as pickle
import scipy.linalg
plt.ion()

DEBUG=False

class Hankel_features(object):
    def __init__(self, use_actions=True, dim=5, filedir='examples/psr/data/features/hankel/', past_dim=0):
        self.use_actions = use_actions
        self.dim = dim
        self.past_dim = past_dim if past_dim<>0 else dim
        self.histories = []
        self.futures = []
        self.sfutures = []
        self.extfutures = []
        self.actions = []
        self.observations = []
        self.obs_mean = []
        #self.all_size = 0
        self.dO = 0
        self.dA = 0
        self.filedir=filedir
        mkpath(self.filedir)
        
    
    def build_all_features(self, x, z, verbose=DEBUG, add_const=False, all_size=0):
        ''' Build histories, futures, observations from Hankel matrix '''
        if verbose: print('select features')
        if all_size ==0:
            all_size = self.all_size
        if self.use_actions:
            self.actions      = z[:self.dA, self.dim:all_size-1]                                               # Current action
            self.observations = x[:self.dO, self.dim:all_size-1]                                               # Current observation
            if ssp.isspmatrix(x):
                self.histories    = ssp.vstack(( x[:self.past_dim, :all_size-self.dim-1], z[:self.past_dim, :all_size-self.dim-1] ), format='csc')  # Past sequences of observations
                self.futures      = ssp.vstack(( x[:self.dim, self.dim:all_size-1], z[:self.dim, self.dim:all_size-1] ), format='csc')    # Immediate future sequences of observations 
                self.sfutures   = ssp.vstack(( x[:self.dim, self.dim+1:all_size], z[:self.dim, self.dim+1:all_size] ), format='csc')    # Future one-step-removed sequences of observations
                #self.extfutures   = ssp.vstack(( x[:, self.dim:all_size], z[:, self.dim:all_size] ), format='csc')    # Future one-step-removed sequences of observations
            else:
                self.histories    = np.vstack(( x[:self.past_dim, :all_size-self.dim-1], z[:self.past_dim, :all_size-self.dim-1] )) # Past sequences of observations
                self.futures      = np.vstack(( x[:self.dim, self.dim:all_size-1], z[:self.dim, self.dim:all_size-1] ))   # Immediate future sequences of observations 
                self.sfutures   = np.vstack(( x[:self.dim, self.dim+1:all_size], z[:self.dim, self.dim+1:all_size] ))     # Future one-step-removed sequences of observations
                #self.extfutures   = np.vstack(( x[:, self.dim:all_size], z[:, self.dim:all_size] ))      # Future one-step-removed sequences of observations 
                #if add_const:
                self.extfutures = np.concatenate( [self.sfutures, self.observations, self.actions, np.ones((1,self.sfutures.shape[1]),dtype=float)], axis=0)
        else:
            self.histories    = x[:self.past_dim, :all_size-self.dim-1]    # Past sequences of observations
            self.futures      = x[:self.dim, self.dim:all_size-1]    # Immediate future sequences of observations 
            self.sfutures   = x[:self.dim, self.dim+1:all_size]    # Future one-step-removed sequences of observations
            #self.extfutures   = x[:, self.dim:all_size]    # Future one-step-removed sequences of observations
            self.observations = x[:self.dO, self.dim:all_size-1]  # Current observation
            #if add_const:
            #self.extfutures = np.concatenate( [self.sfutures, self.observations, np.ones((1,self.sfutures.shape[1]),dtype=float)], axis=0)
            self.actions = [];
        if verbose: print('histories', self.histories.shape)
        if verbose: print('futures', self.futures.shape)
        if verbose: print('extfutures', self.extfutures.shape)
        if verbose: print('futures', self.sfutures.shape)
        if verbose: print('actions', self.actions.shape)
        if verbose: print('observations', self.observations.shape)
        assert self.histories.shape[0]==self.futures.shape[0],'mismatch dim'
        return
    
    def build_unique_features(self, x, z, verbose=DEBUG, add_const=False, all_size=0):
        ''' Build histories, futures, observations from Hankel matrix '''
        if verbose: print('select features')
        if all_size==0:
            all_size=self.all_size
        if self.use_actions:
            self.actions      = z[:self.dA, self.dim:all_size-1]                                      # Current action
            self.observations = x[:self.dO, self.dim:all_size-1]                                          # Current observation
        else:
            self.observations = x[:self.dO, self.dim:all_size-1]  # Current observation
            self.actions = [];
        if verbose: print('histories', self.histories.shape)
        if verbose: print('futures', self.futures.shape)
        if verbose: print('extfutures', self.extfutures.shape)
        if verbose: print('futures', self.sfutures.shape)
        if verbose: print('actions', self.actions.shape)
        if verbose: print('observations', self.observations.shape)
        return

      
    def extract_joint_features(self, observations, actions, verbose=DEBUG):
        if verbose: print('compute test features ...')
        if verbose: print(self.all_size, self.dO, self.dA)
        if type(observations)==np.ndarray:
            x = self.build_hankel(observations)
        elif ssp.isspmatrix(observations):
            x = self.build_sparse_hankel(observations)
        if self.use_actions:
            if type(actions)==np.ndarray:
                z = self.build_hankel(actions)
            elif ssp.isspmatrix(observations):
                z = self.build_sparse_hankel(actions)
        if verbose: print(self.dA)
        return x,z 
    
    def compute_test_features(self, observations, actions, all_size=0, verbose=DEBUG, add_const=False):
        x,z = self.extract_features(observations, actions, all_size=all_size)
        self.build_unique_features(x,z, add_const=add_const)
        return     
    
    def compute_train_features(self, observations, actions, all_size=0, verbose=DEBUG, add_const=False):
        x,z = self.extract_features(observations, actions, all_size=all_size)
        self.build_all_features(x,z, add_const=add_const)
        return 
    
    def build_sparse_hankel(self, values, verbose=DEBUG, dim=0):
        ''' dxN samples from trajectories '''
        if verbose: print('Build Hankel...')
        if dim==0:
            dim = self.dim
        d,N = values.shape
        tau = N-dim
        x = ssp.lil_matrix( (dim*d, tau), dtype=float)
        for i in xrange(dim):
            x[i*d : (i+1)*d,:] = values[:, i:N-i]
        x = x.tocsc()
        if verbose: print(x.shape)
        return x
    
    def build_hankel(self, values, verbose=DEBUG, dim=0):
        ''' dxN samples from trajectories '''
        if verbose: print('Build Hankel...')
        if dim==0:
            dim = np.max([self.past_dim, self.dim+1])
        d,N = values.shape
        tau = N - dim 
        x = np.zeros((dim*d, tau), dtype=float)
        for i in xrange(dim):
            x[i*d : (i+1)*d,:] = values[:, i:(N-(dim-i))] #-1
        if verbose: print(x.shape)
        if self.past_dim > self.dim:
            delta = self.past_dim - self.dim - 1
            sub = x[x.shape[0]-(delta)*d:,-self.dim-1:]
            xx = np.zeros((x.shape[0],self.dim+1))
            xx[:sub.shape[0],:] = sub
            x = np.hstack([x, xx ])
        elif self.dim > self.past_dim:
            sub = x[:self.past_dim*d,:self.past_dim]
            xx = np.zeros((x.shape[0],self.past_dim))
            xx[-sub.shape[0]:,:] = sub
            x = np.hstack([xx, x]) 
        return x
    
    def extract_features(self, seq_obs, seq_actions, verbose=DEBUG, all_size=0):
        ''' build hankel matrix from observations'''
        if verbose: print('get sequence features and concatenate hankel...')
        if type(seq_obs) is not list:
            seq_obs = [seq_obs]
        if type(seq_actions) is not list:
            seq_actions = [seq_actions]
        self.dO = seq_obs[0].shape[0]
        self.dA = seq_actions[0].shape[0]
        self.num_seqs = len(seq_obs)
        #if all_size<>0: L = all_size;
        if all_size==0: all_size = seq_obs[0].shape[1]
        assert len(seq_obs)==len(seq_actions), 'length actions <> length obs!'
        M_seqs = 0.0
        self.all_size=0
        x_all=[]; z_all =[];
        self.obs_mean = np.zeros((self.dO), dtype=float)
        for j in xrange(self.num_seqs):# for each sequence
            assert self.dO == seq_obs[j].shape[0], embed() #'wrong obs dim on seq%d'%j
            assert self.dA == seq_actions[j].shape[0], 'wrong action dim on seq%d'%j
            if (seq_obs[j].shape[1] < (self.past_dim + self.dim + 2)): #1 extra for grad
                continue
            self.obs_mean =  self.obs_mean + seq_obs[j].sum(1)
            M_seqs += seq_obs[j].shape[1]
            x,z = self.extract_joint_features( seq_obs[j], seq_actions[j])
            x_all.append(x[:,:all_size])
            z_all.append(z[:,:all_size])
        if ssp.issparse(x_all[-1]):
            module=ssp
        else:
            module=np
        self.num_seqs = len(x_all)
        self.traj_locs = [x_all[j].shape[1] for j in xrange(len(x_all))]
        self.traj_locs.insert(0,0)
        self.traj_locs = np.cumsum(self.traj_locs)
        x_all = module.hstack(x_all)
        z_all = module.hstack(z_all)
        self.obs_mean = self.obs_mean * 1/float(M_seqs)
        self.all_size = x_all.shape[1] 
        #print ('all size is ', self.all_size)
        return x_all, z_all
    
    def load_features(self,filename='rff_feats'):
        f = open(self.filedir+filename+'.pkl','rb')
        self.observations = pickle.load(f)
        self.histories = pickle.load(f)
        self.futures = pickle.load(f)
        self.sfutures = pickle.load(f)
        self.extfutures = pickle.load(f)
        self.actions = pickle.load(f)
        f.close()
        return      

    def save_features(self, filename='rff_feats'):
        f = open(self.filedir+filename+'.pkl','wb')
        pickle.dump(self.observations, f)
        pickle.dump(self.histories, f)
        pickle.dump(self.futures, f)
        pickle.dump(self.sfutures, f)
        pickle.dump(self.extfutures, f)
        pickle.dump(self.actions, f)
        f.close()
        return
   
    
    def save_feature(self, feature, filename='rff_feats'):
        f = open(self.filedir+filename+'.pkl','wb')
        pickle.dump(feature, f)
        f.close()
        return
    
    def load_feature(self, feature, filename='rff_feats'):
        f = open(self.filedir+filename+'.pkl','rb')
        feature = pickle.load(f)
        f.close()
        return feature
    

if __name__ == '__main__':
    print('do cyclic rff test')
    
    train_trajs, train_spans, test_trajs, test_spans, obs_dim, action_dim =\
    get_cylinder_pushing_data("examples/sim_dataZ/", sparse=1, testSize=20, trainSize=100)
    size_end=4
    train_obs, train_actions = process_trajectories_unsup(train_trajs, obs_dim, action_dim, size_end)
    H_feats = Hankel_features(use_actions=True, dim=5, filedir='examples/psr/data/features/hankel/sim_dataZ/')
    H_feats.compute_train_features(train_obs.T, train_actions.T, add_const=True)
    
    embed()