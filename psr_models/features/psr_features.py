from __future__ import print_function
import numpy as np
import numpy.linalg
import time
import scipy as sp
import scipy.sparse
import scipy.spatial
import matplotlib.pyplot as plt
from psr_models.utils import linalg as lg
import cPickle as pickle
from IPython import embed
from psr_models.utils.utils import read_matrix
from psr_models.utils.svdot import rbf_dot
from psr_models.utils.utils import get_cylinder_pushing_data, append_trajectories,process_trajectories_unsup
#from psr_models.utils.sparse_utils import build_coo_outer_matrix
from psr_models.utils.plot_utils import plot_predictions
from psr_models.features.hankel_features import Hankel_features
from psr_models.utils.linalg import svd_f,svds, rand_svd_f
from distutils.dir_util import mkpath
import psr_models.utils.kernel as kern
import psr_models.utils.feats as feat
from psr_models.features.feat_extractor import *

plt.ion()
np.random.seed(100)
DEBUG=False

class structtype():
    pass

class PSR_features(object):
    def __init__(self, feat_set, params=None):
        self._fut = params['fut']
        self._past = params['past']
        self._p = params['dim']
        self._feature_set = feat_set
        self._dbg_preset_U_efo = params.get('Uefo', None) 
        self._dbg_preset_U_efa = params.get('Uefa', None)
        self._dbg_preset_U_oo = params.get('Uoo', None)
        self._dbg_preset_U_st = params.get('Ust', None) 
        self._frozen = False
        self._U_st = None
        
    def load(self, params):
        self._fut = params['fut']
        self._past = params['past']
        self._p = params['dim']
        self._dbg_preset_U_efo = params.get('Uefo', None) 
        self._dbg_preset_U_efa = params.get('Uefa', None)
        self._dbg_preset_U_oo = params.get('Uoo', None)
        self._dbg_preset_U_st = params.get('Ust', None)
        return
     
    def _extract_timewins(self, traj_obs, traj_act):
        data = structtype()
        d = structtype()

        bounds = (self._past, self._fut)                
        past_extractor = feat.finite_past_feat_extractor(self._past)
        fut_extractor = feat.finite_future_feat_extractor(self._fut)
        shifted_fut_extractor = feat.finite_future_feat_extractor(self._fut, 1)
        extended_fut_extractor = feat.finite_future_feat_extractor(self._fut+1)
        immediate_extractor = lambda X,t: X[:,t]
                        
        data.past_obs, data.series_index, data.time_index = \
            feat.flatten_features(traj_obs, past_extractor, bounds)
        data.past_act,_,_ = feat.flatten_features(traj_act, past_extractor, bounds)
        data.past = np.vstack((data.past_obs, data.past_act))
        data.fut_obs,_,_ = feat.flatten_features(traj_obs, fut_extractor, bounds)
        data.fut_act,_,_ = feat.flatten_features(traj_act, fut_extractor, bounds)
        data.shfut_obs,_,_ = feat.flatten_features(traj_obs, shifted_fut_extractor, bounds)
        data.shfut_act,_,_ = feat.flatten_features(traj_act, shifted_fut_extractor, bounds)
        data.exfut_act,_,_ = feat.flatten_features(traj_act, extended_fut_extractor, bounds)
        data.obs,_,_ = feat.flatten_features(traj_obs, immediate_extractor, bounds)
        data.act,_,_ = feat.flatten_features(traj_act, immediate_extractor, bounds)
                        
        d.h = data.past.shape[0]
        d.o = data.obs.shape[0]
        d.a = data.act.shape[0]
        d.fo = data.fut_obs.shape[0]
        d.fa = data.fut_act.shape[0]
        
        data.num_seqs = len(traj_obs)
        data.locs = np.where(data.series_index[1:]-data.series_index[:-1] ==1)[0].tolist()
        data.locs.append(len(data.series_index))
        data.locs.insert(0,0)
        data.d = d
        self._d = d
        return data
    
    def _extract_grads(self, traj_grads, data):
        bounds = (self._past, self._fut)   
        immediate_extractor = lambda X,t: X[:,t]
        data.grads,_,_ = feat.flatten_features(traj_grads, immediate_extractor, bounds)
        
        assert data.grads.shape[1] == data.obs.shape[1], 'bad gradient length'
        return
        
    def _extract_feats(self, data, gradients=True):                
        # Past
        feats = structtype()
        self._fext_past = self._feature_set['past']
        feats.past = self._fext_past.build(data.past).process(data.past)
        # Immediate
        self._fext_obs = self._feature_set['obs']
        feats.obs = self._fext_obs.build(data.obs).process(data.obs)
        self._fext_act =  self._feature_set['act']
        feats.act = self._fext_act.build(data.act).process(data.act)
        # Future
        self._fext_fut_obs = self._feature_set['fut_obs']
        feats.fut_obs = self._fext_fut_obs.build(data.fut_obs).process(data.fut_obs)
        self._fext_fut_act = self._feature_set['fut_act']
        feats.fut_act = self._fext_fut_act.build(data.fut_act).process(data.fut_act)
        # Shifted Future         
        feats.shfut_obs = self._fext_fut_obs.process(data.shfut_obs)
        feats.shfut_act = self._fext_fut_act.process(data.shfut_act)
        
        if gradients:
            feats.act_grad = self._fext_act.process_grad(data.act)
                    
        # Derived Features:
        # Extended Future
        # Note that for exteded future observation, the current observation is
        # the "lower order" factor. This makes filtering easier.        
        feats.exfut_obs = lg.khatri_dot(feats.shfut_obs, feats.obs)
        feats.exfut_act = lg.khatri_dot(feats.act, feats.shfut_act)  

    
        if self._dbg_preset_U_efo is None:
            self._U_efo,_,feats.exfut_obs = lg.rand_svd_f(feats.exfut_obs, k=self._p)
        else:
            self._U_efo = self._dbg_preset_U_efo
            feats.exfut_obs = np.dot(self._U_efo.T, feats.exfut_obs)
            
        if self._dbg_preset_U_efa is None:
            self._U_efa,_,feats.exfut_act = lg.rand_svd_f(feats.exfut_act, k=self._p)
        else:
            self._U_efa = self._dbg_preset_U_efa
            feats.exfut_act = np.dot(self._U_efa.T, feats.exfut_act)
        
        # Observation Covariance
        feats.oo = lg.khatri_dot(feats.obs, feats.obs)        
        
        if self._dbg_preset_U_oo is None:            
            self._U_oo,_,feats.oo = lg.rand_svd_f(feats.oo, k=self._p)
        else:
            self._U_oo = self._dbg_preset_U_oo
            feats.oo = np.dot(self._U_oo.T, feats.oo)
        
        K = structtype()
        K.obs = feats.obs.shape[0]        
        K.act = feats.act.shape[0]
        K.past = feats.past.shape[0]
        K.fut_obs = feats.fut_obs.shape[0]
        K.fut_act = feats.fut_act.shape[0]
        K.exfut_obs = feats.exfut_obs.shape[0]
        K.exfut_act = feats.exfut_act.shape[0]
        K.oo = feats.oo.shape[0]
              
        feats.locs = data.locs
        feats.num_seqs = data.num_seqs
        #feats.K = K
        self._feat_dim = K 
        return feats
        
    def freeze(self, value=True):
        self._fext_past.freeze(value)
        self._fext_obs.freeze(value)
        self._fext_act.freeze(value)
        self._fext_fut_obs.freeze(value)
        self._fext_fut_act.freeze(value)
        self._frozen = value
        if value:
            self._dbg_preset_U_efo = self._U_efo
            self._dbg_preset_U_efa = self._U_efa
            self._dbg_preset_U_oo = self._U_oo
            self._dbg_preset_U_st = self._U_st
        else:
            self._dbg_preset_U_efo = None
            self._dbg_preset_U_efa = None
            self._dbg_preset_U_oo = None
            self._dbg_preset_U_st = None
        return self._frozen
    
     
    def validate_trajectories(self, trajs, Lmin):
        idx = np.array([i for i in xrange(len(trajs)) if trajs.observations[i].shape[1] > (self._fut+self._past+Lmin)]) #need only 2 extra more for less variance
        trajs[:] = trajs.get(idx)
        return trajs
    
    def _get_fexts(self, base_fext):
        assert not self._frozen, 'frozen features cannot copy from base extractor.'
        assert base_fext._frozen, 'base features not frozen!'
        self._dbg_preset_U_efo = base_fext._U_efo
        self._dbg_preset_U_efa = base_fext._U_efa
        self._dbg_preset_U_oo = base_fext._U_oo
        if base_fext._U_st is not None:
            self._dbg_preset_U_st = base_fext._U_st
        return

    def compute_features(self, trajs, base_fext=None, Lmin=5):
        if base_fext is not None:
            self._get_fexts(base_fext)
        L = len(trajs)
        assert Lmin > 1, 'need at least Lmin=2 for skipped and gradients'
        trajs = self.validate_trajectories(trajs, Lmin)
        print ('valid trajs with min of ', len(trajs), ' out of ', L)
        assert len(trajs)>0, 'too small trajectories!'
        data = self._extract_timewins(trajs.observations, trajs.actions)
        feats = self._extract_feats(data)
        self._extract_grads(trajs.grads, data)
        return feats, data
      
    def build_state_features(self, feats, states):
        if self._dbg_preset_U_st is None:
            self._U_st,_,feats.states = lg.rand_svd_f(states, f=(lambda X: X), k=self._p)                      
        else:
            self._U_st = self._dbg_preset_U_st
            feats.states = np.dot(self._U_st.T, states)
        self._fext_st = lambda X: np.dot(self._U_st.T, X)
        self._feat_dim.st = self._U_st.shape[1]
        feats.st = self._feat_dim.st
        return feats.states
    
    def save(self, params):
        params['past'] = self._past
        params['fut'] = self._fut
        params['Uefo'] = self._U_efo
        params['Uefa'] = self._U_efa
        params['Uoo'] = self._U_oo
        params['Ust'] = self._U_st
        return
        



def test_extractors():
    A = np.array([[0,1,2,0,2,1], [1,3,2,0,1,1]])    
    
    G = np.zeros((12,6))
    for i in xrange(6):
        G[A[0,i]*4+A[1,i],i] = 1

    rffdim = 100
    rdim = 6
    
    f1 = IndicatorFeatureExtractor()    
    f1.build(A)
    F1 = f1.process(A)
    f1r = RandPCAFeatureExtractor(f1, rdim)
    f1r.build(A)
    F1r = f1r.process(A)

    assert np.all(F1 == G)
    
    f2 = RFFFeatureExtractor(rffdim).build(A)
    F2 = f2.process(A)
    f2r = RandPCAFeatureExtractor(f2, rdim)
    f2r.build(A)
    F2r = f2r.process(A)
    print (np.linalg.norm(F1r-F2r))
    embed()
    return

if __name__ == '__main__':

    test_extractors()
    
    