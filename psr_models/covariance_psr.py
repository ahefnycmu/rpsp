#!/usr/bin/python
"""
Created on Thu Jul 21 11:21:37 2016

@author: zmarinho
"""
from __future__ import print_function
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.sparse as ssp
import scipy.sparse.linalg
import scipy.spatial as spp
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython import embed
import psr_models.utils.linalg as lg
from psr_models.utils.svdot import rbf_dot,rbf_sparse_dot
from psr_models.utils.utils import *
import time
import cPickle as pickle
from psr_models.utils.plot_utils import plot_predictions,plot_modes, plot_PCs
from psr_models.features.hankel_features import Hankel_features
from psr_models.features.rff_features import RFF_features
from psr_models.hsepsr import HSEPSR
from psr_models.utils.regression import *
import psr_models.utils.numeric as num
import psr_models.utils.feats as util_feat
from psr_models.features.feat_extractor import create_RFFPCA_featureset
from psr_models.features.psr_features import PSR_features
plt.ion()
np.set_printoptions(suppress=True)
DEBUG=False
import cProfile, pstats, StringIO
pr = cProfile.Profile()
s=StringIO.StringIO('results/time.txt')
sortby = 'time'

def profile():
    pr.disable()
    ps = pstats.Stats(pr,stream=s).sort_stats(sortby);
    ps.print_stats();
    print(s.getvalue());
    return

class structtype():
    pass



class covariancePSR(object):
    def __init__(self, feat_ext, params={}, env= None):
        self._params = params
        self._start = np.ones((self._params['dim'],1), dtype=float)/ float(self._params['dim'])
        self._feature_extractor = feat_ext
        self._env = env
        self._W_s2fut = None
        return
    
    def get_dim(self):
        return self._params['dim']
    
      
    def train(self, feat_ext, feats, data, imp_weights=[None,None]):
        #pr.enable()
        self.K = feat_ext._feat_dim
        self.feature_extractor = feat_ext
        
        if DEBUG: print ('Stage 1 regression')
        self._s1_regression_joint( feats, imp_weights)
         
        if DEBUG: print('Stage 2 Regression')
        states = self._s2_regression(feats)
        self._create_mappings(feats, data, reg=self._params['reg'])
        
        #filter states
        self._build_model( feats )
        self._create_parameters()
        #profile()
        return {'data':data, 'feats':feats, 'imp_weights':imp_weights, 'states':feats.states,
                'ex_states':self._ext_states, 'im_states':self._im_states}
       
    def _s2_regression(self, feats):                
        self.feature_extractor.build_state_features(feats, self._fut_states)
        self.K = self.feature_extractor._feat_dim
        self._params['dim'] = self.K.st
        
        self._W_s2ext = self.conditional_op(feats.states, self._ext_states, self._params['reg'])
        self._W_s2oo = self.conditional_op(feats.states, self._im_states, self._params['reg'])
        return feats.states
                            
    def _s1_regression_joint(self, feats, imp_weights):        
        self._fut_states = self._s1_joint_conditional_operator(feats.fut_obs, feats.fut_act, feats.past,\
                                                                self.K.fut_obs, self.K.fut_act, self._params['reg'], imp_W=imp_weights[0])
        #print('fut_states cond: ', np.linalg.cond(self._fut_states))
        
        self._ext_states = self._s1_joint_conditional_operator(feats.exfut_obs, feats.exfut_act, feats.past,\
                                                                self.K.exfut_obs, self.K.exfut_act, self._params['reg'], imp_W=imp_weights[1])
        #print('exfut_states cond: ', np.linalg.cond(self._ext_states))
        self._im_states = self._s1_joint_conditional_operator(feats.oo, feats.act, feats.past,\
                                                                self.K.oo, self.K.act, self._params['reg'], imp_W=None)
        #print('im_states cond: ', np.linalg.cond(self._im_states))
        return 
        
    def _s1_joint_conditional_operator(self, output, input, instrument, do, di, reg=None, imp_W=[None,None]):
        if reg is None:
            reg = self._params['reg']
        Coa, Caa = denoise_cov([output, input], instrument, reg=reg, trainer=joint_ridge_regression, const=False, imp_W=imp_W)
        states = np.zeros(shape=Coa.shape, dtype=float)
        for j in xrange(Coa.shape[1]):
            denoised_i = Caa[:,j].reshape((di,di), order='F')
            denoised_o = Coa[:,j].reshape((do,di), order='F')
            states[:,j] = self.conditional_op(denoised_i,denoised_o, reg=reg).reshape(-1, order='F')
        return states #return W optionally  on denoise_cov
      
    def _build_model(self, feats, states0=None):
        states = self._update_states(feats, states0)
        self._build_one_predictor(feats)
        self._build_horizon_predictor( feats)
        return states
    
    def _update_states(self, feats, states0=None):
        feats.states = self.filter_trajs(states0=states0, feats=feats)
        self.states_proj = self.conditional_op(feats.past , feats.states)   # past states #W.s1a_proj
        self._start = np.mean(feats.states, axis=1) #check!!
        self.state = self._start
        return feats.states
    
    
    def _create_mappings(self, feats, data, reg=None):
        if reg is None:
            reg = self._params['reg']
        self.map={}
        self.map['rff2obs'] = self.conditional_op(feats.obs, data.obs)
        self.map['oo2obs'] = self.conditional_op(feats.oo, data.obs, reg=reg)
        self.map['oo2rff'] = self.conditional_op(feats.oo, feats.obs, reg=reg)
        self.map['rff2fut'] = self.conditional_op(feats.fut_obs, data.fut_obs, reg=reg)
        return
    
    def _build_horizon_predictor(self, feats):
        action_cond = lg.khatri_dot(feats.states, feats.fut_act)
        W2fut = self.conditional_op(action_cond, feats.fut_obs) #conditional operator Wfut_o|fut_a,h 
        self._W_s2fut = np.dot( self.map['rff2fut'], W2fut) #from fut_o_fx to futures_o |fut_a,h
        return
    
    def _build_one_predictor(self, feats):
        action_cond = lg.khatri_dot(feats.states, feats.act)
        W2oo = self.conditional_op(action_cond, feats.oo) #conditional operator Wfut_o|fut_a,h
        self._W_s2obs = np.dot( self.map['oo2obs'], W2oo )#.reshape((self.K.oo,-1), order='F') ) # observation operator
        return
    
    def conditional_op(self, X, Y, reg=None):
        if reg is None:
            reg = self._params['reg']
        W = ridge_regression(X, Y, reg=reg, add_const=False).W   
        return  W
    
    def compute_state_features(self, past):
        states = np.dot(self.states_proj, past)
        return states
       
    def filter_trajs(self, states0=None, feats=None):
        if states0 is None:
            states0 = np.vstack([feats.states[:,feats.locs[t]] for t in xrange(feats.num_seqs)] ).T
        states = []
        for t in xrange(feats.num_seqs):
            start = feats.locs[t]
            L = feats.locs[t+1] - feats.locs[t]
            states.append( self.iterative_filter(feats, states0[:,t], L=L,i=start)[0] )
        states = np.hstack(states)
        return states
    
    def filter(self, f, obs, a=np.nan):
        if obs.ndim==1:
            obs = obs.reshape(-1,1)
        if a.ndim==1:
            a = a.reshape(-1,1)
        feat_ext = self.feature_extractor
        #convert to rff history features
        o_fx = feat_ext._fext_obs.process(obs)
        if not np.isnan(a).any():
            a_fx = feat_ext._fext_act.process(a)
        sf, ops = self.filter_core(f, o_fx, a_fx=a_fx)
        return sf
    
    def predict(self, f, a=np.nan):
        feat_ext = self.feature_extractor
        if a.ndim==1:
            a = a.reshape(-1,1)
        if not np.isnan(a).any():
            #convert action to rff features
            a_fx = feat_ext._fext_act.process(a)
        if f.ndim==1:
            assert f.shape[0]==self.K.st, embed()
            f = f.reshape((self.K.st, -1))
        # convert state to 1step predictor and condition on action
        obs = np.dot( self._W_s2obs, lg.khatri_dot(f, a_fx) )
        #obs = self._validate_prediction(obs)
        return obs
    
    def predict_future(self, f, a=np.nan):
        feat_ext = self.feature_extractor
        if f.ndim==1:
            f = f.reshape((-1,1))
        if a.ndim==1:
            a = a.reshape((-1,1))
        #convert action to rff features
        fut_a_fx = feat_ext._fext_fut_act.process(a) #.reshape(self.K.fut_act,-1)
        # convert state to fut predictor and condition on action
        fut = np.dot( self._W_s2fut, lg.khatri_dot(f, fut_a_fx) )
        #fut = self._validate_prediction(fut.reshape((self._params['dim'], -1), order='F').T).T.reshape((-1,1), order='F') 
        return fut
    
    def _validate_prediction(self, prediction, mino=-3,maxo=3):
        if self._env is not None:
            o_min = [mino]*len(self._env._visible_idx)
            o_max = [maxo]*len(self._env._visible_idx)
            np.clip( prediction.T, o_min, o_max, out=prediction.T)
        return prediction

    def iterative_filter(self, feats, f, L=0, i=0):
        ''' Filter iterativel for the length of the features size of sequence'''
        if L==0: 
            N = feats.obs.shape[1] #length of trajectory
            L = np.divide(N, feats.num_seqs)
        states = np.zeros((self.K.st, L), dtype= float)
        filter_traj = []
        
        # Update states (Feed forward)  
        for l in xrange(i,i+L,1):
            states[:,l-i] = f.squeeze()
            f, ftraj  = self.filter_core( f, feats.obs[:,l], a_fx = feats.act[:,l])
            assert len(ftraj)>0, embed()
            assert not np.isnan(f).any(), embed()
            filter_traj.append(ftraj)
        return states, filter_traj

    #just for a derivative debug purposes
#     def filter_a(self, f, o_fx, a_fx1=np.nan, a_fx2=np.nan, Woo=None, Wex=None):
#         if Woo is None:
#             Woo = self._W_s2oo
#         if Wex is None:
#             Wex = self._W_s2ext 
#         #condition Coo on action 
#         feat_ext = self.feature_extractor
#         if len(a_fx1.shape)==1:
#             a_fx2 = a_fx2.reshape((-1,1), order='F')
#             a_fx1 = a_fx1.reshape((-1,1), order='F')
#             o_fx = o_fx.reshape((-1,1), order ='F')
# 
#         Coo_a_fx = np.dot( np.dot( Woo, f).reshape((-1,self.K.act), order='F'), a_fx1) #C_oo_prj
#         Coo_a = np.dot( feat_ext._U_oo, Coo_a_fx ).reshape(self.K.obs, self.K.obs) #undo projection
#         o_cond = self.conditional_op(Coo_a, o_fx.T) 
#         
#         #condition C_ext on action
#         #get extended state
#         Cexto_exta_fx = np.dot( Wex, f).reshape((-1,self.K.exfut_act), order='F')
#         Uextfut_a = feat_ext._U_efa.reshape(-1, self.K.act)
#         
#         extfut_fx_a = np.dot( Uextfut_a, a_fx2 ).reshape((self.K.exfut_act,self.K.fut_act), order='F') # B 
#         
#         Cexto_futa = np.dot(Cexto_exta_fx, extfut_fx_a) #condition on ext_a #C_eto_ta
#         
#         #project
#         Uextfut_o = feat_ext._U_efo.reshape((self.K.obs,-1), order='F')  #UU
#         Cexto_a = np.dot(o_cond, Uextfut_o ).reshape((self.K.fut_obs, self.K.exfut_obs), order='F') #A #undo projection of extended obs
#         sf = np.dot(Cexto_a, Cexto_futa).reshape((self.K.fut_obs*self.K.fut_act,1), order='F') #condition on o  Q_t+1
#         sf = np.dot( feat_ext._U_st.T, sf ) #shifted state Q_t+1 projected
#         
#         filter_ops = (o_cond.T, Coo_a_fx, Cexto_futa, Cexto_a, extfut_fx_a ) #v, C_oo_prj,C_eto_ta,A,B
#         assert not np.isnan(filter_ops[0]).any(), embed()
#         return sf, filter_ops
    
    def filter_core(self, f, o_fx, a_fx=np.nan, Woo=None, Wex=None):
        feat_ext = self.feature_extractor
        if Woo is None:
            Woo = self._W_s2oo
        if Wex is None:
            Wex = self._W_s2ext 
        #condition Coo on action 
        if len(a_fx.shape)==1:
            a_fx = a_fx.reshape((-1,1), order='F')
            o_fx = o_fx.reshape((-1,1), order ='F')

        Coo_a_fx = np.dot( np.dot( Woo, f).reshape((-1,self.K.act), order='F'), a_fx) #C_oo_prj
        Coo_a = np.dot( feat_ext._U_oo, Coo_a_fx ).reshape(self.K.obs, self.K.obs) #undo projection
        o_cond = self.conditional_op(Coo_a, o_fx.T)
        
        #condition C_ext on action \ get extended state
        Cexto_exta_fx = np.dot( Wex, f).reshape((-1,self.K.exfut_act), order='F')
        
        Uextfut_a = feat_ext._U_efa.reshape((-1, self.K.act), order='F') #order F or not?????????????????????????????????????????????????????
        # exfut_a x a  x N
        extfut_fx_a = np.dot( Uextfut_a, a_fx ).reshape((self.K.exfut_act,self.K.fut_act), order='F') # B 
        
        Cexto_futa = np.dot(Cexto_exta_fx, extfut_fx_a) #condition on ext_a #C_eto_ta
        
        #project
        Uextfut_o = feat_ext._U_efo.reshape((self.K.obs,-1), order='F')  #UU
        Cexto_a = np.dot(o_cond, Uextfut_o ).reshape((self.K.fut_obs, self.K.exfut_obs), order='F') #A #undo projection of extended obs
        sf = np.dot(Cexto_a, Cexto_futa).reshape((self.K.fut_obs*self.K.fut_act,1), order='F') #condition on o  Q_t+1
        sf = np.dot( feat_ext._U_st.T, sf ) #shifted state Q_t+1 projected
        
        filter_ops = (o_cond.T, Coo_a_fx, Cexto_futa, Cexto_a, extfut_fx_a ) #v, C_oo_prj,C_eto_ta,A,B
        assert not np.isnan(filter_ops[0]).any(), embed()
        
        return sf, filter_ops
    
    def _create_parameters(self):
        self.model = {}
        self.model['Wex'] = np.copy(self._W_s2ext)
        self.model['Woo'] = np.copy(self._W_s2oo)
        self.model['Wfut'] = np.copy(self._W_s2fut)
    
    def get_parameters(self):
        return self.model
    
    def _set_parameters(self, model):
        self._W_s2ext = np.copy(model['Wex'])
        self._W_s2oo = np.copy(model['Woo'])
        self._W_s2fut = np.copy(model['Wfut'])
        return
    
    def _load_model(self, filename):
        try:
            f = open(filename, 'rb')
            W_s2ext = pickle.load(f)
            W_s2oo = pickle.load(f)
            W_s2fut = pickle.load(f)
            maps = pickle.load(f)
            settings = pickle.load(f)
            f.close()
            print ('Load successful! ', filename)
            assert W_s2ext.shape==(self.K.exfut_obs*self.K.exfut_act, self._params['dim']), 'W2ext wrong dim!'
            assert W_s2oo.shape==(self.K.oo*self.K.act, self._params['dim']), 'W2oo wrong dim!'
            assert W_s2fut.shape==(self._feature_extractor._d.fo, self._params['dim']), 'W2fut wrong dim!'
            assert maps['rff2fut'].shape==(self._feature_extractor._d.fo, self.K.fut_obs), 'rff2fut wrong dim!'
            assert maps['oo2obs'].shape==(self._feature_extractor._d.o, self.K.oo), 'oo2obs wrong dim!'
            for (k,v) in settings:
                if k in self._params:
                    assert self._params[k] == settings[k], '%s is different from saved model '%k
            self._W_s2ext = W_s2ext
            self._W_s2oo = W_s2oo
            self._W_s2fut = W_s2fut
            self.map = maps
            self.feature_extractor.load(settings)
            self._params = settings
        except Exception:
            print ('Failed to load!!')
            embed()
        return
    
    def save_model(self, filename):
        self._feature_extractor.save(self._params)
        assert self._params['Ust'] is not None
        try:
            f = open(filename, 'wb')
            pickle.dump(self._W_s2ext, f)
            pickle.dump(self._W_s2oo, f)
            pickle.dump(self._W_s2fut, f)
            pickle.dump(self.map)
            pickle.dump(self._params)
            f.close()
        except Exception:
            print('Failed')
            embed()
        return
    
    def backprop(self, g_sf, f, o_fx, a_fx, model, reg, filter_ops, g_a=None):
        ''' backpropagation for psr models with covariance'''
        if DEBUG: print('rff backprop')
        feat_ext = self.feature_extractor
        o_cond = filter_ops[0]
        Coo_a_fx = filter_ops[1]
        Cexto_futa = filter_ops[2]
        Cexto_a = filter_ops[3]
        extfut_fx_a = filter_ops[4]
        g_Usf = np.dot(g_sf , feat_ext._U_st.T)
        Q = np.dot( g_Usf.reshape(-1,self.K.fut_obs) , Cexto_a)
        g_Cex = np.dot(Q.T, extfut_fx_a.T).reshape((1,-1), order='F')
        
        g_Wex = lg.khatri_dot(f.reshape(-1,1),g_Cex.T).T #np.einsum('i,j->ij',f, g_Cex).reshape((1,-1))        
        g_f1 = np.dot(g_Cex , model['Wex'])
        
        UC_ex = np.dot(feat_ext._U_efo , Cexto_futa).reshape((self.K.obs, -1), order='F')
        g_v = np.dot(g_Usf , UC_ex.T)
        
        C_oo = np.dot(feat_ext._U_oo , Coo_a_fx).reshape(self.K.obs, self.K.obs)        
        C_oo2 = np.dot(C_oo.T , C_oo)
        diago= np.diag_indices(self.K.obs)
        C_oo2[diago] = C_oo2[diago] + reg     
        
        gviCoo2 = np.linalg.solve(C_oo2,g_v.T).T
        g_Cooprj_1 = -lg.khatri_dot(o_cond,np.dot(gviCoo2 , C_oo).T).reshape((-1), order='F')
        g_Cooprj_2 = -lg.khatri_dot(np.dot(C_oo,o_cond), gviCoo2.T).reshape((-1), order='F')
        g_Cooprj_3 = lg.khatri_dot( o_fx[:,None], gviCoo2.T).reshape((-1), order='F')
                
        g_Cooprj = np.dot( (g_Cooprj_1+g_Cooprj_2+g_Cooprj_3) , feat_ext._U_oo)
        
        g_f2f = np.dot(g_Cooprj , model['Woo'].reshape((self.K.oo,-1),order='F'))# not a vector
        g_f2 = (g_f2f.reshape((self.K.act, self.K.st), order='F').T * a_fx).sum(1)
    
        g_f = (g_f1.reshape(-1) + g_f2)
        g_Woo = lg.khatri_dot(lg.khatri_dot(f[:,None], a_fx[:,None]), g_Cooprj[:,None]).T
        if g_a is not None:
            #ga1
            g_a1 = np.dot(lg.khatri_dot(f[:,None], g_a[:,None]).T,   g_f2f )
            #ga2
            g_extfut_a = np.dot(feat_ext._U_efa.reshape(-1, self.K.act), g_a)\
                                .reshape((self.K.exfut_act,self.K.fut_act), order='F')
            g_ext_a = np.dot(Q.T, g_extfut_a.T).reshape((1,-1), order='F')
            g_a2 = np.dot( np.dot(model['Wex'], f), g_ext_a.T)
            gsf_a = g_a1 + g_a2
            return g_f, g_Wex, g_Woo, gsf_a, g_a1, g_a2
        return g_f, g_Wex, g_Woo
    
#     def bp_grad(self, states, filter_traj, feats, data, model, seq=0, reg=1e-6,wpred=1.0,policy_grad=0.0):
#         start = feats.locs[seq]
#         end = feats.locs[seq+1]
#         n = states.shape[1] #number of points
#         # Difference between predicted and actual observations
#         diff_all = data.fut_obs[:,start:end] - np.dot( model['Wfut'], lg.khatri_dot(states,feats.fut_act[:,start:end]) )
#         g_sf = np.zeros((1,self.K.st), dtype=float)
#         g_a = []
# 
#         a_k_d = lg.khatri_dot(feats.fut_act[:,start:end], diff_all)
#         g_f = [- 2*wpred*np.dot(a_k_d[:,n-1] , model['Wfut'].reshape((-1, self.K.st), order='F') ) - policy_grad]      
#         
#         for i in xrange(n-2,-1,-1):                      
#             g_sf = g_sf - 2*wpred*np.dot(a_k_d[:,i+1] , model['Wfut'].reshape((-1, self.K.st), order='F') - policy_grad )      
#             f = states[:,i]
#             o_fx = feats.obs[:,start+i]
#             a_fx = feats.act[:,start+i]
#             jac_a = feats.act_grad[:,start+i,:]
#             g_sf, _, _, gj_a = self.backprop(g_sf, f, o_fx, a_fx, model, reg, filter_traj[i], g_a=jac_a)
#             g_sf= g_sf[None,:]
#             g_a.append(gj_a)
#             g_f.append(g_sf)
#         return g_a, g_f
    
    def bp_traj(self, states, filter_traj, feats, data, model, seq=0, reg=1e-6, wpred=1.0,wnext=0.0, wgrad=10.0):
        ''' filter all list of lists with all of the filter core parameters in that order'''
        if DEBUG: print('bp traj')
        start = feats.locs[seq]
        end = feats.locs[seq+1]
        n = states.shape[1] #number of points
        d_fut_o = data.d.fo
        # Difference between predicted and actual observations
        diff_all = data.fut_obs[:,start:end] - np.dot( model['Wfut'], lg.khatri_dot(states,feats.fut_act[:,start:end]) )
        W2obs = np.dot( self.map['oo2obs'], model['Woo'].reshape((self.K.oo,-1), order='F'))
        diff_next = data.obs[:,start:end] - np.dot(W2obs, lg.khatri_dot(states, feats.act[:,start:end]))
        g_Wex = np.zeros((1, self.K.st*self.K.exfut_obs*self.K.exfut_act), dtype=float)
        g_Woo = np.zeros((1, self.K.st*self.K.oo*self.K.act ), dtype=float)
        g_Wfut = np.zeros((1,self.K.st*d_fut_o*self.K.fut_act ), dtype=float)
        g_sf = np.zeros((1,self.K.st), dtype=float)
        a_k_d = lg.khatri_dot(feats.fut_act[:,start:end], diff_all)
        a_k_n = lg.khatri_dot(feats.act[:,start:end], diff_next)
        
        wnext = wnext*np.linalg.norm(np.dot(a_k_d.mean(1) , model['Wfut'].reshape((-1, self.K.st), order='F')))/np.linalg.norm(np.dot(a_k_n.mean(1) , W2obs.reshape((-1, self.K.st), order='F')))
        if data.grads is not None:
            wpred = wpred*np.linalg.norm(data.grads.max(1).reshape(g_sf.shape))/np.linalg.norm(np.dot(a_k_d.mean(1) , model['Wfut'].reshape((-1, self.K.st), order='F')))
             
        for i in xrange(n-2,-1,-1):
            sf = states[:,i+1]                            
            g_Wfut = g_Wfut - 2*lg.khatri_dot(sf[:,None],a_k_d[:,i+1][:,None]).T
            grad = data.grads[:,start+i+1].reshape(g_sf.shape)
            g_sf = g_sf - 2*wpred*np.dot(a_k_d[:,i+1] , model['Wfut'].reshape((-1, self.K.st), order='F') )\
            - wgrad*grad - 2*wpred*wnext*np.dot(a_k_n[:,i+1] , W2obs.reshape((-1, self.K.st), order='F') )             
            f = states[:,i]
            o_fx = feats.obs[:,start+i]
            a_fx = feats.act[:,start+i]
            g_sf, gj_Wex, gj_Woo = self.backprop(g_sf, f, o_fx, a_fx, model, reg, filter_traj[i])
            g_sf= g_sf[None,:]
            g_Woo = g_Woo + gj_Woo
            g_Wex = g_Wex + gj_Wex  
        g_Woo = (g_Woo.reshape((self.K.oo*self.K.act, self.K.st), order='F') + reg * model['Woo']) / float(n-1)
        g_Wex = (g_Wex.reshape((self.K.exfut_obs*self.K.exfut_act, self.K.st), order='F')+reg * model['Wex'])/float(n-1)
        g_Wfut = (g_Wfut.reshape(( d_fut_o, self.K.st*self.K.fut_act), order='F') + reg * model['Wfut']) / float(n-1)
        
        g_norm = np.sqrt(np.linalg.norm(g_Woo, 'fro')**2 + \
                         np.linalg.norm(g_Wex, 'fro')**2 + \
                         np.linalg.norm(g_Wfut, 'fro')**2)
        #print('=======================================> g_norm=', g_norm)
        max_norm = 10.0
        if (g_norm > max_norm): #clip gradients norm
            g_Woo = g_Woo / float(g_norm) * max_norm
            g_Wex = g_Wex / float(g_norm) * max_norm
            g_Wfut = g_Wfut / float(g_norm) * max_norm
        #print('=======================================> g_norm=', g_norm)
        
        grads = {'Wex':g_Wex, 'Woo':g_Woo, 'Wfut': g_Wfut}
        assert not np.isnan(g_Wex).any(), embed()
        assert not np.isnan(g_Woo).any(), embed()
        assert not np.isnan(g_Wfut).any(), embed()
        return grads
    
        
    #test next observation prediction

    def iterative_test_1s(self, data=None, N=0, state0=None):
        if N > data.act.shape[1] or N==0:
            N = data.act.shape[1]
        if state0 is None:
            state0 = self._start.reshape(-1,1)
        predicted_observations = []; error = []; states = []
        state = np.copy(state0).reshape(-1,1)
        # Run the system testHorizon steps forward without an observation;
        
        for j in xrange(N):            
            o_1s = self.predict(state, data.act[:,j].reshape(-1,1)) #predict current observation
            err    = np.sum((o_1s.squeeze() - data.obs[:,j])**2)
            predicted_observations.append(o_1s )
            error.append(err)
            state  = self.filter(state, data.obs[:,j].reshape(-1,1), a=data.act[:,j].reshape(-1,1)).squeeze() # condition only on current observation and action
            states.append(state.reshape((-1,1),order='F'))
            if data.series_index[(j+1)%N]<>data.series_index[j]:
                state = np.copy(state0)
        predicted_observations = np.concatenate(predicted_observations,axis=1)
        error = np.asarray(error)
        states = np.concatenate(states, axis=1)
        return predicted_observations, error, states
    
    def iterative_test_tH(self, data=None, N=0, state0=None):
        if state0 is None:
            state = self._start.squeeze()[:]
        if N > data.fut_act.shape[1] or N==0:
            N = data.fut_act.shape[1]
        expectedState = np.copy(self.state)
        predicted_observations = []
        error = []
        states=[]
        state = np.copy(state0)
        # Run the system testHorizon steps forward without an observation;
        for j in xrange(N):
            fo = self.predict_future(expectedState, data.fut_act[:,j])
            err = (fo[:,0] - data.fut_obs[:,j])**2
            predicted_observations.append(fo.reshape((1,data.d.o,-1), order='F'))
            error.append(err.reshape(-1,1))
            state  = self.filter(state, data.obs[:,j], a=data.act[:,j]) # condition only on current observation and action
            states.append(state.reshape((-1,1),order='F'))
            if data.series_index[(j+1)%N]<>data.series_index[j]:
                state = np.copy(state0)
        predicted_observations = np.concatenate(predicted_observations,axis=0)
        error = np.concatenate(error, axis=1)
        states = np.concatenate(states, axis=1)
        return predicted_observations, error, states
        
    def iterative_predict(self, actions, verbose=False, observations=None, tH=1, state0=None):
        if state0 is None:
            state = self._start.squeeze()[:]
        expectedState = np.copy(state)
        predicted_observations=[]
        error = []
        # Run the system testHorizon steps forward without an observation;
        for j in xrange(actions.shape[1]-tH):
            pred_obs = self.predict(expectedState, actions[:,j:j+tH]) #predict current observation
            predicted_observations.append(pred_obs.reshape((1,-1,tH),order='F'))
            if observations<>None:
                error.append(np.sum((pred_obs - observations[:,j:j+tH])**2))
        predicted_observations = np.concatenate(predicted_observations,axis=0).T
        error = np.asarray(error)
        return predicted_observations, error
    
    #override

    def test(self, data, title='', verbose=False, plot=True, f=sys.stdout):
        # Run the system testHorizon steps forward without an observation;
        predicted_observations_1s, error_1s, states_1s = self.iterative_test_1s(data)
        predicted_observations_tH, error_tH, states_tH = self.iterative_test_tH(data)
        previous_err = np.sum(np.abs(data.obs[:,:-1] - data.obs[:,1:])**2,axis=0)
        if plot:
            plot_modes(predicted_observations_1s[0,:], \
                       predicted_observations_tH[:,0,0],\
                       data.obs[0,:],\
                       label1='1s', label2='%ds'%self.fut, label3='gold',\
                       ylabel='observations', filename=self.file+'predictions')
            plot_modes(error_1s, \
                       error_tH.mean(0), \
                       previous_err, \
                       label1='1s', label2='%ds'%1, label3='prev',\
                       ylabel='error', filename=self.file+'error')
            
            print ('Accuracy predictions next prediction: mean=%f std=%f'%(error_1s.mean(), error_1s.std()), file=f )
            print ('Accuracy predictions over %d horizon: mean=%f std=%f'%(self.testHorizon, error_tH.mean(), error_tH.std()), file=f )

        plot_PCs(states_tH,np.zeros((states_tH.shape[1])), title='states', filename=self.file+'2PCs_states')
        return predicted_observations_tH, error_tH, states_tH
     
    
        
def test_RK4(argv):
    inputfile = 'examples/psr/data/noskip/' 
    print(argv)
    if len(argv)>=1:
        inputfile = argv[0] 
    #read data
    observations = read_matrix(inputfile, name='Y.txt', delim=' ')
    actions = read_matrix(inputfile, name='I.txt', delim=' ')
    
    #cObs, mObs = center_data(observations)
    #cActions, mActions = center_data(actions)
    #print('Train set :', actions.shape, observations.shape, mObs, mActions)
    tObservations = read_matrix(inputfile, name='tY.txt', delim=' ')
    tActions = read_matrix(inputfile, name='tI.txt', delim=' ')
    tActions = np.concatenate([tActions,tActions[:,-1:None]], axis=1)
    #ctObs = tObservations - np.tile(mObs,(tObservations.shape[1],1)).T
    #ctActions = tActions - np.tile(mActions, (tActions.shape[1],1)).T
    print('Test set : ', tActions.shape, tObservations.shape, tObservations.mean(1), tActions.mean(1))

    rff_dim = 5000 #D
    rdim = 20
    fut = 10
    past = 20
    reg = 1e-9
    psr = covariancePSR(dim=rdim, use_actions=False, reg=reg)
    feat_set = create_RFFPCA_featureset(rff_dim, rdim)
    
    train_fext = PSR_features(feat_set, fut, past, rdim)
    train_feats, train_data = train_fext.compute_features([observations],[actions])
    psr.train(train_fext, train_feats, train_data)
    train_fext.freeze()
    
    test_fext = PSR_features(feat_set, fut, past, rdim)
    test_feats, test_data = test_fext.compute_features([tObservations], [tActions], base_fext=train_fext)
    
    psr.test( test_data)

    return



def test_cyclic_psr(inputfile):
    inputfile='examples/psr/data/rff/'
    X = read_matrix(inputfile, name='X.txt', delim=',')
    d = 5
    trainsize = 200+1+2*d
    testsize = np.min([X.shape[1]-trainsize,50])
    train_obs = X[:,d:d+trainsize]
    train_actions = X[:,d-1:d+trainsize-1]*2.0
    test_obs = X[:,d+trainsize:d+trainsize+testsize]
    test_actions = X[:,d+trainsize-1:d+trainsize+testsize-1]
    
    rff_dim = 5000 #D
    rdim = 20
    fut = 10
    past = 20
    reg = 1e-9
    psr = covariancePSR(dim=rdim, use_actions=False, reg=reg)
    feat_set = create_RFFPCA_featureset(rff_dim, rdim)
    
    train_fext = PSR_features(feat_set, fut, past, rdim)
    train_feats, train_data = train_fext.compute_features([train_obs],[train_actions])
    psr.train(train_fext, train_feats, train_data)
    train_fext.freeze()
    
    test_fext = PSR_features(feat_set, fut, past, rdim)
    test_feats, test_data = test_fext.compute_features([test_obs], [test_actions], base_fext=train_fext)
    
    psr.test( test_data)
    embed()
    return
    
def test_demo_rffpsr(inputfile):
    inputfile='examples/psr/data/controlled/'
    Xtr = read_matrix(inputfile, name='Xtr', delim=',')
    Utr = read_matrix(inputfile, name='Utr', delim=',')
    Xtest = read_matrix(inputfile, name='Xtest', delim=',')
    Utest = read_matrix(inputfile, name='Utest', delim=',')
    Xval = read_matrix(inputfile, name='X_val', delim=',')
    Uval = read_matrix(inputfile, name='U_val', delim=',')
    d = 1
    N = 10 #number os seqs
    L = 100 #length of sequence
    Xtr = matrix2lists( Xtr, d, N=N, L=L)
    Utr = matrix2lists( Utr, d, N=N, L=L)
    Xtest = matrix2lists( Xtest, d, N=N, L=L)
    Utest = matrix2lists( Utest, d, N=N, L=L)
    Xval = matrix2lists( Xval, d, N=5,L=L)
    Uval = matrix2lists( Uval, d, N=5,L=L)
    

    fut         = 10
    past        = 20
    rstep       = 0.01
    Lmax        = np.max([Xtr[j].shape[1] for j in xrange(len(Xtr))])
    min_rstep   = 1e-5
    refine      = 5
    val_batch   = 5
    rff_dim     = 5000 #D
    rdim        = 20
    reg         = 1e-6
   
    feat_set = create_RFFPCA_featureset(rff_dim, rdim)
    
    train_fext = PSR_features(feat_set, fut, past, rdim)
    val_fext = PSR_features(feat_set, fut, past, rdim)
    test_fext = PSR_features(feat_set, fut, past, rdim)
    
    
    
    psr = covariancePSR(dim=rdim, use_actions=False, reg=reg)
    rpsr = num.RefineModelGD(rstep=rstep, optimizer='sgd', val_batch=val_batch,\
                              refine_adam=False, min_rstep=min_rstep)
    
    train_feats, train_data = train_fext.compute_features(Xtr,Utr)
    psr.train(train_fext, train_feats, train_data)
    train_fext.freeze()
    
    val_feats, val_data = val_fext.compute_features(Xval,Uval, base_fext=train_fext)
    test_feats, test_data = test_fext.compute_features(Xtest, Utest, base_fext=train_fext)
        
    embed()
    rpsr.model_refine(psr, train_feats, train_data, n_iter=refine, val_feats=val_feats, val_data=val_data, reg=reg)
        
    psr.test( test_feats, test_data)
    return



if __name__ == '__main__':
    #test_demo_rffpsr(sys.argv[1:])
    test_cyclic_psr
    
