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
import psr_models.utils.feats as feat
#import psr_models.utils.kernel as kern
plt.ion()
np.random.seed(100)
DEBUG=False
class structtype():
    pass

class RFF_features(object):
    def __init__(self, use_actions=True, fut=5, k=10000, sigma=1.0, r_dim=5, filedir='examples/psr/data/pnpush_rff/', past=20):
        #super(RFF_features, self).__init__(use_actions=use_actions, dim=fut, filedir=filedir, past=past)
        self.filedir = filedir
        self.rff_dim = k
        self.sigma = sigma
        self.reduced_dim = r_dim
        self._fut = fut 
        self._past = past
        self.use_actions = True
        self.obs_rff = None
        self.act_rff = None
        self.Uo = None
        self.Uo_fx = []
        self.Uoo = None
        self.Uoo_fx = []
        self.Ua = None
        self.Ua_fx = []
        self.Upast = None
        self.Upast_fx = []
        self.Ufut_a = None
        self.Ufut_a_fx = []
        self.Ufut_o = None
        self.Ufut_o_fx = []
        self.Uextfut_a = None
        self.Uextfut_a_fx = []
        self.Uextfut_o = None
        self.Uextfut_o_fx = []
        self.extfutures = None
        self.Ust = None
        self.Ust_fx = []
        self.S_st = 0
        self.S_o = 0
        self.S_oo = 0
        self.S_fut_o = 0
        self.S_extfut_o = 0
        self.S_a = 0
        self.S_fut_a = 0
        self.S_extfut_a = 0
        self.proj_o = None
        self.proj_fut_o = None
        self.proj_ext_fut_o = None
        self.proj_a = None
        self.proj_fut_a = None
        self.proj_extfut_a = None
        self.traj_ids = []
 
    
#     #override
#     def build_all_features(self, x, z, verbose=DEBUG, add_const=False, L=0, N=0):
#         ''' Build histories, futures, observations from Hankel matrix '''
#         if verbose: print('select features')
#         #if L==0:
#         #    all_size = self.all_size
#         #    L = np.divide(all_size, self.num_seqs)
#         #    print('allsize ',all_size, x.shape, L,self.num_seqs)
#         if N==0:
#             N = self.num_seqs
#         assert self.num_seqs == (len(self.traj_locs)-1), 'wrong number of sequences'
#         pref = np.max([0, self.dim-self.past_dim])
#         zs = [z[:,self.traj_locs[i]:self.traj_locs[i+1]] for i in xrange(N)]
#         xs = [x[:,self.traj_locs[i]:self.traj_locs[i+1]] for i in xrange(N)]
#         
#         self.actions      = np.concatenate([zs[i][:self.dA, self.past_dim:zs[i].shape[1]-1] for i in xrange(N)], axis=1)            # Current action
#         self.observations = np.concatenate([xs[i][:self.dO, self.past_dim:xs[i].shape[1]-1] for i in xrange(N)], axis=1)                                                # Current observation
#         self.histories    = np.vstack((np.concatenate([xs[i][self.dO*pref:, :xs[i].shape[1]-self.past_dim-1] for i in xrange(N)], axis=1),\
#                                        np.concatenate([zs[i][self.dA*pref:, :zs[i].shape[1]-self.past_dim-1] for i in xrange(N)], axis=1) )) # Past sequences of observations
#         self.futures_o    = np.concatenate([xs[i][:self.dO*self.dim, self.past_dim:xs[i].shape[1]-1] for i in xrange(N)], axis=1)   # Immediate future sequences of observations 
#         self.futures_a    = np.concatenate([zs[i][:self.dA*self.dim, self.past_dim:zs[i].shape[1]-1] for i in xrange(N)], axis=1)   # Immediate future sequences of observations 
#         self.sfutures_o   = np.concatenate([xs[i][:self.dO*self.dim, self.past_dim+1:xs[i].shape[1]] for i in xrange(N)], axis=1)   # Future one-step-removed sequences of observations
#         self.sfutures_a   = np.concatenate([zs[i][:self.dA*self.dim, self.past_dim+1:zs[i].shape[1]] for i in xrange(N)], axis=1)   # Future one-step-removed sequences of observations
#         
#         self.locs = [ xs[i].shape[1]-1-self.past_dim for i in xrange(N) ]
#         assert np.asarray(self.locs).any()>=0, 'too small sequences'
#         self.locs.insert( 0, 0 )
#         self.locs = np.cumsum( self.locs )
#         
#         if verbose: print('histories', self.histories.shape)
#         if verbose: print('futures_o', self.futures_o.shape)
#         if verbose: print('sfutures_o', self.sfutures_o.shape)
#         if verbose: print('actions', self.actions.shape)
#         if verbose: print('observations', self.observations.shape)
#         return
    
    def build_state_features(self, model, states):
        self.U_st , _, self.U_st_fx = lg.rand_svd_f(states, f=(lambda X: X), n = states.shape[1],\
                                                    k=self.reduced_dim, it=1, slack=50, blk=1000)
        self.U_st_f = lambda X: np.dot(self.U_st, X)
        self.S_st = self.U_st_fx.shape[0]
        return self.U_st_fx
    
    def copy_projections(self, feat_ext):
        ''' shallow copy of projections'''
        #past
        feat_ext.Upast = self.Upast
        feat_ext.proj_past = self.proj_past
        feat_ext.S_past = self.S_past
        #current observation
        feat_ext.Uo = self.Uo
        feat_ext.proj_o = self.proj_o
        feat_ext.S_o = self.S_o
        #current covariance observation
        feat_ext.Uoo = self.Uoo 
        feat_ext.S_oo = self.S_oo
        #future_o
        feat_ext.Ufut_o = self.Ufut_o
        feat_ext.proj_fut_o = self.proj_fut_o
        feat_ext.S_fut_o = self.S_fut_o
        #extfuture_o
        feat_ext.Uextfut_o = self.Uextfut_o
        feat_ext.S_extfut_o = self.S_extfut_o
        if self.use_actions:
            #current observation
            feat_ext.Ua = self.Ua
            feat_ext.proj_a = self.proj_a
            feat_ext.S_a = self.S_a
            #future_o
            feat_ext.Ufut_a = self.Ufut_a
            feat_ext.proj_fut_a = self.proj_fut_a
            feat_ext.S_fut_a = self.S_fut_a
            #extfuture_o
            feat_ext.Uextfut_a = self.Uextfut_a
            feat_ext.S_extfut_a = self.S_extfut_a
        feat_ext.dO = self.dO
        feat_ext.dA = self.dA
        #states
        feat_ext.Ust = self.Ust
        feat_ext.U_st_f = self.U_st_f
        feat_ext.S_st = self.S_st
        return
    
    def apply_projections(self, feat_ext, const=False):
        '''project features according to projection matrices'''
        feat_ext.Upast_fx = feat_ext.proj_past(feat_ext.histories)
        feat_ext.Uo_fx = feat_ext.proj_o(feat_ext.observations)
        feat_ext.Uoo_fx = np.dot(feat_ext.Uoo.T, lg.khatri_dot(feat_ext.Uo_fx, feat_ext.Uo_fx) )
        feat_ext.Ufut_o_fx = feat_ext.proj_fut_o(feat_ext.futures_o)
        sfut_o_fx = self.proj_fut_o(feat_ext.sfutures_o)
        extfut_o = lg.khatri_dot(  sfut_o_fx, feat_ext.Uo_fx )
        feat_ext.Uextfut_o_fx = np.dot( feat_ext.Uextfut_o.T, extfut_o )
        if self.use_actions:
            feat_ext.Ua_fx = feat_ext.proj_a(feat_ext.actions)
            feat_ext.Ufut_a_fx = feat_ext.proj_fut_a(feat_ext.futures_a)
            sfut_a_fx = self.proj_fut_a(feat_ext.sfutures_a)
            ext_fut_a = lg.khatri_dot(feat_ext.Ua_fx, sfut_a_fx)
            feat_ext.Uextfut_a_fx = np.dot( feat_ext.Uextfut_a.T, ext_fut_a )
            if const:
                print('const')
                #The KR product of action and shifted test actions already
                # contains a constant feature. Make sure it stays there.
                self.Uextfut_a = lg.orth(np.vstack([  self.Uextfut_a, np.vstack([np.zeros((self.Uextfut_a.shape[0]-1, 1)) , [[1]]]).T    ]))
                self.Uextfut_a_fx = np.dot(self.Uextfut_a, ext_fut_a)
                self.S_extfut_a = self.Uextfut_a_fx.shape[0]
                self.Uextfut_o = lg.orth(np.hstack([  self.Uextfut_o, np.vstack([np.zeros((self.Uextfut_o.shape[0]-1, 1)) , [[1]]])    ]))
                self.Uextfut_o_fx = np.dot(extfut_o.T, self.Uextfut_o).T
                self.S_extfut_o = self.Uextfut_o_fx.shape[0]
        return
    
    def compute_projected_features(self, observations, actions, all_size=0, verbose=DEBUG, const=True):
        ''' use training feature_extractor projections to build validation features'''
        feats = RFF_features( use_actions=self.use_actions, fut=self._fut, k=self.rff_dim,\
                                  sigma=self.sigma, r_dim=self.reduced_dim, filedir=self.filedir+'val_', past=self._past)
        #x,z = feats.extract_features(observations, actions, all_size=all_size)
        #feats.build_all_features(x, z)
        observations, actions = self.validate_trajectories(observations, actions)
        feats._extract_timewins(observations, actions) 
        self.copy_projections(feats)
        self.apply_projections(feats, const=const)
        return feats
          
    def proj_add_const(self, U, X):
        if X.ndim==1:
            Y = np.hstack([np.dot(U.T,X), 1])
        else:
            Y = np.vstack([np.dot(U.T,X), np.ones((1,X.shape[1]))])
        return Y
    
    def add_const(self, X):
        Y = np.vstack([X,np.ones((1,X.shape[1]))])
        return Y
    
#     def compute_train_hankel_features(self, observations, actions, all_size=0, verbose=DEBUG,\
#                                 save=False, const=False, gradients=True):  
#         try:
#             if DEBUG: print('Load train features...in ', self.filedir)
#             self.load_features('rff_train_features')
#         except IOError:
#             if DEBUG: print('Compute train features', all_size)
#             x,z = self.extract_features(observations, actions, all_size=all_size)
#             self.build_all_features(x,z) #observations, actions) #x,z) 
#             if save: 
#                 self.save_features('rff_train_features')
#        
#         self.compute_rff_features(save,const=const)
#         if gradients:
#             self.compute_rff_grad_features()
#         return 
     
    def validate_trajectories(self, obs, act):
        observations = [obs[i] for i in xrange(len(obs)) if obs[i].shape[1] > (self._fut+self._past+2) ] #+1 for skipped future and one more for gradients
        actions = [act[i] for i in xrange(len(act)) if act[i].shape[1] > (self._fut+self._past+2) ]
        return observations, actions
     
    def compute_features(self, observations, actions, gradients = True, const=False):
        observations, actions = self.validate_trajectories(observations, actions)
        self._extract_timewins(observations, actions)
        self.compute_rff_features(False, const=const)
        if gradients:
            self.compute_rff_grad_features()
        return
    
    def _extract_timewins(self, traj_obs, traj_act):
       
        bounds = (self._past, self._fut)                
        past_extractor = feat.finite_past_feat_extractor(self._past)
        fut_extractor = feat.finite_future_feat_extractor(self._fut)
        shifted_fut_extractor = feat.finite_future_feat_extractor(self._fut, 1)
        extended_fut_extractor = feat.finite_future_feat_extractor(self._fut+1)
        immediate_extractor = lambda X,t: X[:,t]
                        
        self.past_obs, self.series_index, self.time_index = \
            feat.flatten_features(traj_obs, past_extractor, bounds)
        self.past_act,_,_ = feat.flatten_features(traj_act, past_extractor, bounds)
        self.histories = np.vstack((self.past_obs, self.past_act))
        self.futures_o,_,_ = feat.flatten_features(traj_obs, fut_extractor, bounds)
        self.futures_a,_,_ = feat.flatten_features(traj_act, fut_extractor, bounds)
        self.sfutures_o,_,_ = feat.flatten_features(traj_obs, shifted_fut_extractor, bounds)
        self.sfutures_a,_,_ = feat.flatten_features(traj_act, shifted_fut_extractor, bounds)
        self.extfutures_o,_,_ = feat.flatten_features(traj_act, extended_fut_extractor, bounds)
        self.observations,_,_ = feat.flatten_features(traj_obs, immediate_extractor, bounds)
        self.actions,_,_ = feat.flatten_features(traj_act, immediate_extractor, bounds)
                        
        self.dpast = self.histories.shape[0]
        self.dO = self.observations.shape[0]
        self.dA = self.actions.shape[0]
        self.dfO = self.futures_o.shape[0]
        self.dfA = self.futures_a.shape[0]
        
        self.num_seqs = len(traj_obs)
        self.locs = np.where(self.series_index[1:]-self.series_index[:-1] ==1)[0].tolist()
        self.locs.append(len(self.series_index))
        self.locs.insert(0,0)
        
        return 
    

    
    
    #TODO: update by adding to HankelMatrix
    #def update_train_features(self, obersvations, actions):
    
    def rff_feats(self, data, filename, rff=None, const=False, save=False, verbose=DEBUG):
        if verbose: print (filename, data.shape)
        buffer = np.max([self.reduced_dim*2, 50])
        batch = 10
        sigma = lg.median_bw(data, 5000)
        if sigma==0.0:
            sigma=1.0
        Z = np.random.normal(size=(self.rff_dim, data.shape[0]))/float(sigma)
        assert not np.isnan(Z).any(), embed()
        #Z=np.eye(self.rff_dim, data.shape[0])
        if rff is None:
            rff = (lambda X: self.apply_rff_mat(X,Z) )
        tic= time.time()
        try:
            if verbose:print('Load rff...')
            U , Ufx = self.load_projs(filename)
            if verbose: print('load took:',time.time()-tic)
        except IOError:
            if verbose: print('Extract RFF features...', data.shape)
            #U , Ufx = svd_f(data, self.reduced_dim, rff, buffer, batch)
            U, s, Ufx = rand_svd_f(data, f=rff, k=self.reduced_dim, it=2,slack=50, blk=1000)
            
            if verbose: print('svd_f took:',time.time()-tic) 
            if save:
                self.save_projs(U, Ufx, filename=filename)
        if const:
            Ufx = np.concatenate([Ufx, np.ones((1,Ufx.shape[1]))],axis=0)
            proj = lambda X: self.proj_add_const( U, self.apply_rff_mat(X, Z))
            grad = lambda X: self.proj_add_const( U, self.rff_grad_mat(X, Z))
        else:
            proj = lambda X: np.dot(U.T, self.apply_rff_mat(X, Z))
            grad = lambda X: np.dot(U.T, self.rff_grad_mat(X, Z))
        k = Ufx.shape[0]
        return U, Ufx, proj, k, grad
    
    def compute_rff_grad_features(self):
        self.o_grad = self.o_grad_f(self.observations)
        self.a_grad = self.a_grad_f(self.actions) #dA x da x N
        return
        
    def compute_rff_features(self, save=False, const=False, verbose=DEBUG):
        #past
        if verbose: print('past',)
        self.Upast , self.Upast_fx, self.proj_past, self.S_past, _ = self.rff_feats(self.histories, 'rff_proj_past', const=True)
        if verbose: print(self.S_past)
        #current observation
        if verbose: print('o',)
        self.Uo , self.Uo_fx, self.proj_o, self.S_o, self.o_grad_f = self.rff_feats(self.observations,'rff_proj_obs', const=const)
        #current covariance observation
        oo = lg.khatri_dot(self.Uo_fx, self.Uo_fx)
        self.Uoo , self.Uoo_fx, x, self.S_oo, _ = self.rff_feats(oo,'rff_proj_obs', rff=(lambda X:X) , const=const)
        if const:
            self.Uoo = np.concatenate([self.Uoo, np.ones((self.Uoo.shape[0],1))],axis=1)
            
        if verbose: print(self.S_o, self.S_oo)
        #future observation
        if verbose: print('fut_o',)
        self.Ufut_o , self.Ufut_o_fx, self.proj_fut_o, self.S_fut_o, _ = self.rff_feats(self.futures_o, 'rff_proj_fut_o', const=const)
        #skipped future observation
        sfut_o_fx = self.proj_fut_o(self.sfutures_o)
        if verbose: print(self.S_fut_o)
        #extended future observation
        if verbose: print('extfut_o',)
        extfut_o = lg.khatri_dot(  sfut_o_fx, self.Uo_fx )
        self.Uextfut_o , self.Uextfut_o_fx, x, self.S_extfut_o, _ = self.rff_feats(extfut_o, 'rff_proj_extfut_o', rff=(lambda X: X), const=const)
        if const:
            if verbose: print('const')
            #The KR product of action and shifted test actions already
            # contains a constant feature. Make sure it stays there.
            self.Uextfut_o = lg.orth(np.hstack([  self.Uextfut_o, np.vstack([np.zeros((self.Uextfut_o.shape[0]-1, 1)) , [[1]]])    ]))
            self.Uextfut_o_fx = np.dot(extfut_o.T, self.Uextfut_o).T
            self.S_extfut_o = self.Uextfut_o_fx.shape[0]
            #Ufx = np.concatenate([self.Uextfut_o, np.ones((self.Uextfut_o.shape[0],1))],axis=1)
            #self.Uextfut_o_fx = np.dot( extfut_o.T, Ufx).T
            
        self.dO = self.Uo.shape[1]    
        if verbose: print(self.S_extfut_o)
        
        if self.use_actions:
            #current action
            if verbose: print('a',)
            self.Ua , self.Ua_fx, self.proj_a, self.S_a, self.a_grad_f = self.rff_feats(self.actions,'rff_proj_act',rff=None, const=const)
            if verbose: print(self.S_a)
            #future action
            if verbose: print('fut_a',)
            self.Ufut_a , self.Ufut_a_fx, self.proj_fut_a, self.S_fut_a, _ = self.rff_feats(self.futures_a,'rff_proj_fut_a', rff=None, const=const)
            #skipped future action
            sfut_a_fx = self.proj_fut_a(self.sfutures_a)
            if verbose: print(self.S_fut_a)
            #extended future action
            if verbose: print('extfut_a',)
            extfut_a = lg.khatri_dot(self.Ua_fx, sfut_a_fx)
            self.Uextfut_a , self.Uextfut_a_fx, x, self.S_extfut_a, _ = self.rff_feats(extfut_a, 'rff_proj_extfut_a',  rff=(lambda X: X), const=const)
            if verbose: print(self.S_extfut_a)
            if const:
                if verbose: print('const')
                #The KR product of action and shifted test actions already
                # contains a constant feature. Make sure it stays there.
                self.Uextfut_a = lg.orth(np.vstack([  self.Uextfut_a, np.vstack([np.zeros((self.Uextfut_a.shape[0]-1, 1)) , [[1]]]).T    ]))
                self.Uextfut_a_fx = np.dot(self.Uextfut_a, extfut_a)
                self.S_extfut_a = self.Uextfut_a_fx.shape[0]
                #Ufx = np.concatenate([self.Uextfut_a, np.ones((1,self.Uextfut_a.shape[1]))],axis=0)
                #self.Uextfut_fx_a = np.dot(Ufx, extfut_a)
            self.dA = self.Ua.shape[1]
        return
    
#     def apply_rff(self, data, sigma):
#         ''' data is dxN d:number features in data rff_dim is Fourier feature dim'''
#         if sigma==0:
#            sigma=self.sigma
#         d = data.shape[0]
#         Z = np.random.randn(self.rff_dim, d)/float(sigma) 
#         b = 2*np.pi*np.random.randn(self.rff_dim)
#         Z = np.dot( Z, data).T + b
#         Y = np.sqrt(2)/float(np.sqrt(self.rff_dim)) * np.concatenate([ np.cos(Z.T), np.sin(Z.T) ],axis=0)
#         return Y
    
    def apply_rff_mat(self, data, Z):
        ''' data is dxN d:number features in data rff_dim is Fourier feature dim' Z are the fourrier frequencies'''
        Y = np.dot(Z, data)
        d = data.shape[0]
        Y = 1.0/float(np.sqrt(self.rff_dim)) * np.concatenate([ np.cos(Y), np.sin(Y) ], axis=0)
        assert not np.isnan(Y).any(), embed()
        return Y
    
    def rff_grad_mat(self, data, Z):
        Y = np.dot(Z, data)
        d = data.shape[0]
        G = np.concatenate([ -np.sin(Y), np.cos(Y) ], axis=0)
        #Y = 1.0/float(np.sqrt(self.rff_dim)) * np.array([(np.vstack([Z,Z]).T*G[:,i]).T for i in xrange(G.shape[1]) ])
        Y = 1.0/float(np.sqrt(self.rff_dim)) * lg.khatri_dot(np.vstack([Z,Z]).T,G.T).T
        assert not np.isnan(Y).any(), embed()
        return Y
        
    def tfx(self, T,f,X):
        Y = np.dot(T,f(X))
        return Y
    
    def save_projs(self, U, Ufx, filename='rff.pkl'):
        f = open(self.filedir+filename+'.pkl','wb')
        pickle.dump(U, f)
        pickle.dump(Ufx, f)
        f.close()
        return
    
    def load_projs(self, filename):
        f = open(self.filedir+filename+'.pkl','rb')
        U = pickle.load(f)
        Ufx = pickle.load(f)
        f.close()
        print('Load U=', U.shape, ' Ufx=', Ufx.shape)
        return U, Ufx   
    




if __name__ == '__main__':
    print('do cyclic rff test')
    inputfile='examples/psr/data/rff/'
    X = read_matrix(inputfile, name='X.txt', delim=',')
    fut = 5
    past=10
    trainsize = 200+1+2*fut
    train_obs = [X[:,fut:fut+trainsize]]
    train_actions = train_obs
    
    rff_feats = RFF_features( fut=fut, past=past, filedir='examples/psr/data/rff/tests/')
    rff_feats.compute_train_features(train_obs, train_actions)
    
    
    
    embed()
    
    rff_2feats = RFF_features( fut=fut, past=past, filedir='examples/psr/data/rff/tests/')
    rff_feats.update_train_features(train_obs[:,:50], train_actions[:,:50])
    rff_feats.update_train_features(train_obs[:,50:], train_actions[:,50:])
    
    
