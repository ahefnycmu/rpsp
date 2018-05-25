# -*- coding: utf-8 -*-
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
from utils import linalg as lg
from IPython import embed
from psr_models.utils.svdot import rbf_dot,rbf_sparse_dot
from psr_models.utils.utils import *
import time
import cPickle as pickle
from utils.plot_utils import plot_predictions
from psr_models.features.hankel_features import Hankel_features
from psr_models.features.rff_features import RFF_features
plt.ion()

DEBUG=False

class HSEPSR(object):
    def __init__(self, dim=20, hS=15, trS=-1, tS=-1, fE=100, tH=30, use_actions=True,\
                 dO=-1, dA=-1, file=None, reg=1e-6, feat_ext=None):
        self.hankelSize = hS
        self.dim = dim;
        self.trainSize = trS;
        self.testSize = tS;
        self.filterExtent = fE;
        self.testHorizon = tH;
        self.use_actions = use_actions;
        self.file = file
        self.S1_mat=None;
        self.G_HH=None;
        self.reg = reg
        self.Uproj = None
        self.state = None
        self.filter_dims = []
        self.dO=0
        self.dA=0
        self.feature_extractor = feat_ext
        if feat_ext<>None:
            self.set_feature_extractor(feat_ext) 

    def state_dim(self):
        return self.dim
         
    def build_gram(self, A, B=None, sigma=None,verbose=DEBUG):
        ''' build gram matr.ix NxN with median width from A A is dxN '''
        if verbose: print('Build Gram...') 
        if sigma == None:
            if type(A)==np.ndarray:
                dist = spp.distance.pdist(A.T)**2
                sigma = np.median(dist)
                if sigma==0:
                    sigma = np.percentile(dist,80)
                assert sigma>0, 'error dist sigma 0'
            elif ssp.isspmatrix(A):
                AL = A.tolil()
                l = np.max([len(dat) for dat in AL.data])
                AM = np.vstack([np.pad(dat, (0, l-len(dat)), 'constant', constant_values=(0.0)) for dat in AL.data])
                dist = spp.distance.pdist(AM.T)**2
                sigma = np.median(dist)
                A = A.toarray()
                if B<>None:
                    B=B.toarray()
        G = rbf_dot(1./(1e+0*sigma), A, B)
        if verbose: print(G.shape)
        return G
    
    def compute_smooth_matrix(self, verbose=DEBUG):
        if verbose: print('select features')
        G_HH_reg = self.G_HH
        di = np.diag_indices(self.G_HH.shape[1])
        G_HH_reg[di] = G_HH_reg[di] + self.reg 
        self.S1_mat = np.dot(np.linalg.pinv(G_HH_reg) , self.G_HH)
        self.S1_matp = np.dot(self.S1_mat, self.Uproj)
        return self.S1_mat
    
    def stage1(self, A, f=sys.stdout): 
        ''' Stage 1A regression if A are futures S1B if A is extended futures'''
        assert self.G_HH<>None, 'build gram first then regress'
        if self.S1_mat==None:
            self.compute_smooth_matrix()
        A_proj = np.dot(A, self.S1_mat)
        print('difference from regression ', np.linalg.norm(A-A_proj)\
              /float(np.linalg.norm(A)),file=f)
        return A_proj
        
    def filter(self, state, Gobs, Gaction=1.0):
        ''' filter state based on current action and last observation'''
        condition_k = (Gobs*Gaction)
        state = np.dot( np.einsum('ij,j->ij', self.part2, condition_k), np.dot(self.part1,state) )
        state  = state/np.dot(self.normalizer,state) 
        return state
        
    def predict(self, state, Gaction=1.0):
        ''' apply linear transformation to predict next observation when no observation is available but action is'''
        # Calculate expected future obs as a function of state
        up = np.einsum('i,ij->ij', Gaction, self.part1);
        up = np.dot(up , state)
        predFrame = np.dot(self.feature_extractor.observations,up**6) #mean average of training samples peaked on max
        predFrame = predFrame/np.sum(up**6)
        
        # Calculate next expected state
        expectedState = np.dot(self.part2, up)
        expectedState = expectedState/np.dot(self.normalizer,expectedState)
        return predFrame, expectedState, up
    
    def predictFromTrain(self, G, train_o, ext_fut, verbose=DEBUG):
        # Get the closest training observation in RKHS distance
        dist = np.diag(G) - 2 * np.dot(G , ext_fut[:,0])
        idx = np.argmin(dist)
        pred_o = train_o[:,idx] 
        return pred_o[:,None]
    
    def compute_gramians(self, feat_ext=False, verbose=DEBUG):
        if feat_ext==False:
            feat_ext = self.feature_extractor
        #self.trainSize = feat_ext.all_size
        self.mean_pred = feat_ext.obs_mean
        if verbose: print('compute_grams ...')
        self.G_HH   = self.build_gram(feat_ext.histories)
        self.G_FF   = self.build_gram(feat_ext.futures)
        self.G_FsF  = self.build_gram(feat_ext.futures, feat_ext.sfutures)
        self.G_EE   = self.build_gram(feat_ext.extfutures, feat_ext.extfutures)
        self.G_ss   = self.build_gram(feat_ext.sfutures, feat_ext.sfutures)
        self.G_OO   = self.build_gram(feat_ext.observations, feat_ext.observations)
        self.G_AA   = self.build_gram(feat_ext.actions, feat_ext.actions)
        self.G_Oo   = lambda o: self.build_gram(feat_ext.observations, o)
        self.G_Aa   = lambda a: self.build_gram(feat_ext.actions, a)
        return   
    
    def compute_lower_projection(self, verbose=DEBUG, dec='eigs'):
        if verbose: print('compute lower projection')
        if self.G_HH.shape[0]>7000:
            print('\n\nUSE EIGS WITH CAUTION size > 4000!')
            embed()
        if dec=='eigs':
            tic = time.time()
            if ssp.isspmatrix(self.G_HH):
                self.Omega, A = ssp.linalg.eigs( np.dot(self.G_HH,self.G_FF), k=self.dim, which='LM')
            elif type(self.G_HH)==np.ndarray:
                A, self.Omega    = lg.eigs(np.dot(self.G_HH,self.G_FF), self.dim);
            
            self.Omega  = np.abs(self.Omega)           # (positive semidefinite, so shouldn't be neg.)
            ALA         = np.abs(np.dot(A.T , np.dot(self.G_FF , A)))
            D           = np.diag(1.0/np.sqrt(np.diag(ALA)))
            self.Uproj  = np.real(np.dot(A, D)) #should be real
    
            print('eigen decomp took:', time.time()-tic)
        return
    
    def learn_train_parameters(self):
        #compute model parameters
        fut_proj = np.dot( self.G_FF , self.Uproj)
        del self.G_FF
        self.part1    = np.dot( self.G_HH, np.dot(fut_proj , np.diag(1./self.Omega)) )
        self.stationary  = np.dot( fut_proj.T, np.dot(self.S1_mat , \
            np.ones((self.G_HH.shape[1],1),dtype=float)/float(self.G_HH.shape[1]) ) );
        self.normalizer  = np.dot( np.ones((1,self.G_HH.shape[1]),dtype=float), self.part1 );
        del self.G_HH
        self.part2    = np.dot( self.Uproj.T, self.G_FsF)
        del self.G_FsF
        return
 
    def train_memsave(self, feat_ext=False, verbose=DEBUG, dec='eigs'):
        if feat_ext==False:
            feat_ext = self.feature_extractor
        #self.trainSize = feat_ext.all_size
        self.mean_pred = feat_ext.obs_mean
        if verbose: print('compute_grams ...')
        self.G_HH   = self.build_gram(feat_ext.histories)
        self.G_FF   = self.build_gram(feat_ext.futures)
        self.compute_lower_projection(verbose, dec=dec)
        self.compute_smooth_matrix()
        self.G_FsF  = self.build_gram(feat_ext.futures, feat_ext.sfutures)
        self.learn_train_parameters()
        #self.G_EE   = self.build_gram(feat_ext.extfutures, feat_ext.extfutures)
        #self.G_ss   = self.build_gram(feat_ext.sfutures, feat_ext.sfutures)
        self.G_OO   = self.build_gram(feat_ext.observations, feat_ext.observations)
        #self.G_AA   = self.build_gram(feat_ext.actions, feat_ext.actions)
        self.G_Oo   = lambda o: self.build_gram(feat_ext.observations, o)
        self.G_Aa   = lambda a: self.build_gram(feat_ext.actions, a)
        return 
    
    def train_all(self, feat_ext=False, verbose=DEBUG, dec ='eigs'):
        self.compute_gramians(feat_ext=feat_ext)
        self.compute_lower_projection(verbose, dec=dec)
        self.compute_smooth_matrix(verbose)
        self.learn_train_parameters()
        return
    
    def train(self, feat_ext, verbose=False, filename='', save=False):
        self.feature_extractor = feat_ext
        print ('save:',save, ' into ',filename+'.pkl')
        try:
            f = open(filename+'.pkl','rb')
            self.part1 = pickle.load(f)
            self.part2 = pickle.load(f)
            self.stationary = pickle.load(f)
            self.normalizer = pickle.load(f)
            self.feature_extractor.observations = pickle.load(f)
            self.feature_extractor.actions = pickle.load(f)
            self.G_OO = pickle.load(f) 
            self.mean_pred = pickle.load(f)
            f.close()  
            self.G_Oo   = lambda o: self.build_gram(self.feature_extractor.observations, o)
            self.G_Aa   = lambda a: self.build_gram(self.feature_extractor.actions, a)
            print('Load training data successfully! odim:%d, adim%d size%d'%\
                  (self.feature_extractor.observations.shape[0], self.feature_extractor.actions.shape[0],\
                   self.G_OO.shape[0]))
        except IOError:
            self.train_all(feat_ext=self.feature_extractor)
            if save:
                f = open(filename+'.pkl','wb')
                pickle.dump(self.part1, f)
                pickle.dump(self.part2, f)
                pickle.dump(self.stationary, f)
                pickle.dump(self.normalizer, f)
                pickle.dump(self.feature_extractor.observations, f)
                pickle.dump(self.feature_extractor.actions, f)
                pickle.dump(self.G_OO, f)
                pickle.dump(self.mean_pred, f)
                f.close()
        transition  = np.dot( self.part2, self.part1)
        if verbose: print('transition:\n',self.transition)
        if verbose: print('normalizer:\n', self.normalizer)
        if verbose: print('stationary:\n', self.stationary)
        self.start = self.stationary

        return
        
        

    def split_predictions(self, predictions, test_trajs, start=False):
        lengths = [traj.shape[0] for traj in test_trajs]
        offset =  self.testHorizon + 1 #self.filterExtent +
        lengths[0] -= offset
        lengths = [0] + lengths
        pred_trajs = [np.array(predictions[lengths[i]:lengths[i+1],:,:], dtype=float) for i in xrange(len(lengths)-1)]
        if start:
            aux = np.zeros((pred_trajs[0].shape[0]+offset,pred_trajs[0].shape[1],pred_trajs[0].shape[2]), dtype=float)
            aux[:offset,:,:] = pred_trajs[0][0,:,:]
            aux[offset:,:,:] = pred_trajs[0][:,:,:]
        else:
            aux = np.zeros((pred_trajs[-1].shape[0]+offset,pred_trajs[-1].shape[1],pred_trajs[-1].shape[2]), dtype=float)
            aux[-offset:,:,:] = pred_trajs[-1][-1,:,:]
            aux[:-offset,:,:] = pred_trajs[-1][:,:,:]
        pred_trajs[0] = aux
        return pred_trajs, offset

    def initial_state_filter(self, tObservations, tActions, fE=False, verbose=DEBUG):
        if fE:
            self.filterExtent = fE
        G_Onew = self.G_Oo(tObservations)
        if self.use_actions: 
            G_Anew = self.G_Aa(tActions)
        if verbose: print ('Filtering state for ',self.filterExtent)
        self.state   = self.start
        states=[]
        for i in xrange(self.filterExtent): 
            # HSE-HMM state update (filter based on current action and observation)
            self.state = self.filter(self.state, G_Onew[:,i], G_Anew[:,i])
            states.append(np.real(self.state[:,0]))
            if (np.isnan(self.state)).any():
                print('initial state filter')
                print (states)
                embed() 
        return states

    def iterative_predict(self, actions, tH=False, verbose=False, observations=False):
        if tH==False:
            tH = self.testHorizon
            if actions.shape[1]<tH:
                tH=actions.shape[1]
        if verbose: print('Test PSR ...')
        if self.use_actions: 
            G_Anew = self.G_Aa(actions)
        expectedState = np.copy(self.state)
        predicted_observations=[]
        error = []
        # Run the system testHorizon steps forward without an observation;
        for j in xrange(tH):
            predFrame, expectedState, sf = self.predict(expectedState, G_Anew[:,j]) #predict current observation
            predicted_observations.append(predFrame[None,:,0])
            if observations<>False:
                error.append(np.sum((predFrame[:,0] - observations[:,j])**2)) 
        predicted_observations = np.concatenate(predicted_observations,axis=0).T
        error = np.asarray(error)
        return predicted_observations, error
       
    def iterative_test(self, observations, actions, title='', verbose=False, plot=True, \
                       f=sys.stdout, tH=False, fE=False, G_Anew=1.0, states=[]):
        ''' receive an observation dOxtH and action dAxtH'''
        #feat_ext = Feature_extractor(use_actions=self.use_actions, dim=self.hankelSize)
        #feat_ext.compute_test_features( observations, actions)
        if tH<>False:
            self.testHorizon = tH
        if verbose: print('Test PSR ...')
        G_Onew = self.G_Oo(observations)
        if self.use_actions: 
            G_Anew = self.G_Aa(actions)
        expectedState = np.copy(self.state)
        predicted_observations = []
        error = []
        # Run the system testHorizon steps forward without an observation;
        for j in xrange(self.testHorizon):
            predFrame, expectedState, sf = self.predict(expectedState, G_Anew[:,j]) #predict current observation
            err         = np.sum((predFrame[:,0] - observations[:,j])**2)
            #predFrame_train = self.predictFromTrain(self.G_OO, self.observations, sf)    #pick closest training example
            #err_train   = np.sum((predFrame_train[:,0] - observations[:,j])**2)
            #mean_err    = np.sum((self.mean_pred - observations[:,j])**2)
            #predicted_observations_train.append(predFrame_train[:,0])
            predicted_observations.append(predFrame[None,:,0])
            error.append(err)
        self.state  = self.filter(self.state, G_Onew[:,0], G_Anew[:,0]) # condition only on current observation and action
        states.append(self.state)
        predicted_observations = np.concatenate(predicted_observations,axis=0).T
        error = np.asarray(error)
        
        return predicted_observations, error, states
        
    def test(self, feat_ext, title='', verbose=False, plot=True,\
              f=sys.stdout, tH=False, fE=False):

        if verbose: print (' tH before ', self.testHorizon) 
        if tH<>False:
            self.testHorizon = tH
        if fE<>False:
            self.filterExtent = fE
        if verbose: print ('tH after ',self.testHorizon)
        if verbose: print('Test PSR ...')
        #compute test kernels
        G_Onew = self.G_Oo(feat_ext.observations)
        if feat_ext.use_actions: 
            G_Anew = self.G_Aa(feat_ext.actions)
        # Filtering and Predicting 
        # Prediction Errors
        self.testSize   = feat_ext.observations.shape[1] - self.testHorizon - self.filterExtent

        if verbose:
            print('Filtering and Predicting');
            print('Test size is ', self.testSize, file=f)
        err             = np.zeros((self.testSize-1,self.testHorizon),dtype=float)
        err_train       = np.zeros((self.testSize-1,self.testHorizon),dtype=float)
        mean_err        = np.zeros((self.testSize-1,self.testHorizon),dtype=float)
        predicted_observations          = np.zeros((self.testSize-1,feat_ext.dO,self.testHorizon),dtype=float)
        predicted_observations_train    = np.zeros((self.testSize-1,feat_ext.dO,self.testHorizon),dtype=float)
        if verbose: print(err.shape)
        # State
        state   = self.start
        states  = np.zeros((state.shape[0], self.testSize + self.filterExtent - 1), dtype=float ) 
        for i in xrange(self.testSize + self.filterExtent-1): 
            # HSE-HMM state update (filter based on current action and observation)
            
            state  = self.filter(state, G_Onew[:,i], G_Anew[:,i])
            states[:,i] = np.real(state[:,0])
            if (i >= (self.filterExtent-1)):
                expectedState = state
                ii = i - self.filterExtent 
                # Run the system testHorizon steps forward without an observation;
                for j in xrange(self.testHorizon):
                    predFrame, expectedState, sf = self.predict(expectedState, G_Anew[:,i+j+1])
                   
                    predFrame_train = self.predictFromTrain(self.G_OO, self.feature_extractor.observations, sf)    #pick closest training example
                    
                    err[ii,j]       = np.sum((predFrame[:,0] - feat_ext.observations[:,i+j+1])**2)
                    err_train[ii,j] = np.sum((predFrame_train[:,0] - feat_ext.observations[:,i+j+1])**2)
                    mean_err[ii,j]  = np.sum((self.mean_pred - feat_ext.observations[:,i+j+1])**2)
                 
                    predicted_observations[ii,:,j]       = predFrame[:,0]
                    predicted_observations_train[ii,:,j] = predFrame_train[:,0]
            if (i%50 == 0) and verbose:
                print(['Iteration: ', i])
                print([state.T])
        if plot:
            filename = '%s/%spredictions_2mode%ddim_%dhS_%dtr'%(self.file,title,self.dim,self.hankelSize,self.feature_extractor.observations.shape[1])
            pred3 = feat_ext.observations[:,1:self.testSize]
            plot_predictions( predicted_observations, predicted_observations_train, pred3,\
                                    err, err_train, mean_err, title=title, filename=filename,\
                                    label1='hsepsr', label2='train closest', label3='gold')
        
            print('Save figure in ',filename)
        
            print ('Accuracy predictions next prediction: mean=%f std=%f'%(err[:,0].mean(), err[:,0].std()), file=f )
            print ('Accuracy predictions over %d horizon: mean=%f std=%f'%(self.testHorizon, err.mean(), err.std()), file=f )
        
        return predicted_observations, err
        
def test_2subs(argv):
    inputfile = 'examples/psr/data/noskip/' 
    print(argv)
    if len(argv)>=1:
        inputfile = argv[0] 
    #read data
    observations = read_matrix(inputfile, name='Y.txt', delim=' ')
    actions = read_matrix(inputfile, name='I.txt', delim=' ')
    cObs, mObs = center_data(observations)
    cActions, mActions = center_data(actions)
    print('Train set :', actions.shape, observations.shape, mObs, mActions)

    tObservations = read_matrix(inputfile, name='tY.txt', delim=' ')
    tActions = read_matrix(inputfile, name='tI.txt', delim=' ')
    ctObs = (tObservations.T - mObs).T
    ctActions = (tActions.T - mActions).T
    print('Test set : ', tActions.shape, tObservations.shape, tObservations.mean(1), tActions.mean(1))
    
    #train
    psr = HSEPSR(hS=15, dim=50, fE=1, tH=1, use_actions=True, file=inputfile)
    psr.train(Hankel_features, cObs, cActions)
    #test
    predictions, err = psr.test(Hankel_features, cObs, cActions)
    return

def test_RK4(argv):
    inputfile = 'examples/psr/data/RK4/' 
    print(argv)
    if len(argv)>=1:
        inputfile = argv[0] 
    #read data
    trainSize=1010
    observations = read_matrix(inputfile, name='Ycent.txt', delim=' ')
    actions = read_matrix(inputfile, name='Icent.txt', delim=' ')
    actions = actions.T
    train_obs = observations[:,:trainSize]
    train_actions = actions[:,:trainSize]
    
    tobservations = read_matrix(inputfile, name='tYcent.txt', delim=' ')
    tactions = read_matrix(inputfile, name='tIcent.txt', delim=' ')
    tactions = tactions.T
    tactions = np.concatenate([tactions, tactions[:,-1:], tactions[:,-1:]], axis=1)
    print (train_obs.shape,train_actions.shape,tobservations.shape, tactions.shape)
    #tobservations  = observations[:,trainSize:]
    #tactions = actions[:, trainSize:]

    #cObs, mObs = center_data(observations)
    #cActions, mActions = center_data(actions)
    #print('Train set :', actions.shape, observations.shape, mObs, mActions)

    #ctObs = (tobservations.T - mObs).T
    #ctActions = (tactions.T - mActions).T
    #print('Test set : ', tactions.shape, tobservations.shape, tobservations.mean(1), tactions.mean(1))
    
    #train
    psr = HSEPSR(hS=10, dim=5, fE=99, tH=300, use_actions=True, file=inputfile)
    psr.train(Hankel_features, train_obs, train_actions)
    #test
    predictions, err = psr.test(Hankel_features, tobservations, tactions)
    return

if __name__ == '__main__':
    #test_RK4(sys.argv[1:])
    test_2subs(sys.argv[1:])
    
        