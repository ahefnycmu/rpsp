# -*- coding: utf-8 -*-
'''
Created on Tue Feb 7 12:16:30 2017

@author: ahefny, zmarinho
'''


import numpy as np
import numpy.linalg as linalg
from time import time
import psr_models.utils.feats as feat
import psr_models.utils.regression as regression
from sklearn.linear_model import LogisticRegression
import scipy, scipy.io
from psr_models.features.feat_extractor import *
from psr_models.tests.test_rffpsr_planning import wave,ode,circulant
from distutils.dir_util import mkpath
from psr_models.features.psr_features import PSR_features
from psr_models.covariance_psr import covariancePSR
import psr_models.utils.numeric as num

def load_data(file_name, N_traj, p_blind):
    f = scipy.io.loadmat(file_name)
    trajs = f['trajs']
    p = np.array([p_blind/9,(1-p_blind)+(p_blind/9)])
    p /= np.mean(p)
    
    X_obs = [trajs[i,:,[0,1,8,9,10]] for i in xrange(N_traj)]
    X_act = [trajs[i,:,[4,5]] for i in xrange(N_traj)]
    X_wts= [p[np.prod(trajs[i,:,4:6] == trajs[i,:,-2:], 1)].reshape((1,-1)) for i in xrange(N_traj)]
    return X_obs,X_act,X_wts

def factored_lr_policy(data, feats, fut):
    l = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    
    d, n = data.act.shape
    f = fut
    prob = np.empty((d*(f+1),n))
    
    x = feats.past.T
    
    for i in xrange(f+1):
        for k in xrange(d):            
            y = data.exfut_act[i*d+k,:].T.astype(np.int)
            l.fit(x, y)
            assert np.all(l.classes_ == np.array([-2,0,2]))
            pp = l.predict_proba(x)
            prob[i*d+k,:] = pp[xrange(n),list(y//2+1)]
        
    p_fut = np.prod(prob[:d*f,:],0)        
    p_exfut = np.prod(prob,0)        
    
    return p_fut, p_exfut
 
def lr_policy(data, feats, fut):
    l = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    
    d, n = data.act.shape
    f = fut
    
    prob = np.empty((f+1,n))
    x = feats.past.T
    
    for i in xrange(f+1):
        y1 = data.exfut_act[i*d,:].astype(np.int)
        y2 = data.exfut_act[i*d+1,:].astype(np.int)
        y = (y1//2+1)*3 + (y2//2+1)
             
        l.fit(x,y)
        assert np.all(l.classes_ == np.array([xrange(9)]))
        pp = l.predict_proba(x)
        prob[i,:] = pp[xrange(n),list(y)]
            
    p_fut = np.prod(prob[:f,:],0)        
    p_exfut = np.prod(prob,0)        
    return p_fut, p_exfut

def uniform_blind_policy(data, feats):    
    n = data.obs.shape[1]
    return np.ones((1,n)), np.ones((1,n))

def gaussian_blind_policy(data, feats, l2_lambda):
    # Find the closest blind policy of the form
    # (a_{t+i} | history) ~ N(dot(w_i, h_t), \sigma_i^2)
    W_policy = regression.ridge_regression(feats.past, data.exfut_act, reg=l2_lambda).W
    r = np.dot(W_policy, feats.past) - data.exfut_act
    r2 = r*r
    S = np.mean(r*r,1).reshape((-1,1))
    
    blind_prob = np.sqrt(0.5/(np.pi * S)) * np.exp(-0.5*r2/S)
    
    d_a = data.act.shape[0]
    fut = data.fut_act.shape[0] / d_a
                
    blind_prop_future = np.prod(blind_prob[:d_a*fut,:], 0)
    blind_prop_extended = np.prod(blind_prob, 0)
            
    return blind_prop_future, blind_prop_extended

class importanceSampler(object):
    def __init__(self, fut, past, blind=lambda data, feats: gaussian_blind_policy(data, feats, 1e-3), traj_act_probs=None):
        self._fut = fut
        self._past = past
        self._traj_act_probs = traj_act_probs
        self._blind_policy = blind
    
    #create a traj probs act extractor
    
    def _compute_importance_weights(self, data, feats, traj_act_probs=None):
        if traj_act_probs is None and self._traj_act_probs is None:
            return None, None
        elif traj_act_probs is not None:
            self._traj_act_probs = traj_act_probs
        else:
            bounds = (self._past, self._fut)
            fut_extractor = feat.finite_future_feat_extractor(self._fut)
            extended_extractor = feat.finite_future_feat_extractor(self._fut+1)            
    
            # Compute the probability of action sequences given the non-blind policy            
            prob_future = np.prod(feat.flatten_features(self._traj_act_probs, fut_extractor, bounds)[0], 0)             
            prob_extended = np.prod(feat.flatten_features(self._traj_act_probs, extended_extractor, bounds)[0], 0)             
            
            blind_prop_future, blind_prop_extended = self._blind_policy(data, feats)
                                                            
            weights_future = blind_prop_future / prob_future
            weights_extended = blind_prop_extended / prob_extended
            
            return weights_future, weights_extended
     
def test_sampler(args, flname):
    h = 0.05
    valL = 50 #500
    testL = 50 #500
    min_rstep   = 1e-5
    val_batch   = 5

    
    method = eval(args.method)
    obs, act = method(args.numtrajs + valL + testL,args.len, h)
    feat_set = create_RFFPCA_featureset(args.rff, args.dim)
    trL = args.numtrajs-valL-testL
    Xtest = obs[trL:trL+testL]
    Utest = act[trL:trL+testL]
    Xval = obs[trL+testL:trL+testL+valL]
    Uval = act[trL+testL:trL+testL+valL]
    predictions_1 = []; predictions_2 = [];
    error_1 = []; error_2 = [];
    
    train_fext = PSR_features(feat_set, args.fut, args.past, args.dim)
    val_fext = PSR_features(feat_set, args.fut, args.past, args.dim)
    test_fext = PSR_features(feat_set, args.fut, args.past, args.dim)
    
    psr = covariancePSR(dim=args.dim, use_actions=False, reg=args.reg)
    psr_imp = covariancePSR(dim=args.dim, use_actions=False, reg=args.reg)
    rpsr = num.RefineModelGD(rstep=args.rstep, optimizer='sgd', val_batch=val_batch,\
                              refine_adam=False, min_rstep=min_rstep)
    
    
    Xtr = obs[:args.numtrajs]
    Utr = act[:args.numtrajs]
    IS = importanceSampler(args.fut, args.past, blind=lambda data, feats: gaussian_blind_policy(data, feats, args.reg))
    
    #test for PSR
    train_feats, train_data = train_fext.compute_features(Xtr,Utr)
    imp_w = IS._compute_importance_weights(train_data, train_feats, train_fext, traj_act_probs)
    
    psr.train(train_fext, train_feats, train_data)
    psr_imp.train(train_fext, train_feats, train_data, imp_w)
    
    train_fext.freeze()
    val_feats, val_data = val_fext.compute_features(Xval,Uval, base_fext=train_fext)
    test_feats, test_data = test_fext.compute_features(Xtest, Utest, base_fext=train_fext)
    
    embed()
    
     
        
    
