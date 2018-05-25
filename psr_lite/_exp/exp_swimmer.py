#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:27:55 2017

@author: ahefny
"""

from __future__ import print_function
from psr_lite.utils.p3 import *
import sys, traceback

try:
    import lds
except:
    print ('-'*60)
    print ('WARNING: Could not import lds. LDS model will be ignored.')
    print ('-'*60)
    traceback.print_exc(file=sys.stdout)
    print ('-'*60) 
    
import time
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import psr_lite.psr_base
import psr_lite.rffpsr as rffpsr
import psr_lite.rffpsr_rnn as rffpsr_rnn
import psr_lite.rnn_filter as rnn_filter
import psr_lite.feat_extractor as feat_extractor
import psr_lite.psr_eval as psr_eval

psr_lite.utils.misc.allow_default_rand()
seed = 0

N_trn = 25
N_tst = 10

def load_data(N_traj, train):
    p_blind = 1.0 if train else 0.8
    file_name='data/swimmer/swimmers_trajs_%0.1f.mat' % p_blind
    f = scipy.io.loadmat(file_name)
    trajs = f['trajs']
            
    if train:
        X_tr = [trajs[i,:,[0,1,8,9,10]].T for i in xrange(N_traj)]
        U_tr = [trajs[i,:,[4,5]].T for i in xrange(N_traj)]    
    else:
        X_tr = [trajs[i,:,[0,1,8,9,10]].T for i in xrange(-N_traj,0)]
        U_tr = [trajs[i,:,[4,5]].T for i in xrange(-N_traj,0)]    
        
    return X_tr,U_tr
 
####################################################
# Load Training Data
####################################################        
print ('Loading data ... ',end='')
X_tr,U_tr = load_data(N_trn, True)
print ('done')                        

####################################################
# Train Models
####################################################  
def create_feat_set():
    #nyst = lambda: psr_lite.feat_extractor.NystromFeatureExtractor(1000, max_dim=20, rng=rng)
    #feat_set = feat_extractor.create_uniform_featureset(nyst)     
    feat_set = feat_extractor.create_RFFPCA_featureset(5000,p_dim,orth=False)    
    #feat_set['past'] = feat_extractor.AppendConstant(feat_set['past'])
    return feat_set
      
models = {}
train_time = {}

fut = 10
past = 20
p_dim=20
l2_lambda = rffpsr.uniform_lambda(1e-3)

psr_settings = {'feature_set': create_feat_set(), 
                'projection_dim': p_dim,
                's1_method': 'joint',
                'l2_lambda': l2_lambda,
                'past_projection' : None}
               
# RNN
#s = time.time()
#models['rnn'] = rnn_filter.RNNFilter(5,fut,optimizer_iterations=5000,optimizer_step=1e-3,val_trajs=2)
#models['rnn'].train(X_tr, U_tr)
#train_time['rnn'] = time.time()-s

# RFFPSR
np.random.seed(seed)
psr_settings = {       
        'feature_set': create_feat_set(),
        'projection_dim' : p_dim,
        's1_method': 'joint',
        'l2_lambda': 1e-3,
        'past_projection' : None
        }
            
s = time.time()                  
models['rffpsr'] = rffpsr.RFFPSR(fut,past,**psr_settings)
models['rffpsr'].train(X_tr, U_tr)
train_time['rffpsr'] = time.time()-s

# RFFPSR With Refinment
s = time.time()                  
models['rffrnn'] = rffpsr_rnn.RFFPSR_RNN(models['rffpsr'], optimizer='sgd',
                                         optimizer_iterations=200, optimizer_step=1e-5,
                                         optimizer_min_step=1e-8, val_trajs=5, psr_iter=0,
                                         opt_U=False, opt_V=False)

models['rffrnn'].train(X_tr, U_tr)
train_time['rffrnn'] = time.time()-s + train_time['rffpsr']

#RFFPSR-Cond
np.random.seed(seed)
psr_settings = {       
        'feature_set': create_feat_set(),
        'projection_dim' : p_dim,
        's1_method': 'cond',
        'l2_lambda': 1e-3,
        'past_projection' : None
        }
    
                           
models['rffpsr_cond'] = rffpsr.RFFPSR(fut,past,**psr_settings)
models['rffpsr_cond'].train(X_tr, U_tr)

# RFFPSR-Cond With Refinment
np.random.seed(seed)
models['rffrnn_cond'] = rffpsr_rnn.RFFPSR_RNN(models['rffpsr_cond'], optimizer='sgd',
                                         optimizer_iterations=100, optimizer_step=0.1,
                                         optimizer_min_step=1e-8, val_trajs=2, psr_iter=0,
                                         opt_U=False, opt_V=False)

models['rffrnn_cond'].train(X_tr, U_tr)

# RFFARX 
np.random.seed(seed)
models['rffarx'] = psr_lite.psr_base.AutoRegressiveControlledModel(fut, past, create_feat_set()['past'], 1e-3)
models['rffarx'].train(X_tr, U_tr)

# Last 
models['last'] = psr_lite.psr_base.LastObsModel(X_tr[0].shape[1], fut)
models['last'].train(X_tr, U_tr)

####################################################
# Test Models
####################################################  
print(train_time)      
print ('Loading test data ... ',end='')
X_tst,U_tst = load_data(N_tst, False) 
print ('done')

mse = {}
burn_in = fut
err_fn = psr_eval.square_error

eval_models = ['rffpsr', 'rffrnn', 'rffpsr_cond', 'rffrnn_cond', 
               'rffarx', 'lds', 'last', 'hsepsr', 'rnn']

for m in eval_models:
    if m == 'lds':
        try:            
            _, mse[m] = lds.test_lds(fut, past, X_tr, U_tr, X_tst, U_tst, burn_in, err_fn)                        
        except:
            print ('-'*60)
            print ('WARNING: Could not evaluate LDS')
            print ('-'*60)
            traceback.print_exc(file=sys.stdout)
            print ('-'*60)
    else:            
        key = m
        if models.has_key(key):
            print (key)
            psr = models[key]
            
            err_func = lambda x,y: psr_eval.square_error(x[0],y[0])    
            mse[key] = psr_eval.eval_psr(psr, X_tst, U_tst, burn_in=10, err_func=err_fn)   
            print (mse[key])                
        else:
            print ('WARNING: Model %s not found' % key)
            
#%% ##################################
# Plot and Save Results
######################################    
plot_models = {'rffpsr':('rs--',{}), 'rffrnn':('rs-',{}), 
               'rffpsr_cond':('b^--',{}), 'rffrnn_cond':('b^-',{}),
               'lds':('m*--',{'markerfacecolor':'m','ms':8}), 'rffarx':('ko-',{}), 'last':('k:',{}),
               'hsepsr':('m*-',{'markerfacecolor':'m','ms':8}), 'rnn':('g',{})}

for m in mse.items():
    style = {'markerfacecolor':'white'}
    ms = plot_models[m[0]]
    style.update(ms[1])
    plt.plot(np.arange(1,fut+1), m[1], ms[0], **style)
    plt.hold(True)    
    
plt.hold(False)
plt.grid(True)
#plt.ylim([0, 0.3])
plt.show()

import cPickle as pickle
with open('mse_swimmer.pkl', 'wb') as out_file:
    out_file.write(pickle.dumps({'plot_models':plot_models, 'train_time':train_time, 'mse':mse}))

import IPython; IPython.embed()