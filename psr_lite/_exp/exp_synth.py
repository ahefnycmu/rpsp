#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:26:14 2017

@author: ahefny
"""

# In parent directory, run athe following command:
# PYTHONPATH='.:_exp' python -m exp_synth    

import sys, traceback

try:
    import lds
except:
    print ('-'*60)
    print ('WARNING: Could not import lds. LDS model will be ignored.')
    print ('-'*60)
    traceback.print_exc(file=sys.stdout)
    print ('-'*60)    

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import psr_lite.psr_base
import psr_lite.rffpsr as rffpsr
import psr_lite.rffpsr_rnn as rffpsr_rnn
import psr_lite.rnn_filter as rnn_filter
import psr_lite.gru_filter as gru_filter
import psr_lite.feat_extractor as feat_extractor
import psr_lite.psr_eval as psr_eval
import psr_lite.utils.misc
import psr_lite.hsepsr as hsepsr
import time
    
psr_lite.utils.misc.allow_default_rand()
np.random.seed(0)

#%% ##################################
# Load Data
######################################

N_tr = 10
N_tst = 10

data = scipy.io.loadmat('data/synth_data.mat')
X_all = [x.T for x in data['X_all'][0]]
U_all = [u.T for u in data['U_all'][0]]

X_tr = [np.require(x, requirements=['A']) for x in X_all[:N_tr]]
U_tr = [np.require(x, requirements=['A']) for x in U_all[:N_tr]]
X_tst = [np.require(x, requirements=['A']) for x in X_all[-N_tst:]]
U_tst = [np.require(x, requirements=['A']) for x in U_all[-N_tst:]]

#%% ##################################
# Train Models
######################################
def create_feat_set():
    feat_set = feat_extractor.create_RFFPCA_featureset(5000,p_dim,orth=False)    
    feat_set['past'] = feat_extractor.AppendConstant(feat_set['past'])
    return feat_set
    
models = {}
train_time = {}
past = 20
fut = 10
p_dim = 20

# HSEPSR
s = (4.0,2.0,2.0,1.0,1.0)
#models['hsepsr'] = hsepsr.HSEPSR(fut, past, s=s, l2_lambda=5e-4)
#models['hsepsr'].train(X_tr, U_tr)

# RNN
'''
s = time.time()
models['rnn'] = rnn_filter.RNNFilter(5,fut,optimizer_iterations=5000,optimizer_step=1e-3,val_trajs=2)
models['rnn'].train(X_tr, U_tr)
train_time['rnn'] = time.time()-s
'''

s = time.time()
models['gru'] = gru_filter.GRUFilter(5,5,fut,optimizer_iterations=5000,optimizer_step=1e-3,val_trajs=2)
models['gru'].train(X_tr, U_tr, on_unused_input='ignore')
train_time['gru'] = time.time()-s

'''
# RFFPSR
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
                                         optimizer_min_step=1e-8, val_trajs=2, psr_iter=0,
                                         opt_U=False, opt_V=False)

models['rffrnn'].train(X_tr, U_tr)
train_time['rffrnn'] = time.time()-s + train_time['rffpsr']

#RFFPSR-Cond
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
models['rffrnn_cond'] = rffpsr_rnn.RFFPSR_RNN(models['rffpsr_cond'], optimizer='sgd',
                                         optimizer_iterations=100, optimizer_step=1e-5,
                                         optimizer_min_step=1e-8, val_trajs=2, psr_iter=0,
                                         opt_U=False, opt_V=False)

models['rffrnn_cond'].train(X_tr, U_tr)

# RFFARX 
models['rffarx'] = psr_lite.psr_base.AutoRegressiveControlledModel(fut, past, create_feat_set()['past'], 1e-3)
models['rffarx'].train(X_tr, U_tr)

# Last 
models['last'] = psr_lite.psr_base.LastObsModel(X_tr[0].shape[1], fut)
models['last'].train(X_tr, U_tr)
'''

#%% ##################################
# Test Models
######################################  
print(train_time)

mse = {}
burn_in = fut

x_max = max(max(x) for x in X_tr)
x_min = min(min(x) for x in X_tr)
err_fn = lambda x,y: psr_eval.clamped_square_error(x,y,x_min,x_max)

eval_models = ['rffpsr', 'rffrnn', 'rffpsr_cond', 'rffrnn_cond', 'rffrnn_cond',
               'rffarx', 'lds', 'last', 'hsepsr', 'rnn', 'gru']

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
               'hsepsr':('m*-',{'markerfacecolor':'m','ms':8}), 'gru':('g',{})}

for m in mse.items():
    style = {'markerfacecolor':'white'}
    ms = plot_models[m[0]]
    style.update(ms[1])
    plt.plot(np.arange(1,fut+1), m[1], ms[0], **style)
    plt.hold(True)    
    
plt.hold(False)
plt.grid(True)
plt.ylim([0, 0.3])
plt.show()

import cPickle as pickle
with open('mse_synth.pkl', 'wb') as out_file:
    out_file.write(pickle.dumps({'plot_models':plot_models, 'train_time':train_time, 'mse':mse}))

def plot_single_traj(psr, horizon, traj_idx):
    i = traj_idx    
    err,x,s = psr_eval.run_psr_predict_horizon(psr, X_tst[i], U_tst[i], initial_state=None, burn_in=burn_in, err_func=err_fn)
    
    plt.plot(X_tst[i][horizon:])
    plt.hold(True)
    plt.plot(x[horizon].squeeze())
    plt.hold(False)
    plt.draw()
