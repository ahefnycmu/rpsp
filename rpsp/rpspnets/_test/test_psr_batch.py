'''
A test for batch state update of rffpsr_rnn
'''
import numpy as np

import rpspnets.rffpsr as rffpsr
import rpspnets.rffpsr_rnn as rffpsr_rnn
import rpspnets.feat_extractor as feat_extractor

import theano
import theano.tensor as T

seed = 0
np.random.seed(seed)

N_trn = 2
fut = 10
past = 20

def load_data(N_traj):
    trajs = np.random.rand(N_traj, 50, 5)
        
    X_obs = [trajs[i,:,:3] for i in xrange(N_traj)]
    X_act = [trajs[i,:,3:] for i in xrange(N_traj)]            
    return X_obs, X_act
 
X_obs,X_act = load_data(N_trn)    

####################################################
# Train Models
####################################################        
models = {}

p_dim=20
rng = np.random.RandomState(0)
feat_set = feat_extractor.create_RFFPCA_featureset(5000,p_dim,orth=False,rng=rng)
#feat_set['past'] = rpspnets.feat_extractor.NystromFeatureExtractor(np.inf, rng=rng)
l2_lambda = rffpsr.uniform_lambda(1e-3)

psr_settings = {'rng':rng,
                'feature_set': feat_set, 
                'projection_dim': p_dim,
                's1_method': 'joint',
                'l2_lambda': l2_lambda,
                'past_projection' : 'svd'}
               
# RFFPSR                              
psr = rffpsr.RFFPSR(fut,past,**psr_settings)
psr.train(X_obs, X_act)

# RFFPSR With Refinment
rnn = rffpsr_rnn.RFFPSR_RNN(psr, optimizer='sgd',
                                         optimizer_iterations=1, optimizer_step=1e-5,
                                         optimizer_min_step=1e-8, val_trajs=1, 
                                         psr_cond='kbr', psr_iter=5,
                                         opt_U=False, opt_V=False)
rnn.train(X_obs, X_act)

N = 5
a_feat = np.random.rand(N, psr._feat_dim.act)
o_feat = np.random.rand(N, psr._feat_dim.obs)
state = np.random.rand(N, psr._feat_dim.state)

t_a_feat = T.vector()
t_o_feat = T.vector()
t_state = T.vector()

t_a_feat_mat = T.matrix()
t_o_feat_mat = T.matrix()
t_state_mat = T.matrix()

h1 = [None] * N
t_h1 = rnn.tf_update_state(t_state, t_o_feat, t_a_feat)
#t_h1 = rnn._dbg.out[-1]
f1 = theano.function(inputs=[t_state, t_o_feat, t_a_feat], outputs=t_h1, on_unused_input='ignore')
for i in xrange(N):    
    h1[i] = f1(state[i], o_feat[i], a_feat[i])
h1 = np.array(h1)

t_h2 = rnn.tf_update_state_batch(t_state_mat, t_o_feat_mat, t_a_feat_mat)
#t_h2 = rnn._dbg_batch.out[-1]
f2 = theano.function(inputs=[t_state_mat, t_o_feat_mat, t_a_feat_mat], outputs=t_h2, on_unused_input='ignore')
h2 = f2(state, o_feat, a_feat)

assert np.allclose(h1,h2)
print 'Test Successful !!!'
