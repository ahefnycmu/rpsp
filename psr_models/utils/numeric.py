# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:20:04 2016

@author: zmarinho
"""

import sys
import pdb
import numpy as np
import time
import scipy as sp
import scipy.linalg as spalg
import numpy.linalg
import psr_models.utils.linalg as lg
from IPython import embed
from psr_models.utils.utils import read_matrix, matrix2lists
from psr_models.features.rff_features import RFF_features

np.random.seed()
DEBUG = False



class RefineModelGD(object): #refine psrs
    ''' receive a model, a train feature extractor, and a validation feature extractor and refines parameters\
    model requires: 
        - get/set parameters
        - filter_trajs
        - compute_state_features
        - build_model
        - update_states
        - bp_traj
        '''
    def __init__(self, rstep=0.1, optimizer='sgd', val_batch=3,\
                refine_adam = False, min_rstep=1e-6):
        self.rstep = rstep
        self.optimizer=optimizer
        self.val_batch = val_batch
        self.refine_adam = refine_adam
        self.min_rstep = min_rstep
        return
        
    def initial_validation(self, eval_feats, eval_data, model, verbose=DEBUG):
        ''' initial error'''
        model_parameters = model.get_parameters()
        self.eval_states = []
        
        states0 = np.vstack( [model.compute_state_features(\
                                eval_feats.past[:,eval_feats.locs[j]]) \
                             for j in xrange(eval_feats.num_seqs)] ).T
        
        self.eval_states = model.filter_trajs(states0=states0, feats=eval_feats) #validation states
        
        init_error = self.validation_error(self.eval_states, eval_feats, eval_data, model_parameters)
        if verbose: print('Initial Error it_%d= %f\n'%(0,init_error))
        
        self.best_val_error = init_error
        self.batch_val_error = np.inf
        self.last_batch_val_error = np.inf
        self.best_val_iteration = 0
        self.last_best_model = dict(model_parameters)
        self.best_model = dict(model_parameters)
        return init_error
 
    def validate(self, i, eval_feats, eval_data, model, verbose=DEBUG):
        ''' validate in validation set'''
        if verbose: print('validate')
        #CHECK FOR BUGS NOT TESTED
        model_parameters = model.get_parameters()
        states0 = np.vstack([self.eval_states[:, eval_feats.locs[t]] for t in xrange(eval_feats.num_seqs)]).T
        self.eval_states = model.filter_trajs(states0=states0, feats=eval_feats)
        error = self.validation_error( self.eval_states, eval_feats, eval_data, model_parameters)
        if verbose: print('Validation Error iter_%d= %f'%(i,error))
         
        if error < self.best_val_error:
            self.best_model = dict(model_parameters)
            self.best_val_iteration = i
            self.best_val_error = error
        self.batch_val_error = np.min([error, self.batch_val_error])
        if np.mod(i, self.val_batch) == 0: #check for early stopping
            eps = 1e-3
            # End of validation batch. Check for early stopping.
            if self.batch_val_error > (1+eps) * self.last_batch_val_error:
                # Large increase in error
                # Try decreasing step size
                self.rstep = self.rstep / float(2)
                if self.rstep < self.min_rstep:                
                    print('Early stopping after %d iterations'% i)
                    return 1
                else:
                    print('Reduced step size to %e at iteration %d'%( self.rstep, i))
                    i = i - self.val_batch                #decrease number of iterations by val batch
                    model_parameters = self.last_best_model
            elif self.batch_val_error > (1-eps) * self.last_batch_val_error:
                # Small change in error. Stop
                print('Early stopping after %d iterations'% i)
                return 1
            else:
                # Probably can still improve, proceed.
                self.last_batch_val_error = self.batch_val_error
                self.last_best_model = self.best_model
                self.batch_val_error = np.inf
        return 0
 
    def model_refine(self, model, feats, data, val_feats = None, val_data=None,  n_iter=3, \
                     verbose=DEBUG, reg=1e-6, wpred=1.0, wnext=0.0):
        ''' refine model parameters for n_iter iterations'''
        if verbose: print(' model refine')
        ## Refinement       
        tic = time.time()
        
        model_parameters_start = model.get_parameters() 
        # Compute validation error
        if val_feats is None :
            use_validation = False
            init_error = self.initial_validation(feats, data, model)
        else:
            use_validation = True
            init_error = self.initial_validation(val_feats, val_data, model)
        #self.val_batch = val_batch #check
        contexts = dict([(key,{}) for key in model_parameters_start.iterkeys()])
        i=1
        states=[]
        model_parameters = dict(model_parameters_start)
        while i <= n_iter:                            
            if verbose: print('\nRefinement - Round %d of %d\n'%( i, n_iter-1))
            tic = time.time()
            for t in xrange(feats.num_seqs):          
                start = feats.locs[t]
                L = feats.locs[t+1] - feats.locs[t]
                f0 = model.compute_state_features(feats.past[:,start])
                assert not np.isnan(f0).any(), embed()
                if verbose: print ('f0' ,f0)
                traj_states, filter_traj = model.iterative_filter(feats, f0, L=L,i=start)
                states.append(traj_states)
                assert(len(filter_traj))>0, embed()
                # Back propagation
                gradients = model.bp_traj(traj_states, filter_traj, feats, data, model_parameters,\
                                           seq=t, reg=reg, wpred=wpred, wnext=wnext)
               
                #update parameters
                for key in model_parameters.iterkeys():
                    if self.refine_adam:
                        gradients[key], contexts[key] = adam(gradients[key], contexts[key])
                    model_parameters[key] = model_parameters[key] - self.rstep * gradients[key]
                    if verbose: print('grad norm ',np.linalg.norm(gradients[key]) )
                model._set_parameters(model_parameters)
            #validate all                                      
            if use_validation:     
                early_stop = self.validate(i, val_feats, val_data, model)
                if early_stop:
                    break
            else:
                states0 = np.vstack([self.eval_states[:, feats.locs[t]] for t in xrange(feats.num_seqs)]).T
                self.eval_states = model.filter_trajs(states0=states0, feats=feats)
                error = self.validation_error( self.eval_states, feats, data,  model_parameters)
                if verbose: print('Error min i_%d= %f\n'%(i,error))  
            toc = time.time()   
            if verbose: print('iter %d took %f secs\n'%(i, toc-tic))
            i = i+1
        if use_validation:
            model_parameters = dict(self.best_model)
            if verbose:
                print('\nFinished refinement in %d iterations\n'% (i-1))
                print('Using weights from iteration %d with error = %f\n'% (self.best_val_iteration, self.best_val_error))
            print('best Validation Error (len: %d): start=%f end=%f iter=%d'%( data.obs.shape[1], init_error, self.best_val_error, self.best_val_iteration))
        else:
            if verbose: print('Error: start=%f end=%f\n iter=%d'%( init_error, error, i))
        #set parameters
        model._set_parameters(model_parameters)
        #update states
        states0 = np.vstack( [model.compute_state_features(\
                                feats.past[:,feats.locs[j]]) \
                             for j in xrange(feats.num_seqs)] ).T
        #build model
        states = model._build_model(feats, states0=states0)
        trainerror = self.validation_error(states, feats, data, model.get_parameters())
        valerror=np.inf
        if use_validation:
            states0 = np.vstack([self.eval_states[:, val_feats.locs[t]] for t in xrange(val_feats.num_seqs)]).T
            self.eval_states = model.filter_trajs(states0=states0, feats=val_feats)
            valerror = self.validation_error( self.eval_states, val_feats, val_data,  model.get_parameters())
       
        print 'validation error after build val:', valerror, ' train:',trainerror        
        return states

    def validation_error(self, states, feats, data, model): 
        ''' validation error of all trajectories'''
        if DEBUG: print ('validation error')
        # Horizon Error
        #start = feat_ext.locs[seq]
        #end = feat_ext.locs[seq+1]
        val_rf_to_in = lg.khatri_dot(states, feats.fut_act); #[:,start:end]
        val_rf_to_out = data.fut_obs; #[:, start:end]   
        val_err = np.linalg.norm(val_rf_to_out - np.dot(model['Wfut'] , val_rf_to_in), 'fro')/np.linalg.norm(val_rf_to_out)
        return val_err





def adam( grad, context={} ):
    #ADAM Implementation of ADAM optimization algorithm
    # https://arxiv.org/pdf/1412.6980v8.pdf
    if DEBUG: print ('ADAM')
    if len(context)==0: 
        context['m'] = np.zeros(grad.shape, dtype=float)
        context['v'] = np.zeros(grad.shape, dtype=float)
        context['b1'] = 0.9
        context['b2'] = 0.999 
        context['b1t'] = context['b1']
        context['b2t'] = context['b2']
    
    new_context = context
    new_context['m'] = context['b1'] * context['m'] + (1-context['b1']) * grad
    new_context['v'] = context['b2'] * context['v'] + (1-context['b2']) * grad * grad #el-wise dot

    mh = new_context['m'] / float(1 - context['b1t'])
    vh = new_context['v'] / float(1 - context['b2t'])

    update = mh / (np.sqrt(vh) + 1e-8)

    new_context['b1t'] = context['b1t'] * context['b1']
    new_context['b2t'] = context['b2t'] * context['b2']
    return update, new_context

def unit_test():
    np.random.seed(0)
    n = 1000; p = 1;
    X = np.random.normal(size=(n,p))
    W = np.random.normal(size=(p,1))
    y = np.dot(X,W) + np.random.normal(size=(n,1)) * 0.01
    
    Wls = np.dot( np.linalg.inv(np.dot(X.T,X)), np.dot(X.T,y) )
    c = {}
    Wfut = np.zeros((p,1))
        
    for t in xrange(50000):
        g = 2*np.dot(X.T, np.dot(X,Wfut) - y )
        i = np.random.randint(n) 
        xi = X[i,:]
        g = 2*np.dot(xi.T, np.dot(xi,Wfut) - y[i])        
        u,c = adam(g, c)
        Wfut = Wfut - 1.0 / float(t) * u;
        
        print('Update = %e  '% np.linalg.norm(u))
        print('Error = %e\n'% np.linalg.norm(Wfut-Wls))
    return

def test_validation(inputfile):
    inputfile='examples/psr/data/rff/'
    X = read_matrix(inputfile, name='X.txt', delim=',')
    d = 5
    trainsize = 200+1+2*d
    train_obs = X[:,d:d+trainsize]
    train_actions = train_obs
    
    rff_feats = RFF_features(use_actions=True, dim=d, filedir='examples/psr/data/rff/tests/')
    rff_feats.compute_train_features(train_obs, train_actions)
    
    

    
    
    
    
def test_bp():
    inputfile   = 'examples/psr/data/controlled/'
    d           = 1
    N           = 10
    L           = 100
    dim         = 10
    past        = 20
    rstep       = 0.01
    min_rstep   = 1e-5
    refine      = 100
    val_batch   = 5
    rff_dim     = 5000
    Xtr = read_matrix(inputfile, name='Xtr.txt', delim=',')
    Utr = read_matrix(inputfile, name='Utr.txt', delim=',')
    Xtr = matrix2lists( Xtr, d, N=N, L=L)
    Utr = matrix2lists( Utr, d, N=N, L=L)
    Lmax        = np.max([Xtr[j].shape[1] for j in xrange(len(Xtr))])
    rff_feats = RFF_features(use_actions=True, dim=dim, filedir='examples/psr/data/rff/tests/',\
                              k=rff_dim, r_dim=dim, past_dim=past)
    rff_feats.compute_features(Xtr, Utr)
    from psr_models.covariance_psr import covariancePSR
    psr = covariancePSR(dim=50, hS=1, fE=100, tH=29, use_actions=False, file=inputfile)
    psr.train(rff_feats)
    
    # Actual data
    i = np.random.randint(rff_feats.past.shape[1]-1)
    print i
    f = psr.compute_state_features(rff_feats.past[:,i])
    o_feat = rff_feats.Uo_fx[:,i]
    a_feat = rff_feats.Ua_fx[:,i]
    oo = rff_feats.Uoo_fx[:,i+1]
    aa = rff_feats.Ua_fx[:,i+1]
    
    f_diff = lambda f,Wex,Woo: (oo - np.dot(np.dot(Woo , psr.filter_core(f, o_feat, a_feat, Woo=Woo,Wex=Wex)[0])\
                                           .reshape((-1, rff_feats.S_a), order='F') , aa) )
    f_err = lambda f,Wex,Woo: np.linalg.norm(f_diff(f,Wex,Woo))**2
    
    
    
    diff = f_diff(f,psr.W_s2ext, psr.W_s2oo)
    g_err_sf = -2*np.dot(lg.khatri_dot(aa[:,None],diff[:,None]).T , psr.W_s2oo)
    
    model = psr.get_parameters()
    #jac_a = rff_feats.a_grad_f(rff_feats.actions[:,0].reshape(-1,1))
    jac_a = rff_feats.a_grad[:,i]
    sf, ftraj = psr.filter_core(f, o_feat, a_feat)
    
    g_f, g_Wex, g_Woo, g_a, g_a1, g_a2 = psr.backprop(g_err_sf, f, o_feat, a_feat, model, rff_feats,1e-6, ftraj, g_a=jac_a)

    g_err_Woo = -2*lg.khatri_dot( sf[:,0:None], lg.khatri_dot(aa[:,None],diff[:,None]) ).T
    
    h = 1e-10
    # Gradient w.r.t input state
    fx = lambda fo: f_err(fo,psr.W_s2ext,psr.W_s2oo)
    delta = np.eye(len(f),1)[:,0] #np.random.randn(len(f))
    delta = delta / np.linalg.norm(delta)

    df = (fx(f+delta*h)-fx(f-delta*h))/float(2*h)
    dfh = np.dot( g_f, delta )
    abs_err = np.abs(df - dfh)
    rel_err = abs_err / np.abs(df)
    print('Input: abs_error=%e rel_error=%e\n'%(abs_err, rel_err))
    
    # Gradient w.r.t W.s2_oo
    fx = lambda Woo: f_err(f,psr.W_s2ext,Woo)
    delta = np.eye(psr.W_s2oo.shape[0],psr.W_s2oo.shape[1])#np.random.randn(psr.W_s2oo.shape[0],psr.W_s2oo.shape[1])
    delta = delta / np.linalg.norm(delta)
    df = (fx(psr.W_s2oo+delta*h)-fx(psr.W_s2oo-delta*h))/float(2*h)
    dfh = np.dot((g_Woo+g_err_Woo), delta.reshape((-1,1), order='F') )
    abs_err = np.abs(df - dfh)
    rel_err = abs_err / np.abs(df)
    print('W.s2_oo: abs_error=%e rel_error=%e\n'%( abs_err, rel_err))
   
    # Gradient w.r.t W.s2_ex
    fx = lambda Wex: f_err(f,Wex,psr.W_s2oo)
    delta = np.eye(psr.W_s2ext.shape[0],psr.W_s2ext.shape[1])#np.random.randn(psr.W_s2ext.shape[0],psr.W_s2ext.shape[1])
    delta = delta / np.linalg.norm(delta)
    df = (fx(psr.W_s2ext+delta*h)-fx(psr.W_s2ext-delta*h))/float(2*h)
    dfh = np.dot( g_Wex, delta.reshape((-1,1), order='F') )
    abs_err = np.abs(df - dfh)
    rel_err = abs_err / np.abs(df)
    print('W.s2_ex: abs_error=%e rel_error=%e\n'%( abs_err, rel_err))
    
    
    
    fa_diff = lambda f,Wex,Woo, a_fx: (oo - np.dot(np.dot(Woo , psr.filter_core(f, o_feat, a_fx, Woo=Woo,Wex=Wex)[0])\
                                           .reshape((-1, rff_feats.S_a), order='F') , aa) )
    fa_err = lambda f,Wex,Woo, a_fx: np.linalg.norm(fa_diff(f,Wex,Woo, a_fx))**2


    fa1_diff = lambda f,Wex,Woo, a_fx1, a_fx2: (oo - np.dot(np.dot(Woo , psr.filter_a(f, o_feat, a_fx1=a_fx1, a_fx2=a_fx2, Woo=Woo,Wex=Wex)[0])\
                                           .reshape((-1, rff_feats.S_a), order='F') , aa) )
    fa1_err = lambda f,Wex,Woo, a_fx1, a_fx2: np.linalg.norm(fa1_diff(f,Wex,Woo, a_fx1, a_fx2))**2


 
    delta = np.ones(g_a.shape[0]) 
    delta = delta / np.linalg.norm(delta)

    jac_a_f = lambda a: rff_feats.proj_a(a)
    djac = (jac_a_f(rff_feats.actions[:,i]+h)-jac_a_f(rff_feats.actions[:,i]-h))/float(2*h)
    print ('jacobian error:', np.linalg.norm(jac_a- djac))
    
    fx = lambda a: fa_err(f,psr.W_s2ext,psr.W_s2oo, rff_feats.proj_a(a))
    df = (fx(rff_feats.actions[:,i]+h)-fx(rff_feats.actions[:,i]-h))/float(2*h)
    dfh = np.dot(g_a,delta)
    abs_err = np.abs(df - dfh)
    rel_err = abs_err / float(np.abs(df))
    print('g_a: abs_error=%e rel_error=%e\n'%( abs_err, rel_err))
    
    fxa = lambda a1, a2: fa1_err(f,psr.W_s2ext,psr.W_s2oo, rff_feats.proj_a(a1), rff_feats.proj_a(a2))
    df1 = (fxa(rff_feats.actions[:,i]+h, rff_feats.actions[:,i])-fxa(rff_feats.actions[:,i]-h, rff_feats.actions[:,i]))/float(2*h)
    dfh1 = np.dot(g_a1,delta)
    abs_err = np.abs(df1 - dfh1)
    rel_err = abs_err / float(np.abs(df1))
    print('g_a1: abs_error=%e rel_error=%e\n'%( abs_err, rel_err))
    
    df2 = (fxa(rff_feats.actions[:,i], rff_feats.actions[:,i]+h)-fxa(rff_feats.actions[:,i], rff_feats.actions[:,i]-h))/float(2*h)
    dfh2 = np.dot(g_a2,delta)
    abs_err = np.abs(df2 - dfh2)
    rel_err = abs_err / float(np.abs(df2))
    print('g_a2: abs_error=%e rel_error=%e\n'%( abs_err, rel_err))
    embed() 

if __name__=='__main__':
    #test_dataProjection(sys.argv[1:])
    #test_validation(sys.argv[1:])
    test_bp()