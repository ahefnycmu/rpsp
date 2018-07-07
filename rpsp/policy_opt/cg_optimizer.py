#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:22:48 2017

@author: ahefny
"""

import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from time import time
from rpsp.rpspnets.psr_lite.utils.nn import dbg_nn_assert_notnan, tf_get_normalized_grad_per_param

def _np2theano(name, np_arr):
    fn = [T.vector, T.matrix, T.tensor3][np_arr.ndim-1]
    return fn(name, dtype=np_arr.dtype)

'''
Operations on list numpy arrays
This equivalent to vectorizing the arrays into a single vector,
performing the operation and the converting the vector back into a list of arrays.
'''
def _arrlist_copy(l):
    return [x.copy() for x in l]

def _arrlist_add(l1, l2, c1=1, c2=1):
    return [c1*x+c2*y for (x,y) in zip (l1,l2)]

def _arrlist_scalein(l,c):
    for x in l:
        x *= c
            
def _arrlist_addin(l1, l2, c=1):
    for (x,y) in zip(l1,l2):
        x += c*y
            
def _arrlist_addout(out,l1, l2, c1=1, c2=1):
    for (z,x,y) in zip(out,l1,l2):
        z[:] = c1*x+c2*y
        
def _arrlist_sub(l1, l2, c1=1, c2=1):
    return [c1*x-c2*y for (x,y) in zip (l1,l2)]
            
def _arrlist_dot(l1, l2):
    return sum([np.sum(x*y) for (x,y) in zip (l1,l2)])            
                
def _t_Hvec(f, y, p):        
    '''
    Given a symbolic function f(x).
    Return a symbolic function h(x,y) = H(x) y, where H(x) is the Hessian of
    f wrt p evaluated at x.
    x,y and p are lists of symbolic variables and y has the same shapes as p.
    
    The method returns a symbolic variable list corresponding 
    to partitions of H(x) y.
    '''            
    
    g = T.grad(f, wrt=p, disconnected_inputs='ignore')
    gy = T.sum([T.sum(gi * yi) for (gi,yi) in zip(g,y)])
    h = T.grad(gy, wrt=p, disconnected_inputs='ignore')
    return h
             
def _Hvec_FD(g, inputs, y, p):            
    current_p = [pi.get_value() for pi in p]
    
    eps=1e-5
    ry = [eps*yi for yi in y]
    
    for pi,cpi,ri in zip(p,current_p,ry):
        pi.set_value(cpi + ri)
        
    gp = g(*inputs)
    
    for pi,cpi,ri in zip(p,current_p,ry):
        pi.set_value(cpi - ri)
        
    gm = g(*inputs)
    
    for pi,cpi in zip(p,current_p):
        pi.set_value(cpi)
            
    return [(ppi-pmi)/(2*eps) for ppi,pmi in zip(gp,gm)]        

def _cg_solve(fA, b, iter=10):
    x = np.zeros_like(b)
    r = b-fA(x)   
    p = r.copy()    
    
    for i in xrange(iter):
        r2old = np.sum(r ** 2)
        if r2old < 1e-10:
            break
        
        Ap = fA(p)        
        alpha = r2old / np.dot(p, Ap)         
        x += alpha * p
        r -= alpha * Ap    
        r2new = np.sum(r ** 2)
        p = r + (r2new / r2old) * p        
    
    return x
    
def _arrlist_cg_solve(fA, b, iter=10):
    '''
    Conjugate gradient method on lists of numpy arrays. 
    fA is a function that takes an array list and outputs an array list.
    b is an array list.
    '''
    x = [np.zeros_like(bb) for bb in b]
    Ax = fA(x)         
    r = _arrlist_sub(b,Ax)   
    p = _arrlist_copy(r)
    r2new = _arrlist_dot(r,r)
    
    for i in xrange(iter):        
        for gg in x: assert not np.isnan(np.sum(gg)), 'NaN x'
        for gg in Ax: assert not np.isnan(np.sum(gg)), 'NaN Ax'
        for gg in r: assert not np.isnan(np.sum(gg)), 'NaN r'
        for gg in p: assert not np.isnan(np.sum(gg)), 'NaN p'
        
        if r2new < 1e-10: break
        r2old = r2new        
        
        Ap = fA(p)        
        alpha = r2old / _arrlist_dot(p, Ap) 
        _arrlist_addin(x, p, alpha)
        _arrlist_addin(r, Ap, -alpha)
        r2new = _arrlist_dot(r, r)
        p = _arrlist_add(r, p, 1.0, (r2new/r2old))        
    
    return x, r

import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres

def _arrlist_lsq_solve(fA, b, iter=10):
    '''
    Min residual method on lists of numpy arrays. 
    fA is a function that takes an array list and outputs an array list.
    b is an array list. x = fA^-1 b
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.minres.html#scipy.sparse.linalg.minres
    '''
    x = [np.zeros_like(bb) for bb in b]
    #shape_arr = [bb.shape for bb in b]
    #for shapea in shape_arr: assert shapea==shape_arr[0] 
    fAOp = LinearOperator( (-1,-1), matvec=fA)
    x,info = minres(fAOp, b, x0=x, shift=1e-8, tol=1e-05)#, maxiter=iter, xtype=None, M=None, callback=None, show=False, check=False)
    #r=(0.0,1e3)[info==0]
    return x, np.abs(info)

class ConstrainedOptimizerOps:
    '''
    Class to specify how to compute costs, gradients and Hessians for ConstrainedOptimizer
    '''
    def constraint(self, *constraint_inputs):
        raise NotImplementedError
        
    def cost(self, *cost_inputs):
        raise NotImplementedError
        
    def cost_grad(self, *cost_inputs):
        raise NotImplementedError
        
    def constraint_Hx(self, constraint_inputs, new_params):
        '''
        Return the result of multiplying the Hessian of the constraint with new_params
        '''
        raise NotImplementedError

class DefaultConstraintOptimizerOps(ConstrainedOptimizerOps):
    def __init__(self, t_cost, t_constraint, t_cost_inputs, t_constraint_inputs,
                 params, checks, reg = 1e-5, normalize=False, clip_bounds=[],
                 hvec='exact'):
        '''
        cost is a symbolic function that takes inputs (e.g. training examples).
        constraint is a symbolic function that takes inputs (e.g. training examples).
        Both cost and constraint must use params in their computation graph.
        '''
        self._normalize = normalize
        t_new_params = [_np2theano(p.name, p.get_value(borrow=True)) for p in params]
                                
        print 'Compiling constraint function ... ',
        s = time()
        self.constraint = theano.function(inputs=t_constraint_inputs, outputs=t_constraint,
                                           on_unused_input='ignore')        
        print 'finished in %f seconds' % (time()-s)
         
        print 'Building cost grad function ... ',
        s = time()
        if self._normalize:
            # if isinstance(t_cost,list):
            #     print 'Normalizing combined TRPO gradients'
            #     res = get_grad_update_old(-t_cost[0],-t_cost[1], params)
            #     t_cost = -res['total_cost']
            #     updates = res['updates']
            #     _t_cost_grad = res['grads']
            # else:
            print 'Normalizing single TRPO gradients'
            _t_cost_grad, weight, updates = tf_get_normalized_grad_per_param(-t_cost, params)
            t_cost = weight*t_cost
        else:
            # if isinstance(t_cost,list):
            #     print 'Summing cost into combined'
            #     t_cost = T.sum(t_cost)
            if len(clip_bounds)==2 and clip_bounds[1]<>0.0:
                t_cost = theano.gradient.grad_clip(t_cost,clip_bounds[0],clip_bounds[1])
            _t_cost_grad = T.grad(-t_cost, wrt=params)    
            updates = []
            
        print 'finished in %f seconds' % (time()-s)
        
        print 'Compiling cost function ... ',                        
        s = time()
        self.cost = theano.function(inputs=t_cost_inputs, outputs=t_cost,
                                    on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s)
        
        print 'Compiling cost grad function ... ',
        s = time()        
        self._cost_grad = theano.function(inputs=t_cost_inputs, outputs=[t_cost]+_t_cost_grad,
                                         on_unused_input='ignore', updates=updates)
                                         #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        print 'finished in %f seconds' % (time()-s)       
                
        if hvec == 'exact':                    
            print 'Building Hx function ... ',
            s = time()
            
            Hx = _t_Hvec(t_constraint, t_new_params, params) 
            Hx = [h + reg*p for (h,p) in zip(Hx,t_new_params)]
            print 'finished in %f seconds' % (time()-s)
                            
            print 'Compiling Hx function ...',
            s = time()
            self._constraint_Hx = theano.function(inputs=t_constraint_inputs+t_new_params,
                                                  outputs=Hx, on_unused_input='ignore')
                                                  #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
            
            self.constraint_Hx = lambda inputs,new_params : self._constraint_Hx(*(inputs+new_params))
            print 'finished in %f seconds' % (time()-s)
        else:            
            assert hvec == 'fd'
            print 'Using finite difference Hvec'
            t_g = T.grad(t_constraint, wrt=params)    
            #t_g = [dbg_nn_assert_notnan(gg, 'ng') for gg in t_g]
            
            g = theano.function(inputs=t_constraint_inputs, outputs=t_g, on_unused_input='ignore')            
            self.constraint_Hx = lambda inputs,new_params : [hvi+reg*vi for hvi,vi in zip(_Hvec_FD(g, inputs=inputs, y=new_params, p=params), new_params)]
                        
        self.checks = theano.function(inputs=t_cost_inputs, outputs=checks,
                                        on_unused_input='ignore')

    def cost_grad(self, *cost_inputs):
        cg = self._cost_grad(*cost_inputs)
        return cg[0], cg[1:]

class GaussianFisherConstraintOptimizerOps(ConstrainedOptimizerOps):        
    def __init__(self, t_cost, t_traj_info, t_inputs, params, reg = 1e-5):
        t_new_params = [_np2theano(p.name, p.get_value(borrow=True)) for p in params]
        
        t_mean = t_traj_info['act_mean']
        t_mean = t_mean.reshape((-1,t_mean.shape[-1]))
        t_logstd = t_traj_info['act_logstd']
        t_logstd = t_logstd.reshape((-1,t_logstd.shape[-1]))
        t_new_mean = t_traj_info['new_act_mean']
        t_new_mean = t_new_mean.reshape((-1,t_new_mean.shape[-1]))
        t_new_logstd = t_traj_info['new_act_logstd']
        t_new_logstd = t_new_logstd.reshape((-1,t_new_logstd.shape[-1]))

        print 'Compiling cost function ... ',                        
        s = time()
        self.cost = theano.function(inputs=t_inputs, outputs=t_cost,
                                    on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s)
        
        print 'Building cost grad function ... ',
        s = time()
        _t_cost_grad = T.grad(-t_cost, wrt=params)
        print 'finished in %f seconds' % (time()-s)    

        print 'Compiling cost grad function ... ',
        s = time()        
        self._cost_grad = theano.function(inputs=t_inputs, outputs=[t_cost]+_t_cost_grad,
                                         on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s) 
        
        print 'Building Hx function ... ',
        s = time()        
        mu = T.concatenate([t_new_mean,t_new_logstd],axis=-1)        
        Jx = sum([T.Rop(mu, p, x) for (p,x) in zip(params,t_new_params)])                        
        M = T.tile(T.eye(2), (mu.shape[0], 1, 1))        
        Jx = Jx.reshape((Jx.shape[0],Jx.shape[1],1))
        Jx = T.tile(Jx, (1,1,Jx.shape[1]))
        MJx = Jx
        JMJx = [T.Lop(MJx, p, x, disconnected_inputs='ignore') for (p,x) in zip(params,t_new_params)]   
        Hx = [h + reg*p for (h,p) in zip(JMJx,t_new_params)]
        print 'finished in %f seconds' % (time()-s)
        
        # TODO: Use mask to handle  different lengths.
        
        print 'Compiling Hx function ...',
        s = time()
        self._constraint_Hx = theano.function(inputs=t_inputs+t_new_params,
                                              outputs=Hx, on_unused_input='ignore')
        
        self.constraint_Hx = lambda inputs,params : self._constraint_Hx(*(inputs+params))
        print 'finished in %f seconds' % (time()-s)
        
    def cost_grad(self, *cost_inputs):
        cg = self._cost_grad(*cost_inputs)
        return cg[0], cg[1:]
                

        
class SemiAutoConstraintOptimizerOps(ConstrainedOptimizerOps):
    '''
    ConstraintOptimizerOps with semi-automatic operation;
    It processes each trajectory using theano while a python for loop takes the 
    sum over trajectories.
    '''
    def __init__(self, t_cost, t_constraint, t_cost_inputs, t_constraint_inputs,
                 params, reg = 1e-5, hessian_traj_ratio = 0.1):
        '''
        The parameters are similar to DefaultConstrainedOptimizerOps except that
        inputs are now for a single trajectory (e.g. matrices instead of tensor3)
        '''
        t_new_params = [_np2theano(p.name, p.get_value(borrow=True)) for p in params]
        self._hessian_traj_ratio = hessian_traj_ratio
                                                
        print 'Compiling constraint function ... ',
        s = time()
        self._constraint_i = theano.function(inputs=t_constraint_inputs, outputs=t_constraint,
                                     on_unused_input='ignore')         
        print 'finished in %f seconds' % (time()-s)
        
        print 'Compiling cost function ... ',                        
        s = time()
        self._cost_i = theano.function(inputs=t_cost_inputs, outputs=t_cost,
                                    on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s)
        
        print 'Building cost grad function ... ',
        s = time()
        _t_cost_grad = T.grad(-t_cost, wrt=params)
        print 'finished in %f seconds' % (time()-s)
        
        print 'Compiling cost grad function ... ',
        s = time()        
        costs = [t_cost]+_t_cost_grad
        self._cost_grad_i = theano.function(inputs=t_cost_inputs, outputs=costs,
                                            on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s)       
        
        print 'Building Hx function ... ',
        s = time()
        Hx = _Hvec(t_constraint, t_new_params, params) 
        Hx = [h + reg*p for (h,p) in zip(Hx,t_new_params)]
        print 'finished in %f seconds' % (time()-s)
                        
        print 'Compiling Hx function ...',
        s = time()
        self._constraint_Hx_i = theano.function(inputs=t_constraint_inputs+t_new_params,
                                                outputs=Hx, on_unused_input='ignore')
        print 'finished in %f seconds' % (time()-s)                
        
    def cost(self, *cost_inputs):
        num_trajs = cost_inputs[0].shape[0]
        return sum(self._cost_i(*[c[i] for c in cost_inputs]) for i in xrange(num_trajs)) / num_trajs

    def constraint(self, *constraint_inputs):
        num_trajs = constraint_inputs[0].shape[0]
        return sum(self._constraint_i(*[c[i] for c in constraint_inputs]) for i in xrange(num_trajs)) / num_trajs

    def cost_grad(self, *cost_inputs):
        num_trajs = cost_inputs[0].shape[0]
        cg = self._cost_grad_i(*[c[0] for c in cost_inputs])
        cost = cg[0]
        grad = cg[1:]
        for i in xrange(1,num_trajs):
            cg = self._cost_grad_i(*[c[i] for c in cost_inputs])
            c_i = cg[0]
            g_i = cg[1:]

            cost += c_i
            _arrlist_addin(grad, g_i)
        
        cost /= num_trajs
        _arrlist_scalein(grad, 1.0 / num_trajs)
        return cost, grad
        
    def constraint_Hx(self, constraint_inputs, new_params):
        num_trajs = int(max(constraint_inputs[0].shape[0] * self._hessian_traj_ratio, 1))
        
        Hx = self._constraint_Hx_i(*([c[0] for c in constraint_inputs] + new_params))
                
        for i in xrange(1,num_trajs):
            Hx_i = self._constraint_Hx_i(*([c[i] for c in constraint_inputs] + new_params))
            _arrlist_addin(Hx, Hx_i)
                
        _arrlist_scalein(Hx, 1.0 / num_trajs)
        return Hx
    
class ConstrainedOptimizer:
    '''
    Approximately solves the problem:
        min cost(params) s.t. constraint(params) < step 
                                
    The approximate solution works as follows:
        1. A search direction s is computed as: s = - H^{-1} g
        where H is the Hessian of the constraint function and g is gradient of
        the cost function both computed at current parameter values. Conjugate
        gradient is used to compute this operation.
        
        2. Line search is performed along s to find the maximum step that 
        decreases the cost while respecting the constraint.
    NOTE: It is assumed that the gradient of the constraint is zero at the current
    value of params. This applies when the constraint is a distance between
    new and current params.
    '''
    def __init__(self, ops, params, step = 0.01, cg_iterations=10, verbose=True):
        '''
        ops is an instance of ConstrainedOptimizerOps that specified how to compute costs, gradients and Hessian
        params is a theano shared variable that is updated when optimize method is called.
        '''
        self._cg_iterations = cg_iterations
        self._step = step
        self._ops = ops                        
                        
        self._params = params        
        self._new_params = [np.zeros_like(p.get_value(borrow=True))  for p in params]                        
        self._verbose = verbose
                            
    def _get_direction(self, inputs, g):                                    
        Ax = lambda x: self._ops.constraint_Hx(inputs,x)
        gd = _arrlist_cg_solve(Ax, g, self._cg_iterations)
        #gd = _arrlist_lsq_solve(Ax, g, self._cg_iterations)
        
        return gd
        
    def optimize(self, cost_inputs, constraint_inputs):         
        cost, g = self._ops.cost_grad(*cost_inputs)  
        
        for gg in g: assert not np.isnan(np.sum(gg)), 'NaN Gradient'
         
        current_params = [p.get_value() for p in self._params]
        s,As = self._get_direction(constraint_inputs, g)
        _arrlist_scalein(As,-1.0)        
        _arrlist_addin(As,g)        
        sAs = _arrlist_dot(s,As)                        
        max_step = np.sqrt(2.*self._step / (sAs + 1e-8))   
        if np.isnan(max_step):
            if self._verbose: print 'NaN max_step .. using 1.0'
            max_step = 1.0
        min_step = 1e-5
        
        rate = 0.8
        alpha = 1.0                
        
        for gg in s: assert not np.isnan(np.sum(gg)), 'NaN Direction'
        
        while(alpha > min_step):
            for (p,c,ss) in zip(self._params, current_params, s):
                p.set_value(c + max_step * alpha * ss)
            new_cnstr = self._ops.constraint(*constraint_inputs)
            new_cost = self._ops.cost(*cost_inputs)
            check_all = self._ops.checks(*cost_inputs)
            if new_cnstr < self._step and new_cost < cost and check_all: break
            alpha *= rate
            
        if new_cost > cost or new_cnstr >= self._step or np.isnan(new_cost) or np.isinf(new_cost) or not check_all:
            for (p,c,ss) in zip(self._params, current_params, s):
                p.set_value(c)
            print 'Loss not improving old, new, check',cost, new_cost, check_all
            new_cost = cost
            new_cnstr = 0.0
        if self._verbose:
            print 'Constrained Optimization finished with:\n\talpha=%f - Gain=%f - Constraint=%f ' % (alpha, cost-new_cost, new_cnstr)    
                        
if __name__ == '__main__':
    def test_arrlist_cg_solve():
        d = 100
        A = np.random.rand(d,d) 
        A = A.T.dot(A) + 1e-1 * np.eye(d)    
        b = np.ones(d)
        
        def fA(x):
            xx = np.concatenate(x)
            y = np.dot(A,xx)
            return [y[:50],y[50:]]
                      
        bb = [b[:50],b[50:]]  
        x,_ = _arrlist_cg_solve(fA, bb, 50) 
    
        err = _arrlist_sub(fA(x), bb)
        err = _arrlist_dot(err,err)
        rel_err = err / np.sum(b ** 2)

        rel_err_np = np.sum((np.dot(A, np.concatenate(x))-b) ** 2) / np.sum(b ** 2)
          
        print err / np.sum(b ** 2)        
        print rel_err_np
        
        assert np.isclose(rel_err,rel_err_np)
        assert rel_err < 1e-4

    test_arrlist_cg_solve()            
    
    def test_constraint_optimizer():
        d = 2
        x = T.vector('x')
        p = T.vector()
        theta = theano.shared(np.ones(d))
        
        t_cost = T.dot(x,theta)
        diff = (p-theta) * np.array([1,2])
        t_constraint = T.dot(diff,diff)
            
        ops = DefaultConstraintOptimizerOps(t_cost, t_constraint, [x], [p], [theta])
        opt = ConstrainedOptimizer(ops, [theta])
        xx = np.ones(d)
        opt.optimize([xx],[theta.get_value()])
        
    test_constraint_optimizer()