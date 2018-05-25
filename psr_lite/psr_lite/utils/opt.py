from p3 import *

import numpy as np
import numpy.linalg as npla

def grad_descent(dim, grad_fn, x0=None, step=0.1, num_iterations=1000, obj_fn=None, callback=None):
    if x0 is None:
        x0 = np.zeros(dim)
        
    x = x0
    obj = None
    
    for t in xrange(num_iterations):
        g = grad_fn(x)
        x -= step * g
        
        if obj_fn is not None:
            obj = obj_fn(x)
                    
        if callback is not None:
            callback(t, x, g, obj)
            
    return x
                
'''
Implementation of ADAM optimization algorithm
https://arxiv.org/pdf/1412.6980v8.pdf
'''
class AdamUpdater:
    def __init__(self, dim, b1=0.9, b2=0.999):
        self._dim = dim
        self._b1 = b1
        self._b2 = b2
        self.reset()
    
    def reset(self, mean=None, var=None, b1t=None, b2t=None):
        if b1t is None:
            b1t = self._b1
        if b2t is None:
            b2t = self._b2
        if mean is None:
            mean = np.zeros(self._dim)
        if var is None:
            var = np.zeros(self._dim)
            
        self._mean = mean
        self._var = var
        self._b1t = b1t
        self._b2t = b2t
                
    def get_update(self, grad):
        self._mean *= self._b1
        self._mean += (1-self._b1) * grad

        self._var *= self._b2
        self._var += (1-self._b2) * grad * grad

        mh = self._mean / (1-self._b1t)
        vh = self._var / (1-self._b2t)
        
        self._b1t *= self._b1
        self._b2t *= self._b2
        
        return mh / (np.sqrt(vh)+1e-8)
        
    def create_grad_fn_wrapper(self, grad_fn):
        return lambda x: self.get_update(grad_fn(x))
        
    def create_stoch_grad_fn_wrapper(self, grad_fn):
        return lambda x,start,end: self.get_update(grad_fn(x,start,end))

def numerical_jacobian(x, f, h):
    fx = f(x)
    d_in = x.size
    d_out = fx.size
    
    g = np.empty((d_out, d_in))
    
    for i in xrange(d_in):
        xi = x[i]
        x[i] = xi+h        
        fp = f(x)
        x[i] = xi-h
        fn = f(x)
        x[i] = xi
        
        g[:,i] = (fp-fn)/(2*h)
         
    return g
    
'''
VALIDATE_JACOBIAN: Function to check the Jacobian of a vector-valued
function. The function tests the finite difference and the first order
approximation along randomly chosen directions.
Parameters:
   d - Input dimension
   f - Handle to function
   g - Handle to Jacobian function. For a given vector X, the function
   returns the Jacobian matrix (Note: for scalar functions, this is the 
   transpose of the gradient).
   h - finite difference
   num_trials - Number of tests.
   x - A point sampling function or a fixed point to test at.
   If not provided, a random Gaussian point is chosen for each test.
'''
def validate_jacobian(d, f, g, h, num_trials=10, x=None):
    if x is None:
        x = np.random.randn(d)    
    
    if callable(x):
        gen_x = x
    else:
        gen_x = lambda: x
        
    print('Validating Jacobian')
    max_rel_err = -float('inf')
    max_abs_err = -float('inf')
    
    for t in xrange(num_trials):
        xt = gen_x()
        gx = g(xt)
        if len(gx.shape) == 1:
            gx = gx.reshape((1,-1))

        # Pick a random direction
        delta = np.random.randn(d)
        delta = delta / npla.norm(delta)
        
        # Report error in gradient approximation in the direction of delta
        df = (f(xt+delta*h) - f(xt-delta*h))/(2*h)
        dfh = np.dot(gx,delta.reshape((-1,1))).ravel()
        #assert(df.shape == dfh.shape)
    
        abs_err = npla.norm(df - dfh)
        rel_err = abs_err / npla.norm(df)
        print('Test:%d abs_error=%e rel_error=%e' % (t, abs_err, rel_err))
        
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)    
        
    print('Max error: abs=%e rel=%e' % (max_abs_err, max_rel_err))
   


   

   
            
if __name__ == '__main__':        
    np.random.seed(0)        
    n = 1000
    p = 5
    X = np.random.randn(n,p)
    W = np.random.randn(p,1)
    y = np.dot(X,W) + np.random.randn(n,1) * 0.01
        
    Wls = npla.solve(np.dot(X.T,X),np.dot(X.T,y)).ravel()
    
    def test_adam():
        adam = AdamUpdater(p)
              
        Wh = np.zeros(p)
          
        for t in xrange(50000):
            #g = 2*np.dot(X.T,np.dot(X,Wh)-y)        
            i = np.random.randint(0,n)
            xi = X[i,:]
            g = 2*xi*(np.dot(xi,Wh)-y[i])
            u = adam.get_update(g)
            Wh = Wh - 1.0/(t+1) * u;
            
            print('Iteration = %d   Update = %e  ' % (t,npla.norm(u)),)
            print('Error = %e' % npla.norm(Wh-Wls))
     
    def test_gd():
        g_fn = lambda w: 2*np.dot(X.T,np.dot(X,w.reshape((p,1)))-y).ravel()
        g_fn = AdamUpdater(p).create_grad_fn_wrapper(g_fn)
        
        def gd_callback(t, w, g, obj):
            print('Iteration = %d   Update = %e  ' % (t,npla.norm(g)),)
            print('Error = %e' % npla.norm(w-Wls)) 
    
        grad_descent(p, g_fn, step=1e-1, callback=gd_callback)

    def test_val_jacobian():
        f = lambda x: np.array([x[0]**3+2*x[0]*x[1]+x[1]**2,x[0]]).T
        g = lambda x: np.array([[3*x[0]**2 + 2*x[1],2*x[0]+2*x[1]],[1, 0]])
        h = 1e-10
        
        # Example with a given point.
        x0 = np.array([1,0])
        validate_jacobian(2, f, g, h, 10, x0)
        
        # Example with random points.
        validate_jacobian(2, f, g, h, 10)

    
    #test_adam()
    #test_gd()
    test_val_jacobian()
    
    