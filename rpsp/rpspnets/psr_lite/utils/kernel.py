"""
Created on Wed Jul 27 15:20:04 2016
@author: zmarinho, ahefny
"""

from IPython import embed
import sys
if sys.version_info[0] > 2:
    xrange = range
    
import numpy as np
import numpy.linalg as npla
import ctypes
from warnings import warn
from misc import get_default_rand

def fast_pdist2(X):
    n = X.shape[0]
    if n==1: return [0]
        
    XX = (X*X).sum(1).reshape((1,-1)).repeat(n,axis=0)    
    D = XX + XX.T - 2 * np.dot(X,X.T)
    i = np.arange(0,n*n)
    j = i//n < np.mod(i,n)
    D = D.reshape(-1,1)[j]
    return D    

def precentile_bandwidth(X, p, max=5000, rng=None):    
    if rng is None:         
        rng = get_default_rand()
    
    n = X.shape[0]
    if max==0: max = n
        
    if n > max:
        idx = rng.choice(n,max,False)
        X = X[idx,:]
        
    D = fast_pdist2(X)
    med = np.sqrt(np.percentile(D, p))    
    return med
    
def median_bandwidth(X, max=5000, rng=None):
    '''
    Return the median trick bandwidth, defined as the median of pairwise distances
    between randomly chosen columns of X.
    
    Additional parameter 'max' specifies the number of columns to sample from X
    prior to computing pairwise distances (0 means all columns). 
    '''
    if rng is None: 
        rng = get_default_rand()
    return precentile_bandwidth(X, 50, max, rng)

def median_diag_bandwidth(X, max=5000, rng=None):
    
    if rng is None:         
        rng = get_default_rand()
    d,n = X.shape
    if max==0: max = n
        
    if n > max:
        idx = rng.choice(n,max,False)
        X = X[idx,:]
        n = X.shape[0]

    X1 = X.reshape((n,d,1)).transpose(2,0,1)
    X2 = X.reshape((n,d,1)).transpose(0,2,1)
    D = np.abs(X1-X2)
    
    i = np.arange(0,n*n)
    j = i//n < np.mod(i,n)
    D = D.reshape(-1,d)[j,:]

    s = np.median(D, axis=0)
    s *= np.sqrt(d)
    return s
    
      
def gram_matrix_rbf(X1, X2, sigma):
    '''
    Compute RBF gram matrix 
    Given dxm matrix X1 and dxn matrix X2 computes a matrix G
    s.t. G[i,j] = k(X1[i,:], X2[j,:]), where k is RBF kernel with bandwidth sigma.
    If X1 is a vector the method also returns a vector.
    ''' 
    is_vec = False
    if len(X1.shape) == 1:
        is_vec = True
        X1 = X1.reshape((1,-1))
        
    assert X1.shape[1] == X2.shape[1]
    m = X1.shape[0]
    n = X2.shape[0]
    
    x = np.sum(X1 * X1, 1)
    y = np.sum(X2 * X2, 1)

    G = x.reshape((-1,1)).repeat(n,axis=1) - 2 * np.dot(X1,X2.T) + y.reshape((1,-1)).repeat(m,axis=0)
    G = np.exp(-0.5 * G / (sigma * sigma))
    return G.reshape(-1) if is_vec else G
       
def gram_matrix_diagrbf(X1, X2, sigma):
    is_vec = False
    if len(X1.shape) == 1:
        is_vec = True
        X1 = X1.reshape((1,-1))

    assert X1.shape[1] == X2.shape[1]
    m,d = X1.shape
    n = X2.shape[0]
    
    X1 = X1.reshape((m,d,1)).transpose(0,2,1)
    X2 = X2.reshape((n,d,1)).transpose(2,0,1)
    s = sigma.reshape((1,1,-1))
    
    D = (X1-X2)
    G = np.exp(-0.5*np.sum(D*D/(s*s),axis=2))    
    return G.reshape(-1) if is_vec else G
    
def nystrom(X, num_samples, max_dim=None, gram_function=None, return_Xs_W=False, rng=None):         
    '''
    Compute a feature map using Nystrom approximation.
    See: http://papers.nips.cc/paper/4588-nystrom-method-vs-random-fourier-features-a-theoretical-and-empirical-comparison.pdf
    Returns:
        - f: Handle to feature function
    Optional returns (by setting return_Xs_W=True)    
        - Xs: Data samples used for the appproximation
        - W: Weights W used to compute f (f(x) = W k(Xs,x))    
    '''
    if rng is None:         
        rng = get_default_rand()

    g = gram_function
    if g is None:            
        s = median_bandwidth(X, rng)
        g = lambda X,Y: gram_matrix_rbf(X,Y,s)
        
    k = num_samples
    
    n,_ = X.shape
    k = min(k,n)
    idx = rng.choice(n,size=k,replace=False)
    Xs = X[idx,:]
            
    K = g(Xs,Xs)        
    d,V = npla.eig(K)
    d = np.real(d)
    V = np.real(V)
    
    idx = d.argsort()[::-1]   
    d = d[idx]
    V = V[:,idx]

    r = next((i for i in xrange(k) if d[i]/d[0] < 1e-5), k)
        
    if max_dim is not None and r > max_dim: r = max_dim    
    dd = d[:r]
    VV = V[:,:r]
    
    W = VV / np.sqrt(dd)    
    f = lambda x: np.dot(g(x,Xs),W)
        
    if return_Xs_W:
        return f,Xs,W
    else:
        return f

def _sample_orth_rff(D,d,rng):
    DD = int(np.ceil(1.0*D/d)*d)    
    num_blks = DD//d
    blk = DD//num_blks    
    Q = rng.randn(DD,d)
    S = rng.chisquare(d,(DD,1))
    
    for i in xrange(num_blks):
        s = i*blk
        e = (i+1)*blk
        Q[s:e,:],_ = npla.qr(Q[s:e,:])
        
    Q *= np.sqrt(S)
    return Q[:D,:].T
                        
def sample_rff(num_samples,dim,s,orth=False, rng=None):
    if rng is None:         
        rng = get_default_rand()
    
    if orth:
        return _sample_orth_rff(num_samples,dim, rng) / s
    else:
        return rng.randn(dim,num_samples) / s

def _load_mkl_sincos():
    try:
        mkl = ctypes.cdll.LoadLibrary('libmkl_rt.so')
        in_type = np.ctypeslib.ndpointer(dtype=np.float64)
        out_type = np.ctypeslib.ndpointer(dtype=np.float64, flags='WRITEABLE')
        ptr = mkl.vdSinCos
        ptr.argtypes = [ctypes.c_int64, in_type, out_type, out_type]
        ptr.restype = None
        return ptr
    except:        
        return None

_ptr_mkl_sincos = _load_mkl_sincos()

def _sincos(X, out_sinX, out_cosX):
    _ptr_mkl_sincos(X.size, X, out_sinX, out_cosX)

def _sincos_fallback(X, out_sinX, out_cosX):
    out_sinX.ravel()[:] = np.sin(X).ravel()
    out_cosX.ravel()[:] = np.cos(X).ravel()
        
if _ptr_mkl_sincos is not None:
    _ptr_sincos = _sincos
else:
    warn('''
    Could not load MKL sincos, falling back to Numpy implementation.
    To use MKL sincos, place 'libmkl_rt.so' in LD_LIBRARY_PATH. 
    'libmkl_rt.so' can be found for example in Anaconda Python distribution.
    ''')
    _ptr_sincos = _sincos_fallback    
    
def rff(X,W):    
    Z = np.dot(X,W)
    n = X.shape[0]
    k = W.shape[1]    
    
    output = np.ndarray((2,n,k))    
    _ptr_sincos(Z, output[1,:,:], output[0,:,:]) 
    output = output.transpose(1,0,2).reshape((n,2*k))
    output /= np.sqrt(k)
    return output
    
if __name__ == '__main__':    
    X1 = np.random.rand(5,10)
    X2 = np.random.rand(15,10)
    
    # Test gram_matrix_rbf
    G = gram_matrix_rbf(X1,X2,2)    
    
    for i in xrange(5):
        for j in xrange(15):
            dd = X1[i,:] - X2[j,:]            
            dd = np.exp(-0.5 * np.sum(dd * dd) / 4.0)
            assert np.abs(G[i,j] - dd) < 1e-10
            
    # Test gram_matrix_diagrbf
    s = np.random.rand(10) + 0.1
    s2 = s*s
    G = gram_matrix_diagrbf(X1,X2,s)
    
    for i in xrange(5):
        for j in xrange(15):
            dd = X1[i,:] - X2[j,:]            
            dd = np.exp(-0.5 * np.sum(dd * dd / s2))            
            assert np.abs(G[i,j] - dd) < 1e-10