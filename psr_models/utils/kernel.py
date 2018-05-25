"""
Created on Wed Jul 27 15:20:04 2016
@author: zmarinho and ahefny
"""

import numpy as np
import ctypes
import psr_models.utils.linalg as lg
from IPython import embed
from numpy import ndarray

def fast_pdist2(X):
    n = X.shape[1]
    if n==1: return [0]
        
    XX = (X*X).sum(0).reshape((1,-1)).repeat(n,axis=0)    
    D = XX + XX.T - 2* np.dot(X.T,X)
    i = np.arange(0,n*n)
    j = i//n < np.mod(i,n)
    D = D.reshape(-1,1)[j]
    return D

'''
Return the median trick bandwidth, defined as the median of pairwise distances
between randomly chosen columns of X.

Additional parameter 'max' specifies the number of columns to sample from X
prior to computing pairwise distances (0 means all columns). 
'''
def median_bandwidth(X, max=5000):
    n = X.shape[1]
    if max==0: max = n
        
    if n > max:
        idx = np.random.choice(n,max,False)
        X = X[:,idx]
        
    D = fast_pdist2(X)
    med = np.sqrt(np.median(D))
    return med

def percentile_bandwidth(X, p=75, max=5000):
    n = X.shape[1]
    if max==0: max = n
        
    if n > max:
        idx = np.random.choice(n,max,False)
        X = X[:,idx]
    Dist = fast_pdist2(X)
    med = np.sqrt(np.percentile(Dist,p))
    return med

def percentile_precision(X, p=75, max=5000):
    D, n = X.shape
    if max==0: max = n  
    if n > max:
        idx = np.random.choice(n,max,False)
        X = X[:,idx]
    sigma=[]; 
    for d in xrange(D):
        Dist = fast_pdist2(X[d,:].reshape(1,-1))
        med = np.sqrt(np.percentile(Dist,p))
        sigma.append(med)
    return np.array(sigma)

'''
Compute a feature map using Nystrom approximation for RBF kernel.
See: http://papers.nips.cc/paper/4588-nystrom-method-vs-random-fourier-features-a-theoretical-and-empirical-comparison.pdf
Returns:
    - f: Handle to feature function
Optional returns (by setting return_Xs_W=True)    
    - Xs: Data samples used for the appproximation
    - W: Weights W used to compute f (f(x) = W k(Xs,x))    
'''
def nystrom(X, num_samples, bandwidth, return_Xs_W=False):    
    s = bandwidth
    k = num_samples
    
    _,n = X.shape
    k = min(k,n)
    idx = np.random.choice(n,size=k,replace=False)
    Xs = X[:,idx]
            
    K = gram_matrix_rbf(Xs,Xs,s)    
    #V,D = np.linalg.eig(K)
    
    [d,V] = np.linalg.eig(K)
    d = np.real(d)
    V = np.real(V)
    r = (d / d[0]) >= 1e-5
    
    dd = d[r]
    VV = V[:,r]
    
    W = VV.T / np.sqrt(dd).reshape((-1,1))
    f = lambda x: np.dot(W, gram_matrix_rbf(Xs,x,s))

    if return_Xs_W:
        return f,Xs,W
    else:
        return f
    
def nystrom_grad(X, W, Xs, s ):    
    f = lambda x: np.dot(W, gram_matrix_rbf_grad(Xs,x,s))    
    return f


def _sample_orth_rff(D,d):
    DD = int(np.ceil(1.0*D/d)*d)    
    num_blks = DD//d
    blk = DD//num_blks    
    Q = np.random.randn(DD,d)
    S = np.random.chisquare(d,(DD,1))
    
    for i in xrange(num_blks):
        s = i*blk
        e = (i+1)*blk
        Q[s:e,:],_ = np.linalg.qr(Q[s:e,:])
        
    Q *= np.sqrt(S)
    return Q[:D,:]

'''
Compute RBF gram matrix 
Given dxm matrix X1 and dxn matrix X2 computes a matrix G
s.t. G(i,j) = k(X1(:,i), X2(:,j)), where k is RBF kernel with bandwidth sigma is the precision matrix S^-1.
'''    
def gram_matrix_rbf(X1, X2, sigma):
    assert X1.shape[0] == X2.shape[0]
    m = X1.shape[1]
    n = X2.shape[1]
    X1 = (1./sigma*X1.T).T
    X2 = (1./sigma*X2.T).T
    x = np.sum(X1 * X1, 0)
    y = np.sum(X2 * X2, 0)
    G = x.reshape((-1,1)).repeat(n,axis=1) - 2 * np.dot(X1.T,X2) + y.reshape((1,-1)).repeat(m,axis=0)
    G = np.exp(-0.5 * G)
    return G

def gram_matrix_rbf_grad(X1, X2, sigma):
    assert X1.shape[0] == X2.shape[0]
    m = X1.shape[1]
    n = X2.shape[1]
    X1 = (1./sigma*X1.T).T
    X2 = (1./sigma*X2.T).T
    x = np.sum(X1 * X1, 0)
    y = np.sum(X2 * X2, 0)
    G = x.reshape((-1,1)).repeat(n,axis=1) - 2 * np.dot(X1.T,X2) + y.reshape((1,-1)).repeat(m,axis=0)    
    if X2.ndim==2:
        G = -np.exp(-0.5 * G ) * (np.asarray([X2.sum(0)]*X1.shape[1]) - np.asarray([X1.sum(0)]*X2.shape[1]).T)
    else:
        G = -np.exp(-0.5 * G) * (np.asarray([X2.squeeze()]*X1.shape[1]) - np.asarray([X1.squeeze()]*X2.shape[1]).T)
    return G

def sample_rff(num_samples,dim,s):
    return np.random.randn(num_samples,dim) / s


def power_rff(X, num_samples):
    dim = X.shape[0]
    W = np.zeros((num_samples),dim)
    for i in xrange(dim):
        ps = np.abs(np.fft.fft(X[:,i]))**2
        freqs = np.fft.fftfreq(X[:,i].size, 1./30.0)
        idx = np.argsort(freqs)
        W[:,i] = idx[:num_samples]
    return 
    


def _load_mkl_cossin():
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

_ptr_mkl_cossin = _load_mkl_cossin()

def _cossin(X, out_sinX, out_cosX):
    _ptr_mkl_cossin(X.size, X, out_sinX, out_cosX)

def _cossin_fallback(X, out_sinX, out_cosX):
    out_sinX.ravel()[:] = np.sin(X).ravel()
    out_cosX.ravel()[:] = np.cos(X).ravel()
        
if _ptr_mkl_cossin is not None:
    _ptr_cossin = _cossin
else:
    _ptr_cossin = _cossin_fallback    
    
def rff(W,X):    
    if X.ndim==1:
        X = X.reshape(-1,1)
    Z = np.dot(W,X)
    n = X.shape[1]
    k = W.shape[0]
    
    output = np.ndarray((2*k,n))    
    _ptr_cossin(Z, output[k:,:], output[:k,:]) 
    output /= np.sqrt(k)
    return output

def rff_grad(W,X):
    if X.ndim==1:
        X = X.reshape(-1,1)
    Z = np.dot(W,X)
    n = X.shape[1]
    k = W.shape[0]
    g = np.ndarray((2*k,n))  
    _ptr_cossin(Z, g[:k,:], g[k:,:])
    g[:k,:] = -g[:k,:]
    g /= np.sqrt(k)
    Z = np.vstack([W,W])
    grad = lg.khatri_dot(Z.T,g.T).T
    return grad



    
if __name__ == '__main__':    
    X1 = np.random.rand(10,5)
    X2 = np.random.rand(10,15)
    
    G = gram_matrix_rbf(X1,X2,2)
    
    for i in xrange(5):
        for j in xrange(15):
            dd = X1[:,i] - X2[:,j]            
            dd = np.exp(-0.5 * np.sum(dd * dd) / 4.0)
            assert np.abs(G[i,j] - dd < 1e-10)
    