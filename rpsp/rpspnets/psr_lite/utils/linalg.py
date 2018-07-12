"""
Created on Wed Jul 27 15:20:04 2016
@author: zmarinho and ahefny
"""
from p3 import *

import numpy as np
import scipy.linalg as spla
from misc import get_default_rand
from scipy.sparse.linalg import cg


def blk_fn_row(f_blk, n, blk_size = 1000):
    '''
    Compute a function that returns a matirx by iterating through row blocks.
    f_blk is a function such that f_blk(s,e) returns rows s:e of the output 
    matrix.
    '''
    num_blocks = int(np.ceil(n/float(blk_size)))
    
    d = f_blk(0,2).shape[1]
    output = np.zeros((n,d))
    
    for b in xrange(num_blocks):
        blk_start = b*blk_size
        blk_end = np.min([n, blk_start+blk_size])        
        output[blk_start:blk_end,:] = f_blk(blk_start,blk_end)

    return output    

def blk_sum(f_blk, n, blk_size = 1000):
    '''
    Compute sum_{i=1}^n f(i).
    f_blk is a function such that f_blk(s,e) sum_{i=s}^{e-1} f(i)
    '''
    num_blocks = int(np.ceil(n/float(blk_size)))
    
    output = np.zeros_like(f_blk(0,2))
    
    for b in xrange(num_blocks):
        blk_start = b*blk_size
        blk_end = np.min([n, blk_start+blk_size])        
        output += f_blk(blk_start,blk_end)

    return output    

def blk_dot(f, g, n, blk_size = 1000):
    f_blk = lambda s,e: np.dot(f(s,e), g(s,e))
    return blk_sum(f_blk, n, blk_size)

def orth(A, scale = 1):# 4
    '''
    Construct an orthonormal basis for the range of A using SVD
    A : (M, N) returns Q : (M, K) 
        Orthonormal basis for the range of A.
        scale: how much scaling on the eps for tolerance
        K = effective rank of A, as determined by automatic cutoff
    '''

    u, s, vh = spla.svd(A, full_matrices=False)
    M, N = A.shape
    eps = np.finfo(float).eps*scale
    tol = max(M,N) * np.amax(s) * eps
    num = np.sum(s > tol, dtype=int)
    Q = u[:,:num]
    return Q


def rand_svd_f(X, f=None, k=10, compute_ufx=True, it=2, slack=0, blk=1000, rng=None): 
    '''
    Compute randomized left singular vectors of a dxn matrix X represented by a column sampling function f. 
    The method uses randomized SVD algorithm by Halko, Martinson and Tropp.
    Parameters:
     f - Column sampling function: f(s,e) should return columns s through e of the input matrix.
     n - Number of columns of the input matrix.
     k - Maximum number of singular vectors.
     it - (optional, default=2) The exponent q to premultiply (XX')^q by X to suppress small eigen values.
     slack - (optional, default=0) Extra dimensions to use in intermediate steps.
     blk (optional, default=1000) - Number of input columns that can be stored in memory.
    Outputs:
     U - Singular vectors.
     S - Singular values.
     UX - Projected input matrix U' * X    
    '''    
    #k = np.random.randint(10, 20)
    
    if rng is None:         
        rng = get_default_rand()
    if f is None: f = lambda X:X    
            
    n = X.shape[1]
    x = f(X[:,:1])
    d = x.shape[0]
    if d <= k:
        # No need for SVD
        U = np.eye(d)
        S = np.ones((1,d)) # Dummy values
        UX = f(X[:,:n])
        return U, S, UX
    
    p = k + slack
    num_blocks = int(np.ceil(n/float(blk)) )
    
    K = np.zeros((d,p))
    Pb = rng.normal(size=(n,p))
    
    for b in xrange(num_blocks):
        blk_start = b*blk
        blk_end = np.min([n, blk_start+blk])
        Xb = f(X[:,blk_start:blk_end])    
        K += np.dot(Xb , Pb[blk_start:blk_end,:] )        
    
    for i in xrange(it):
        KK = np.zeros((d,p))
        for b in xrange(num_blocks):
            blk_start = b*blk
            blk_end = np.min([n, blk_start+blk])
            Xb = f(X[:,blk_start:blk_end])
            KK += np.dot( Xb , np.dot(Xb.T , K) )
        
        K = KK / float( np.max(np.abs(KK)) )

    Q = orth(K, scale=0.1)
    #Q = spla.orth(K)
    p = Q.shape[1]
    
    qx = np.empty((p,n))     
    M = np.zeros((p,p))
    
    for b in xrange(num_blocks):
        blk_start = b*blk
        blk_end = np.min([n, blk_start+blk])
        qxb = np.dot(Q.T , f(X[:, blk_start:blk_end]) )
        M += np.dot( qxb , qxb.T )
        qx[:,blk_start:blk_end] = qxb
    
    Um, S, x = spla.svd(M, full_matrices=False)
    
    if k < Um.shape[1]:
        Um = Um[:,:k]
        S = S[:k]
    
    U = np.dot(Q , Um)
    S = np.sqrt(S)
    
    UX = np.dot(Um.T , qx)
    return U, S, UX
    
def khatri_rao(A, B):
    '''
    Given a matrix A (d1xn) and B (d2xn) outputs a matrix C ((d1*d2)xn) of
    Khatri-Rao product
    '''
    d1,n = A.shape
    d2,n = B.shape        
    C = A.reshape((1,d1,n),order='F') * B.reshape((d2,1,n),order='F') 
    return C.reshape((d1*d2,n),order='F')
    
def khatri_rao_rowwise(A, B):
    '''
    Given a matrix A (nxd1) and B (nxd2) outputs a matrix C (nx(d1*d2)) of
    Khatri-Rao product
    '''    
    return khatri_rao(A.T, B.T).T 
 
def reg_rdivide(A,B,eps):
    '''
    Computes (A (B + eps * I)^{-1})
    NOTE: This function changes B
    '''
    d = B.shape[0]
    B.ravel()[::d+1] += eps    
    return spla.solve(B.T, A.T).T

def reg_rdivide_nopsd(A,B,eps):
    '''
    Computes (AB (B^2 + eps * I)^{-1})
    This assumes B to be symmetric but not necessarily positive semi-definite.
    This version is more robust to negative eigenvalues.
    '''
    return reg_rdivide(np.dot(A,B), np.dot(B.T,B), eps)
 
def reg_rdivide_nopsd_cg(A,b,eps,maxiter):
    b = np.dot(b, A)
    d = A.shape[0]
    AA = np.dot(A.T,A)
    AA.ravel()[::d+1] += eps                    
    x,_ = cg(AA, b, maxiter=maxiter)
    return x
      
if __name__ == '__main__':
    np.random.seed(0)    
    
    def test_khatri_rao():
        A = np.random.randn(2,10)
        B = np.random.randn(3,10)
        
        C = khatri_rao(A,B)
        
        for i in xrange(2):
            for j in xrange(3):
                for k in xrange(10):
                    assert C[i*3+j,k] == A[i,k] * B[j,k]

    def test_reg_rdivide():
        A = np.random.rand(10,5)    
        B = np.random.rand(5,5)
        B = B + B.T            
        C = B.copy() + np.eye(5) * 0.1
            
        C1 = reg_rdivide(A,B,0.1)
        C2 = np.dot(A, spla.inv(C))
        assert np.max(np.abs(C1-C2)) < 1e-10
      
    def test_blk_functions():
        # blk_fn_row
        A = np.random.rand(1101,30)
        B = np.random.rand(30,40)
        C1 = blk_fn_row(lambda s,e: A[s:e,:].dot(B), 1101, 200)
        C2 = np.dot(A,B)
        assert np.max(np.abs(C1-C2)) < 1e-10
        
        # blk_dot
        A = np.random.rand(50,1101)
        B = np.random.rand(1101,30)
        C1 = blk_dot(lambda s,e: A[:,s:e], lambda s,e: B[s:e,:], 1101)
        C2 = np.dot(A,B)
        assert np.max(np.abs(C1-C2)) < 1e-10
          
    test_khatri_rao()
    test_reg_rdivide()
    test_blk_functions()
                


    
