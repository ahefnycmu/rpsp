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
from IPython import embed
from psr_models.utils.utils import read_matrix


def eigs(A, dim=-1, order='LM'):
    values, vec = spalg.eig(A)
    #order by eigenvalues   
    if order=='SM':
        idv = np.argsort(values)
        values = values[idv]
        vec = vec[:,idv]
    if dim==-1:
        return vec, values
    else:
        return vec[:,:dim], values[:dim]
    
def eigsh(A, dim=-1, order='LM'):
    values, vec = spalg.eigh(A)
    #order by eigenvalues   
    if order=='SM':
        idv = np.argsort(values)
        values = values[idv]
        vec = vec[:,idv]
    if dim==-1:
        return vec, values
    else:
        return vec[:,:dim], values[:dim]
    
def svds(A, dim=-1, order='LM'):
    U, s, Vh = spalg.svd(A, full_matrices=False)
    U = U[:,:dim]
    s = s[:dim]
    Vh = Vh[:dim,:]
    print ('cond ', np.linalg.cond(A))
    print('Reconstruction error ', np.linalg.norm(A-np.dot(U,np.einsum('i,ij->ij',s,Vh) )))
    return U,s,Vh
    
def orth(A, scale = 1):# 4
    """
    Construct an orthonormal basis for the range of A using SVD
    A : (M, N) returns Q : (M, K) 
        Orthonormal basis for the range of A.
        scale: how much scaling on the eps for tolerance
        K = effective rank of A, as determined by automatic cutoff
    See also
    --------
    svd : Singular value decomposition of a matrix
    """
    assert not np.isnan(A).any(), embed()
    u, s, vh = spalg.svd(A, full_matrices=False)
    M, N = A.shape
    eps = numpy.finfo(float).eps*scale
    tol = max(M,N) * numpy.amax(s) * eps
    num = numpy.sum(s > tol, dtype=int)
    Q = u[:,:num]
    return Q

def regdivide(Y,X, reg):
    ''' do Y* inv(X + reg)'''
    assert X.shape[1] == Y.shape[1],'wrong regdivide dim.'
    print('REGDIVIDE IN USE')
    embed()
    dX = np.diag_indices(X.shape[1])
    Xinv = np.copy(X)
    Xinv[dX] = Xinv[dX] + reg
    Xinv = np.linalg.pinv(Xinv)
    Z = np.dot(Y, Xinv)
    return Z

def reg_rdivide(Y,X,reg):
    ''' do Y* inv(X + reg)'''
    d = X.shape[0]
    #X.ravel()[::d+1] += reg
    dX = np.diag_indices(X.shape[1])
    Xinv = np.copy(X)
    Xinv[dX] = Xinv[dX] + reg
    return spalg.solve(Xinv.T, Y.T).T
        

def dataProjection(X, r=0, type='NormalProj'):
    '''build projection matrix of X DxN onto span of X: Xp of Nxr'''
    print 'Project data'
    D = X.shape[0]
    if r == 0:
        Xp = X
    elif type=='PCA':
        U,S,V = np.linalg.svd(X.T,full_matrices=False, compute_uv=True)
        Xp = U[:,:r].T 
    elif type=='NormalProj':
        PrN = np.random.normal(0.0,1.0/float(np.sqrt(r)),(r,D))
        Xp = np.dot(PrN , X)
    return Xp
 


def svd_f(X, k, f=None, p=None, batch=None, compute_ufx=True):
    '''SVD_F Returns top-k left singular vectors of Y (Y(:,t) = f(X,t))
    X  (dimxN) 
    Y=f(X) dxN
    '''
    if f==None:
        f =( lambda x: x)
        compute_ufx = False
        Ufx=None
    if p==None:
        p = k
    if batch==None:
        batch = p
    assert(p >= k), 'p >= k!'
    
    N = X.shape[1]
    d = f(X[:,0]).shape[0]
    
    U = np.zeros((d,p), dtype=float)
    S = np.zeros((p+batch,p+batch), dtype=float)
    num_batches = int(np.ceil(N/float(batch)))
    phi = np.zeros((d,batch), dtype=float)
    
    for i in xrange(num_batches):
        batch_start = i*batch 
        batch_end = batch_start + batch
        if batch_end > N:
            batch_end = N
        
        UC = np.hstack([ U, np.zeros((d,batch), dtype=float)])
        
        for j in xrange(batch_start,batch_end,1):
            jj = j - batch_start
            phi[:,jj] = f(X[:,j])    
            UCsub = UC[:,:p+jj]
            Cj = phi[:,jj] - np.dot(UCsub , np.dot(UCsub.T , phi[:,jj]))
            
            if np.max(abs(Cj)) < 1e-7:
                Cj[:] = 0
            else:
                Cj = Cj /float( np.linalg.norm(Cj))
                #Nomalization can magnify numerical errors that make Cj not
                #orthogonal to U
               
                Cj = Cj - np.dot(UCsub , np.dot(UCsub.T, Cj))
                Cj = Cj /float( np.linalg.norm(Cj))
            UC[:,p+jj] = Cj

        phi[:,jj+1:] = 0
        C = UC[:,p:]   
        O = spalg.orth(C)
        C[:,:O.shape[1]] = O   
        C[:,O.shape[1]:] = 0
        
        P = np.vstack([np.dot(C.T , phi), np.dot(U.T , phi)])
        K = S + np.dot(P, P.T)
    
        Ui, S = eigs(K, order='SM')
        Ui = np.real(Ui) #should be real K is symmetric
        S = np.real(S)
        
        S = np.diag(S)
        
        x = np.arange(p+batch)
        S[x[:batch],x[:batch]] = 0
        
        V = np.hstack([C, U])
        U = np.dot(V , Ui[:,batch:])        
        #U = UU
        #print(i)
        #print( np.dot(U.T , U) - np.diag(np.diag(np.dot(U.T , U))) )
        #O = orth(U)
        #U[:,:O.shape[1]] = O    
        #U[:,O.shape[1]:] = 0
        #print('%dUU '%i, sum(UU))
    U = np.fliplr(U[:,-k:])
    

    U = np.real(U)
    
    if compute_ufx:
        Ufx = np.zeros((k, N), dtype=float)
        for i in xrange(N):
            Ufx[:,i] = np.dot(U.T,f(X[:,i]))
        
    return U, Ufx 


# def rand_svd_f(X, f=None, k=10, n=None, compute_ufx=True, it=2, slack=0, blk=1000):
#     '''%RAND_SVD_F Computes randomized left singular vectors of a dxn matrix X represented by a column sampling function f. 
#     %The method uses randomized SVD algorithm by Halko, Martinson and Tropp.
#     %Parameters:
#     % f - Column sampling function: f(s,e) should return columns s through e of the input matrix.
#     % n - Number of columns of the input matrix.
#     % k - Maximum number of singular vectors.
#     % it - (optional, default=2) The exponent q to premultiply (XX')^q by X to suppress small eigen values.
#     % slack - (optional, default=0) Extra dimensions to use in intermediate steps.
#     % blk (optional, default=1000) - Number of input columns that can be stored in memory.
#     % Outputs:
#     % U - Singular vectors.
#     % S - Singular values.
#     % UX - Projected input matrix U' * X
#     
#     %k = randi(20)+2;
#     '''
#     if n is None:
#         n = X.shape[1]
#     x = f(X[:,:1])
#     assert not np.isnan(x).any(), embed()
#     d = x.shape[0]
#     if d <= k:
#         # No need for SVD
#         U = np.eye(d)
#         S = np.ones((1,d)) # Dummy values
#         UX = f(X[:,:n])
#         return U, S, UX
#     
#     p = k + slack
#     num_blocks = int(np.ceil(n/float(blk)) )
#     
#     K = np.zeros((d,p))
#     Pb = np.random.normal(size=(n,p))
#     for b in xrange(num_blocks):
#         blk_start = b*blk
#         blk_end = np.min([n, blk_start+blk])
#         Xb = f(X[:,blk_start:blk_end])
#         #Pb = randn(size(Xb,2),p)
#         K = K + np.dot(Xb , Pb[blk_start:blk_end,:] )        
#         
#     for i in xrange(it):
#         KK = np.zeros((d,p))
#         for b in xrange(num_blocks):
#             blk_start = b*blk
#             blk_end = np.min([n, blk_start+blk])
#             Xb = f(X[:,blk_start:blk_end])
#             KK = KK + np.dot( Xb , np.dot(Xb.T , K) )
#         
#         K = KK / float( np.max(np.abs(KK[:])) )
#     
#     Q = orth(K, scale=1)
#     p = Q.shape[1]
#     
#     qx = np.zeros((p,n)) 
#     
#     M = np.zeros((p,p))
#     
#     for b in xrange(num_blocks):
#         blk_start = b*blk
#         blk_end = np.min([n, blk_start+blk])
#         qxb = np.dot(Q.T , f(X[:, blk_start:blk_end]) )
#         M = M + np.dot( qxb , qxb.T )
#         qx[:,blk_start:blk_end] = qxb
#     
#     Um, S, x = spalg.svd(M, full_matrices=False)
#     
#     if k < Um.shape[1]:
#         Um = Um[:,:k]
#         S = S[:k]
#     
#     U = np.dot(Q , Um)
#     S = np.sqrt(S)
#     
#     UX = np.dot(Um.T , qx)
#     return U, S, UX



def rand_svd_f(X, f=None, k=10, compute_ufx=True, it=2, slack=0, blk=1000, return_V=False):
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
    Pb = np.random.normal(size=(n,p))
    
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
    
    Um, S, Vh = spalg.svd(M, full_matrices=False)
    
    if k < Um.shape[1]:
        Um = Um[:,:k]
        S = S[:k]
        Vh = Vh[:k,:]
    
    U = np.dot(Q , Um)
    S = np.sqrt(S)
    
    UX = np.dot(Um.T , qx)
    if return_V:
        return U, S, UX, Vh
    return U, S, UX



def khatri_dot(A, B):
    ''' A d1xn B d2xn outputs a martix (d1xd2) x n'''
    C = np.asarray([np.einsum('i,j->ij',a,b) for (a,b) in zip(A.T,B.T)]).reshape((A.shape[1],A.shape[0]*B.shape[0])).T
    return C

def fast_pdist(X):
    n = X.shape[1]
    if n==1:
        return [0]
    XX = np.concatenate([[(X*X).sum(0)]*n])
    D = XX + XX.T - 2* np.dot(X.T,X)
    i = np.arange(0,n*n)
    j = np.floor(i/float(n)) < np.mod(i,n)
    D = D.reshape(-1)
    D0 = D[j]   
    return D0

def median_bw(X, max=0):
    n = X.shape[1]
    if max==0:
        max = n
    if n>max:
        idx = np.random.choice(n,max,False)
        X = X[:,idx]
    D = fast_pdist(X)
    med = np.sqrt(np.median(D))
    return med


def test_dataProjection(argv):
    D = 40; #Dimension of ambient space
    n = 2; #Number of subspaces
    d1 = 1; d2 = 1; #d1 and d2: dimension of subspace 1 and 2
    N1 = 20; N2 = 20; #N1 and N2: number of points in subspace 1 and 2
    if len(argv)>0:
        filename=argv[0]
        X = read_matrix(filename, name=argv[1], delim=' ')
    else:
        X1 = np.dot( np.random.randn(D,d1) , np.random.randn(d1,N1) ); #Generating N1 points in a d1 dim. subspace
        X2 = np.dot( np.random.randn(D,d2) , np.random.randn(d2,N2) ); #Generating N2 points in a d2 dim. subspace
        X = np.concatenate((X1,X2),axis=1);
    s = np.concatenate((np.zeros((1,N1),dtype=float), np.ones((1,N2),dtype=float)),axis=1); #Generating the ground-truth for evaluating clustering results
    r = 0;#np.max([d1,d2])*2 #0; #Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
    #project data to lower dimensional Space of X.T
    Xp = dataProjection(X,r,'PCA')
    
    #if True:
    #    plot_PCs(X,s)
    return

def test_svd_f(argv):
    # Test 1: Low rank data
    print('Test 1')
    d = 50000
    N = 500
    k = 25
    p = 50
#     Z=[]
#     f = open('examples/psr/utils/Z.txt','r')
#     for line in f.readlines():
#         line.rstrip()
#         toks = line.split(',')
#         toks = [float(tok) for tok in toks]
#         Z.append(toks)
#     Z = np.asarray(Z)
    X = np.dot(np.random.randn(d,p/2) , np.random.randn(p/2,N) ) # Some low rank data
    Y = np.random.randn(d,p/2) # other directions that appear only in the beginning
    XX=[Y,X]  
    Z = np.hstack([Y,X])
    Z = Z + np.random.randn(Z.shape[0],Z.shape[1]) * 0.1 # some noise dressing for the taste
     
    tic=time.time()
    U1,s1,Vh1 = svds(Z, k)
    print('svds took',time.time()-tic)
    tic=time.time()
    #U2, Ufx = svd_f(Z, k, None, p, 10)
    U2, x, Ufx = rand_svd_f(Z, f=(lambda x: x), k=k, it=1, slack=p, blk=1000)
    print('svdf took',time.time()-tic)
    #U1 and U2 should have the same columns up to sign changes
    diff = np.eye(k,k,0,dtype=float) - np.abs(np.dot(U1.T , U2))
    assert(np.abs(diff).max() < 1e-6), 'U large error'
    # Test 2: Non linear f
    print ('Test 2')
    f = lambda x: np.concatenate([np.cos(x), np.sin(x)], axis=0)
    d = 5
    k = 5
    p = 10
    X = np.random.randn(d,N)
    Z = f(X)
    tic=time.time()
    U1,s1,Vh1 = svds(Z, k)
    print('svds took',time.time()-tic)
    tic=time.time()
    #U2, Ufx = svd_f(X, k, f, p, 10)
    U2, x, Ufx = rand_svd_f(X, f=f, n=N,k=k, it=1, slack=p, blk=1000)
    print('svdf took',time.time()-tic)
    diff = np.eye(k,k,0,dtype=float) - np.abs(np.dot(U1.T , U2))
    assert(np.abs(diff).max() < 1e-6), embed()#'U large error'
    assert(np.abs(Ufx - np.dot(U2.T,Z)).max() < 1e-5), 'Ufx large error'

    return

def test_khatri_dot():
    n=3
    x1=100
    x2=50
    A = np.random.randn(x1,n)
    B = np.random.randn(x2,n)
    C = [np.einsum('i,j->ij', A[:,k],B[:,k]) for k in xrange(n)]
    D = khatri_dot(A, B)
    embed()
    err = [np.linalg.norm(C[k]-D[:,k].reshape((x1,x2), order='F')) for k in xrange(n)]
    merr = np.mean(err)
    serr = np.std(err)
    print('mean error %f +- std: %f'%(merr,serr))
    return
    


if __name__=='__main__':
    #test_dataProjection(sys.argv[1:])
    test_svd_f(sys.argv[1:])