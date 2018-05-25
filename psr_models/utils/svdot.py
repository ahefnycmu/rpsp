# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:50:08 2016

@author: zmarinho
"""

import numpy as np
import scipy as sp
import scipy.sparse as ssp
from IPython import embed

def rbf_dot(sigma, A,  B=None):
    ''' A and B are dxN return a NxN matrix'''
    if B==None :
        dot_a = np.sum(A**2,0)
        dot_a = np.reshape(dot_a, (dot_a.shape[0],1))
        ones_a = np.ones((A.shape[1], 1), dtype=float)
        a_b = np.dot(2.0*sigma*A.T,A)
        x = np.einsum('ij,jk->ik',sigma*dot_a,ones_a.T)
        x = x + np.einsum('ij,jk->ik',sigma*ones_a, dot_a.T)
        x = a_b - x
        del a_b,dot_a,ones_a
    else:
        dot_a = np.sum(A**2,0)
        dot_b = np.sum(B**2,0)
        dot_a = np.reshape(dot_a, (dot_a.shape[0],1))
        dot_b = np.reshape(dot_b, (1,dot_b.shape[0]))
        ones_a = np.ones((A.shape[1], 1), dtype=float)
        ones_b = np.ones((1, B.shape[1]), dtype=float)
        a_b = np.dot(2.0*sigma*A.T,B)
        x = np.einsum('ij,jk->ik',sigma*dot_a,ones_b)
        x = x + np.einsum('ij,jk->ik',sigma*ones_a, dot_b)
        x = a_b - x
        
        del a_b,dot_a,ones_a,dot_b,ones_b
    #adding a constant feature
    x = np.exp(x)
    x = x + np.ones(x.shape,dtype=float)  #1.0/sigma*np.exp(x)
    return x 
    
def rbf_sparse_dot(sigma, A,  B=None):
    ''' A and B are dxN return a NxN matrix'''
    if B==None :
        A = A.tocsc()
        A.data = A.data**2
        dot_a = A.sum(0).T
        ones_a = np.ones((A.shape[1], 1), dtype=float)
        a_b = A.T.dot(A)
        x = sigma * (2 * a_b - np.outer(dot_a,ones_a.T) - np.outer(ones_a,dot_a.T))
    else:
        embed()
        A = A.tocsc()
        B = B.tocsc()
        A.data = A.data**2
        dot_a = A.sum(0)
        B.data = B.data**2
        dot_b = B.sum(0)
        dot_a = np.reshape(dot_a, (dot_a.shape[0],1))
        dot_b = np.reshape(dot_b, (1,dot_b.shape[0]))
        ones_a = np.ones((A.shape[1], 1), dtype=float)
        ones_b = np.ones((1, B.shape[1]), dtype=float)
        a_b = np.dot(A.T,B)
        dot_a = ssp.csc_matrix(dot_a)
        dot_b = ssp.csc_matrix(dot_b)
        ones_b = ssp.csc_matrix(ones_b)
        ones_a = ssp.csc_matrix(ones_a)
        embed()
        x = sigma * (2 * a_b - np.dot(dot_a,ones_b) - np.dot(ones_a, dot_b))
    #adding a constant feature
    ker = np.exp(x) + np.ones(x.shape,dtype=float)  #1.0/sigma*np.exp(x)
    return ker  
    
    
def testunit(args):
    sigma = args[1]
    print sigma
    A = np.random.randint(0,10,(3,6))
    B = np.random.randint(0,20,(3,8))
    k = rbf_dot(sigma, A,B)
    print A
    print B
    print k
    return
    
def __init__(args):
    testunit(args)
    
    return