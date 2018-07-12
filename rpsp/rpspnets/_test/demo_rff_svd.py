# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:29:40 2016

@author: ahefny
"""

import time
import numpy as np
import scipy.linalg as spla
import utils.kernel as krn
import utils.linalg as ula
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.misc import allow_default_rand

rnd = np.random

allow_default_rand(True)

N = 1000
d = 50
D = 300

rnd.seed(0)
start = time.time()

# Sample N points from d dimensional spherical shell
X = rnd.randn(N,d)
X = X / np.sqrt(np.sum(X * X, 0))
r = rnd.rand(N,1) * 0.9 + 0.1
X *= r

# Embed in a D dimensional space
U = spla.orth(rnd.randn(D,d))
X = np.dot(X,U.T)

# Compute exact Gram matrix
s = krn.median_bandwidth(X, N)
G0 = krn.gram_matrix_rbf(X, X, s)

# Use Nystrom features
K = 1000
gram = lambda X,Y: krn.gram_matrix_rbf(X,Y,s)
f_nst = krn.nystrom(X, K, gram)
Y = f_nst(X)
G1 = np.dot(Y, Y.T)
rerr = spla.norm(G1-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for Nystrom = %f\n' % rerr)

# Use Random Fourier Features
K = 50000
W = krn.sample_rff(K,D,s)
Y = krn.rff(X,W)
G1 = np.dot(Y, Y.T)
rerr = spla.norm(G1-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for RFF = %f\n' % rerr)

# Use Orthogonal Random Fourier Features
K = 50000
W_orth = krn.sample_rff(K,D,s,orth=True)
Y = krn.rff(X,W_orth)
G1 = np.dot(Y, Y.T)
rerr = spla.norm(G1-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for OrthRFF = %f\n' % rerr)

# Use Nystrom features with randomized SVD
p = 200
f = lambda X: f_nst(X.T).T
_,_,Y = ula.rand_svd_f(X.T, f, p)
G2 = np.dot(Y.T, Y)
rerr = spla.norm(G2-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for Nystrom+PCA = %f\n' % rerr)

# Use random fourier features with randomized SVD
p = 200
f = lambda X: krn.rff(X.T,W).T
_,_,Y = ula.rand_svd_f(X.T, f, p)
G2 = np.dot(Y.T, Y)
rerr = spla.norm(G2-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for RFF+PCA = %f\n' % rerr)

# Use orthogonal random fourier features with randomized SVD
p = 200
f = lambda X: krn.rff(X.T,W_orth).T
_,_,Y = ula.rand_svd_f(X.T, f, p)
G2 = np.dot(Y.T, Y)
rerr = spla.norm(G2-G0, 'fro') / spla.norm(G0, 'fro')
print('Relative error for OrthRFF+PCA = %f\n' % rerr)

print(time.time()-start)
