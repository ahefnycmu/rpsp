#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:08:49 2017

@author: ahefny
"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

#with open('mse_swimmer_all_fixsvd.pkl', 'rb') as f: 
with open('mse_swimmer_all_mat.pkl', 'rb') as f: 
    a = pickle.loads(f.read())['mse']
    

data = np.array([x['rffpsr'] for x in a])    
m = np.mean(data, axis=0)
s = np.std(data, axis=0)
x = np.arange(1,11)
plt.plot(x, m, 'r--', lw=2.0, label='joint')
plt.fill_between(x, m+s, m-s, alpha=0.5, color='r')

data = np.array([x['rffpsr_cond'] for x in a])    
m = np.mean(data, axis=0)
s = np.std(data, axis=0)
x = np.arange(1,11)
plt.plot(x, m, 'b--', lw=2.0, label='cond')
plt.fill_between(x, m+s, m-s, alpha=0.3, color='b')

data = np.array([x['rffrnn'] for x in a])    
m = np.mean(data, axis=0)
s = np.std(data, axis=0)
x = np.arange(1,11)
plt.plot(x, m, 'r', lw=2.0, label='joint refined')
plt.fill_between(x, m+s, m-s, alpha=0.5, color='r')

data = np.array([x['rffrnn_cond'] for x in a])    
m = np.mean(data, axis=0)
s = np.std(data, axis=0)
x = np.arange(1,11)
plt.plot(x, m, 'b', lw=2.0, label='cond refined')
plt.fill_between(x, m+s, m-s, alpha=0.3, color='b')

data = np.array([x['rffarx'] for x in a])    
m = np.mean(data, axis=0)
s = np.std(data, axis=0)
x = np.arange(1,11)
plt.plot(x, m, 'k', lw=2.0, label='rffarx')
plt.fill_between(x, m+s, m-s, alpha=0.3, color='k')

plt.legend()
plt.grid()
plt.show()

