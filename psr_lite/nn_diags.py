#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:56:28 2017

@author: ahefny
"""
from __future__ import print_function
import numpy as np
import globalconfig

from psr_lite.utils.nn import CallbackOp

class PredictionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
def dbg_nn_skip_update(X, condition, msg):
    def fn(x):
        if not condition(x):
            print ('Skip update: '+msg, sep='')
            raise PredictionError(0)
    return CallbackOp(fn)(X)

def dbg_nn_raise_PredictionError(X, msg):
    def fn(x):
        errors = np.sum(np.abs(np.sum(x,axis=1))<1e-6)
        #print ('OUT: '+msg, errors, sep='')
        if errors > 10:
            print ('all zeros Error! Skip update: '+msg, errors, sep='')
            raise PredictionError(errors)
    return CallbackOp(fn)(X)

def dbg_raise_BadPrediction(X, msg):
    def fn(x):
        #print ('pred cost: ',x)
        if x > globalconfig.vars.args.dbg_prederror:
            print (msg+' Skip update. high pred cost (>%f)'%globalconfig.vars.args.dbg_prederror, x, sep='')
            raise PredictionError(-1)
    return CallbackOp(fn)(X)
