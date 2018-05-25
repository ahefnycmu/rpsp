#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:43:48 2017

@author: ahefny
"""

import numpy as np
_allow_default_rand = [False]

def allow_default_rand(allow = True):
    _allow_default_rand[0] = allow_default_rand
    
def get_default_rand():
    if _allow_default_rand[0]: 
        return np.random 
    else: 
        raise Exception("Cannot use default random number generator. Must set 'rng' parameter or call 'utils.misc.allow_default_rand()'")
        