#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:07:55 2017

@author: ahefny, zmarinho
"""

class ExplorationStrategy(object):
    def __init__(self, base_policy):
        self._base_policy = base_policy
        
    def sample_action(self, t, observation, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
