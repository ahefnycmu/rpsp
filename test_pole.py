# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:07:25 2016

@author: ahefny
"""

from environments import *
from policies import *
from models import *

#env = GymEnvironment('CartPole-v0')
env = PartiallyObservableEnvironment(GymEnvironment('CartPole-v0'), np.array([0,2]))
#env = GymEnvironment('MountainCar-v0')
o_dim,_ = env.dimensions
a_dim = env.action_info[0]
model = ObservableModel(o_dim)
policy = RandomDiscretePolicy(a_dim)    

trajs = env.run(model, policy, 1, 1000, render=True)

env.close()
