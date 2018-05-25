#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:02:07 2017

@author: ahefny
"""

import OpenGL, OpenGL.GL
import OpenGL.GLUT as GLUT
import numpy as np
import pydart2 as pydart #if import twice no warning
from pydart2.gui.glut.window import GLUTWindow

import numpy as np
import environments
import models
import policies

from IPython import embed


class _Controller:
    """ Add damping force to the skeleton """
    def __init__(self, skel, act_idx):
        self.skel = skel
        self._noise = 0.1
        self._act_idx = act_idx

    def compute(self):
        force = 0.0 * self.skel.q
        force[self._act_idx] = self.act #+ np.random.normal(scale=self._noise,size=self.act.shape)
        #print self.act, self.skel.q
        return force

pydart.init(verbose=True)

glut_loop_fn = GLUT.glutMainLoopEvent if bool(GLUT.glutMainLoopEvent) else GLUT.glutCheckLoop

  
def idle_fn():
    pass
  
class DartRenderer():
    def start(self, world): 
        win = GLUTWindow(world, 'Rendering')
        
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_MULTISAMPLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(*win.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        
        win.window = GLUT.glutCreateWindow(win.title)
        OpenGL.GL.glClearColor(0.,0.,0.,1.)
        GLUT.glutDisplayFunc(win.drawGL) 
        GLUT.glutIdleFunc(idle_fn)
        win.initGL(*win.window_size)
        
        self._win = win


        
    def step(self):    
        self._win.drawGL()
        glut_loop_fn()
        #self._cam = self._win.scene.cameras[0] 
        #embed()
        #GLUT.glutKeyboardFunc(pydart.guikeyPressed)
        #self._win.is_animating = True
        #print self._win.scene.tb
        #self._win.scene.cameras #there are 2 cameras
        return
        
    def end(self):
        GLUT.glutHideWindow(self._win.window)

class DartEnvironment(environments.Environment):
    def __init__(self, file_name, act_idx):
        self._world = pydart.World(1e-2, file_name)
        skel = self._world.skeletons[0]
        self._controller = _Controller(skel, act_idx)
        skel.controller = self._controller
        self._render = False
        self._renderer = DartRenderer()        
        self._act_idx = act_idx         
        self._render_ready = False

         
    def reset(self):        
        self._world.reset()                    
        return self._reset()
        
    def _reset(self):
        pass
        
    def render(self):
        if not self._render_ready:
            self._renderer.start(self._world)
            self._render_ready = True
            
        self._renderer.step()
             
    def _ensure_state_lims(self):
        state = np.copy(self._controller.skel.q)
        np.clip(state, self._controller.skel.q_lower, self._controller.skel.q_upper, state)
        full_state = list(state) + list(self._controller.skel.dq)
        self._set_state(full_state)
        return
        
        
    def step(self, a):  
        self._controller.act = a
        self._world.step()
        self._ensure_state_lims()
        if self._render:
            self._renderer.step()
        res = self._step(a)
        return res
    
    def _step(self, a):
        raise NotImplementedError
        
    def close(self):
        if self._render_ready:
            self._renderer.end()
    
    @property
    def state(self):
        return self._world.x 
    
    def _set_state(self, value):
        self._world.skeletons[0].set_states(value)
        return
    
    @property
    def dimensions(self):
        return (self._world.skeletons[0].x.shape[0],len(self._controller._act_idx)) 


class DartCartPole(DartEnvironment):
    def __init__(self):
        DartEnvironment.__init__(self, 'dart_cartpole.xml', [0]) #actuator on joint 0
        self._theta_threshold_radians = 0.209   
        return 
            
    def _reset(self):                
        self._world.skeletons[0].q = (0.0, 0.1)
        return self._world.skeletons[0].x
      
    def _step(self, a):
        obs = np.array(list(self._world.skeletons[0].q) + list(self._world.skeletons[0].dq))
        if np.abs(obs[1]) <  self._theta_threshold_radians: 
            return obs,1,False
        else:
            return obs,0,True
    
        
        
class DartSwimmer(DartEnvironment):
    def __init__(self):
        DartEnvironment.__init__(self, 'dart_swimmer.xml', [3,4])
        self.ctrl_cost_coeff = 5e-3 #1.0
        return
        
     
    def _reset(self):
        state0 = np.array([0.0, 0.0, 0.0,  0.15, 0.15,
                           0.0, 0.0, 0.0,  0.0, 0.0])#floor, x,y,theta_world, theta1,theta2
        self._set_state(state0)
        return state0
    
    def _step(self, a):
        obs = np.array(list(self._world.skeletons[0].q) + list(self._world.skeletons[0].dq))
       
        reward_fwd = np.linalg.norm(self._world.skeletons[0].dq[1]) #only in z direction
        reward_ctrl = - self.ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        #print 'total rwd:%.3f, fwd %.3f, ctrl %.3f'%(reward,reward_fwd,reward_ctrl)
        return obs, reward, False #, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

            
if __name__ == '__main__':           
    #env = DartCartPole()
    env = DartSwimmer()
    
    t = env.run(models.ObservableModel(env.dimensions[0]), 
                policies.RandomGaussianPolicy(env.dimensions[1]),
                1, 1000, render=True)
    print t[0].rewards.sum()
    
    #world = pydart.World(1e-2, 'dart_pendulum.xml')
    #skel = world.skeletons[0]

    #skel.q = (0.0, 0.1)
    #print('init pose = %s' % skel.q)
    #skel.controller = Controller(skel)
    #            
    #renderer = DartRenderer()
    #renderer.start(world)
    #while world.t < 3.0:    
    #    world.step()    
    #    renderer.step()    
    #    
    #renderer.end()    
    print 'Done' 

