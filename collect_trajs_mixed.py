#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 21:29:23 2017

@author: ahefny
"""

from argparse import ArgumentParser
from models import Trajectory
from input_device import InputDevice
from input_policy import InputPolicy
from manual_testbed import create_manual_environment
import numpy as np

def _append_traj(traj_list, min_traj_length, o0, obs, rwd, act):
    if len(obs) > min_traj_length:                        
        obs_arr = np.array(obs)
        act_arr = np.array(act)
        rwd_arr = np.array(rwd)
        
        traj = Trajectory(obs=obs_arr,states=obs_arr.copy(),act=act_arr,rewards=rwd_arr,obs0=o0,state0=o0.copy())
        traj_list.append(traj)
        
        print 'Collected %d samples [%d total]' % (len(obs), sum(t.length for t in traj_list))
    else:
        print 'Trajectory is too short'
    
    obs[:] = []
    act[:] = []
    rwd[:] = []

def collect_trajs_mixed_policy(env, min_traj_length, expert_policy, exploration_policy, control_pred):            
    traj_list = []
    exploration_mode = False        
    done_all = False
    done = True
    o = None
    o0 = None
 
    obs = []
    act = []
    rwd = []

    context = {}
    
    while not done_all:    
        switch, done_all = control_pred(context)        
        
        if done or done_all:
            if exploration_mode:
                _append_traj(traj_list, min_traj_length, o0, obs, rwd, act)            
                
            o0 = env.reset()
            o = o0
            exploration_policy.reset()

        policy = exploration_policy if exploration_mode else expert_policy                    
        a,p,info = policy.sample_action(o)    
        o,r,done = env.step(a)
        env.render()
                                        
        if exploration_mode:
            obs.append(o)
            act.append(a)
            rwd.append(r)                        
        
        if switch:
            if exploration_mode:
                _append_traj(traj_list, min_traj_length, o0, obs, rwd, act)                
                print 'Switching to expert policy'                
            else:
                o0 = o
                exploration_policy.reset()
                print 'Switching to exploration policy'
            
            exploration_mode = not exploration_mode
            
    return traj_list
    
if __name__ == '__main__':
    from policies import RandomGaussianPolicy    
    import sys
    import cPickle as pickle
    
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, required=False, default='/dev/input/js0')
    parser.add_argument('--reset_btn', type=int, required=False, default=0, help='Button used to reset environment')
    parser.add_argument('--switch_btn', type=int, required=False, default=1, help='Button used to switch policy')
    parser.add_argument('--stop_btn', type=int, required=False, default=3, help='Button used to end session')
    
    parser.add_argument('--env', type=str, required=False, default='swimmer')
    parser.add_argument('--save_dir', type=str, required=False, default='.')
    parser.add_argument('--save_file', type=str, required=False)
    
    args = parser.parse_args()
    
    rng = np.random
    env, expert_policy = create_manual_environment(args.env, rng, args.device, args.reset_btn)
    exploration_policy = RandomGaussianPolicy(env.dimensions[1], rng)
    
    switch_device = InputDevice(args.device)
    switch_device.open()
    
    def control_pred(context):
        switch_device.poll()
        return (switch_device.is_button_pressed(args.switch_btn),
                switch_device.is_button_pressed(args.stop_btn))
                    
    trajs = collect_trajs_mixed_policy(env, 100, expert_policy,
                                       exploration_policy, control_pred)
        
    while args.save_file is None:
        print 'Collected %d samples. Save (y/n)?' % (sum(t.length for t in trajs))
        c = sys.stdin.readline().strip()
        
        if c == 'y' or c == 'Y':
            print ('Save to %s/' % args.save_dir)
            args.save_file = sys.stdin.readline().strip()
        elif c == 'n' or c == 'N':
            break
        
    if args.save_file is not None:
        s = pickle.dumps(trajs, protocol=2)
        f = open('%s/%s' % (args.save_dir, args.save_file), 'w')
        f.write(s)
        f.close()
                
        
    