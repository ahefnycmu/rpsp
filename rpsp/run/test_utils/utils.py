from __future__ import print_function
import matplotlib
import numpy as np

matplotlib.use('Agg')
from distutils.dir_util import mkpath
import time

def render_trajectory(env, env_states, render=True, speedup=0.1):
    # from fully observable state of environment set states  no monitor video available
    env.set_state(env_states[0][0], env_states[0][1])
    for j in range(len(env_states)-1):
        if render:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)   
        env.set_state(env_states[j+1][0], env_states[j+1][1])   
    return 

def replay_trajectory(env, traj, render=True, speedup=5):
    # from trajectory random state and actions generate simulated observations can save video with monitor
     
    try:
        env.rng().set_state(traj.rng)
    except AttributeError:
        env._orig_env.np_random.set_state(traj.rng)
    o = env.reset()
    R = 0
    #print ('o0', o, traj.obs0)
    for j in range(len(traj.act)):
        if render:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
        try:   
            o,r,done = env.step(traj.act[j])
            #print ('r', np.isclose(r,traj.rewards[j,0]))
            #print('obs',np.isclose(o, traj.obs[j,:]).all())
        except ValueError:
            o,r,done,_ = env.step(traj.act[j])
        R+=r
        if done: break
    print('total reward of trajectory is:',R)
    return R
    
def play_video_trajs(fdir, fname, save=False, play=True, niter=-1):
    ''' play video for last iteration only for all trajectories'''
    from rpsp.run.stats_test import load_model
    from rpsp.filters.models import Trajectory
    from rpsp.envs.load_environments import load_environment
    results, args = load_model(fname, fdir)
    args.file = 'videos/'+args.env+'/'
    mkpath(args.file)
    if save:
        args.monitor='video_trials'
        args.vrate=1
    else:
        args.monitor=None
    env,_,_ = load_environment(args)    #TODO: remove from envs not used
    
    build_traj=False
    for trial in range(len(results)):
        print ('Trial ', trial)
        try:
            trajs = results[trial]['trajs'][niter] #too big
            num_trajs = len(trajs)
        except KeyError:
            build_traj=True
            actions = results[trial]['act'][niter]
            rngs = results[trial]['rng'][niter]
            num_trajs = len(results[trial]['act'][niter])
            try:
                rwds = results[trial]['rewards'][niter]
            except KeyError:
                rwds = np.zeros((num_trajs,args.len))
            
        #last element is final learning trajectories play on
        R= 0
        for it in range(num_trajs):
            #render_trajectory(env,  traj.env_states, render=play)
            if build_traj:
                traj = Trajectory(act=actions[it], rng=rngs[it], rewards=rwds[it])
            else:
                traj = trajs[it]
            
            R += replay_trajectory(env, traj, render=play)
        print('Total batch reward',R/float(num_trajs))

def find_best_traj(results):        
    best_traj = (0,0,0,0)    
    for trial in range(len(results)):
        N=len(results[trial]['act']) #number of iterations kept for plot
        for iteration in range(N):
            #T = len(results[trial]['rewards'])
            #rwds = results[trial]['rewards'][T-N+iteration]
            rwds = [ np.sum(t) for t in results[trial]['rwd'][iteration]]
            #num_trajs = len(results[trial]['act'][iteration])
            it = np.argmax(rwds)
            max_rwd = np.max(rwds)
            if max_rwd > best_traj[-1]:
                best_traj=(trial,iteration,it,max_rwd)
                     
    return best_traj
   
def load_item(key, d, it, size=(0,0)):
    try:
        data = d[key][it]
    except KeyError:
        data = np.zeros(size)
    except IndexError:
        data = np.zeros(size)
    return data
            
def play_best_video_traj(fdir, fname, save=False, play=True, iter=-1):
    ''' play video for last iteration only for all trajectories'''
    from rpsp.run.stats_test import load_model
    from rpsp.filters.models import Trajectory
    from rpsp.envs.load_environments import load_environment
    results, args = load_model(fname, fdir)
    print('using rllab env?', args.use_rllab)
    args.file = fdir
    #args.use_rllab =True
    mkpath(args.file)
    if save:
        args.monitor='/video_trials/'
        args.vrate=1
    else:
        args.monitor=None
    env,_,_ = load_environment(args)    #TODO: remove from envs not used
    
    if fname.find('_t'):
        trial, niter, it, best_rwd = find_best_traj([results])
        results_trial = results
    else:
        trial, niter, it, best_rwd = find_best_traj(results)
        results_trial = results[trial]
    print (trial, niter, it, best_rwd)
    print ('Trial ', trial)
    try:
        traj = results_trial['trajs'][niter][it] #too big
        replay_trajectory(env, traj, render=play)
    except KeyError:
        actions = results_trial['act'][niter]
        rngs = results_trial['rng'][niter]
        num_trajs = len(results_trial['act'][niter])
        rwds = load_item('rwd', results_trial, niter, size=(num_trajs,args.len))
        obs = load_item('obs', results_trial, niter, size=(num_trajs,args.len))
        env_states = load_item('env_states', results_trial, niter, size=(num_trajs,args.len))
        print ('BEST trajectory')

        traj = Trajectory(act=actions[it], rng=rngs[it], rewards=rwds[it], env_states=env_states[it], obs=obs[it])
        
        replay_trajectory(env, traj, render=play)
        print ('BEST trial')
        R=0.0
        for i in range(num_trajs):
            #render_trajectory(env,  traj.env_states, render=play)
            traj = Trajectory(act=actions[i], rng=rngs[i], rewards=rwds[it])
            R+=replay_trajectory(env, traj, render=play)
        
        print('Total batch reward',R/float(num_trajs))
        
        

        

if __name__ == '__main__':
    import sys
    
    fdir=sys.argv[1] 
    #'results/test/replay4_hpsrFalse_ecay1.0_iter5_dateproj_lr0.01_ency0_dim10_nitN50_thodobsVR_nitS0_condkbr_eeze1_past10_pred1.0_licyFalse_reg0.01_ount1.0__actrelu_wpca0.0_rm_gTrue_amma0.99_h0True_jlen10__optadam_ency0_rajs10/'
    fname=sys.argv[2]
    #'obsVR.pkl'
    #play_video_trajs(fdir, fname)
    play_best_video_traj(fdir,fname)