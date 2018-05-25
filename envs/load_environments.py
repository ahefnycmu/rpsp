#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:38:44 2017

@author: ahefny
"""
# import numpy as np
# import NN_policies
from envs.environments import *
from envs.imitation import *
from envs.simulators import *
# import psr_lite.psr_lite.rffpsr
# import psr_lite.psr_lite.feat_extractor
# import psr_lite.psr_lite.rffpsr_rnn
# import psr_lite.psrlite_policy
# from models import ObservableModel, FiniteHistoryModel, FiniteDeltaHistoryModel
# import policies
# import psr_lite.psr_lite_vrpg
# from policy_learn import learn_policy
from IPython import  embed
import gym
import globalconfig
#from test_utils.plot import call_plot
#from psr_lite.psr_lite.noisy_model import NoisyModel


def get_func(function_string):
    mod = load_env(function_string)
    mod_name, func_name = function_string.rsplit('.',1)
    func = getattr(mod, func_name)
    return func

def load_env(function_string):
    import importlib
    mod_name, func_name = function_string.rsplit('.',1)
    mod = importlib.import_module(mod_name)
    return mod
    
class structtype():
    pass

'''
Each create_env function takes command line arguments and creates 
environment, exploration model and exploration policy. The latter two are used
to generate initial trajectories.

A function can return None for model and policy. In this case an observable model
and a random Gaussian policy are used.
'''

def create_env_cartpole(args):    
    env = ContinuousEnvironment('CartPole-v0', CartpoleContinuousSimulator(), rng=args.rng)
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([0,2,1,3]))
    else:
        env = PartiallyObservableEnvironment(env, np.array([0,2]))
    #save videos
    if args.monitor is not None:        
        try:
            import gym.wrappers
            env.env = gym.wrappers.Monitor(env=env.env, directory=args.flname+args.monitor,\
                                   force=True, video_callable=lambda count: count%100==0)
        except Exception:
            env.monitor.start(args.flname+args.monitor, force=True) #call env.env if just GymEnvironment
            env.monitor.configure(video_callable=lambda count: count%int(2*args.numtrajs)==0) #1 per iteration   
    return env, None, None

def create_env_dartCartPole(args):
    import pydart2
    from dart.dart_environments import DartCartPole
    env = DartCartPole()
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([0,1,2,3]))
    else:
        env = PartiallyObservableEnvironment(env, np.array([0,1]))
        
    return env, None, None

def create_env_dartSwimmer(args):
    import pydart2
    from dart.dart_environments import DartSwimmer
    env = DartSwimmer()
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.arange(2,10))
    else:
        env = PartiallyObservableEnvironment(env, np.array([2,3,4]))
        
    return env, None, None

def create_env_RLSwimmer(args): #Not tested
    from psr_lite.datagen.Swimmer import Swimmer 
    env = RLEnvironment(Swimmer, args.rng)
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([2,3,6,7,8])) #2,3,6,7,8 position angles and velocities
    else:
        env = PartiallyObservableEnvironment( env, np.array([6,7,8]) ) #only positions of angles
    return env, None, None

def create_env_gym(args):
    env = ContinuousEnvironment(args.env, args.sim(), rng=args.rng)
    #save videos
    if args.monitor is not None:
        try:
            import gym.wrappers
            import glfw
            env.env = gym.wrappers.Monitor(env=env.env, directory=args.flname+args.monitor,\
                                   force=True, video_callable=lambda count: count%100==0)
        except Exception:
            env.monitor.start(args.flname+args.monitor, force=True) #call env.env if just GymEnvironment
            env.monitor.configure(video_callable=lambda count: count%int(args.numtrajs)==0) #1 per iteration
    return env, None, None

def create_env_MountainCar(args):
    env = ContinuousEnvironment(args.env, MountainCarContinuousSimulator(), rng=args.rng)
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([0,1]))
    else:
        env = PartiallyObservableEnvironment(env, np.array([0]))
    #save videos
    if args.monitor is not None:
        try:
            import gym.wrappers
            env.env = gym.wrappers.Monitor(env=env.env, directory=args.flname+args.monitor,\
                                   force=True, video_callable=lambda count: count%100==0)
        except Exception:
            env.monitor.start(args.flname+args.monitor, force=True) #call env.env if just GymEnvironment
            env.monitor.configure(video_callable=lambda count: count%int(args.numtrajs)==0) #1 per iteration
    return env, None, None


def create_env_Acrobot(args):
    env = ContinuousEnvironment(args.env, AcrobotContinuousSimulator(), rng=args.rng)
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([0,1,2,3]))
    else:
        env = PartiallyObservableEnvironment(env, np.array([0,1]))
    #save videos
    if args.monitor is not None:
        
        try:
            import gym.wrappers
            env.env = gym.wrappers.Monitor(env=env.env, directory=args.flname+args.monitor,\
                                   force=True, video_callable=lambda count: count%100==0)
        except Exception:
            env.monitor.start(args.flname+args.monitor,  force=True) #call env.env if just GymEnvironment
            env.monitor.configure(video_callable=lambda count: count%int(args.numtrajs)==0) #1 per iteration
    return env, None, None


def create_env_Pendulum(args):
    env = ContinuousEnvironment(args.env, PendulumContinuousSimulator(), rng=args.rng)
    #env = GymEnvironment('Pendulum-v0',discrete=False)
    if args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array([0,1]))
    else:
        env = PartiallyObservableEnvironment(env, np.array([0]))
    #save videos
    if args.monitor is not None:
        
        try:
            import gym.wrappers
            env.env = gym.wrappers.Monitor(env=env.env, directory=args.flname+args.monitor,\
                                   force=True, video_callable=lambda count: count%100==1, seed=args.rng)
        except Exception:
            env.monitor.start(args.flname+args.monitor, force=True) #call env.env if just GymEnvironment
            env.monitor.configure(video_callable=lambda count: count%int(args.vrate)==0) #1 per iteration
    return env, None, None

def monitor_env(args, env, env_name, entry_point):
    from gym.envs.registration import register 
    from gym.wrappers import Monitor
    
    mon_name = (env_name + '-trial%d-v1'%args.trial, env_name)[args.monitor is None]
    register(
    id=mon_name,
    entry_point=entry_point,
    tags={'wrapper_config.TimeLimit.max_episode_steps': args.len},
    reward_threshold=1e5,
    )
    print('monitor wrapper')
    
    monitor = Monitor(env.env, 
                      directory=args.flname+args.monitor,
                      force=True, video_callable=lambda it:it%int(args.vrate) == 0,
                      write_upon_reset=True) 
    
    env = GymEnvironment(monitor, discrete=False, rng=args.rng)     
    return env

def create_gym_env(args, env_name, entry_point, wrapper=None):
   
    if args.env.rsplit('-')[-1]=='v1':
        env = gym.make(env_name)
    elif args.env.rsplit('-')[-1]=='v2':
        #call_env = get_func(wrapper+'_rllab')
        #mod,f = wrapper.rsplit('.',1)
        #env = call_env(rng=args.rng)
        load_env(wrapper+'_rllab')
        mod,f = wrapper.rsplit('.',1)
        entry_point = mod+':'+f
        env = gym.make(env_name)
        #env.set_rng(args.rng)
    elif args.env.rsplit('-')[-1]=='v3':
        call_env = get_func(wrapper)
        mod,f = wrapper.rsplit('.',1)
        entry_point = mod+':'+f 
        print('v3',entry_point)
        env = call_env(rng=args.rng)
    else:
        raise "v1 for gym environment v2 for rllab env"
    
    env.seed(int(args.seed)+args.trial)
    if args.monitor is not None:
        print('using monitor env')
        env = monitor_env(args, env, env_name, entry_point)
    else:
        env = GymEnvironment(env, discrete=False, rng=args.rng)
    return env        

def create_partial_environment(args, env, part_obs, full_obs=None):
    if args.critic is not None:
        vf = load_value_function(args.critic + '/vstar.h5', args.critic + '/meanstd.npz')
        env = ImitiationEnvironment(env, vf)
    
    if not args.fullobs:
        env = PartiallyObservableEnvironment(env, np.array(part_obs))
    elif full_obs is not None:
        env = PartiallyObservableEnvironment(env, np.array(full_obs))
    return env


def create_gym_mujoco_env(args, env_name, entry_point, part_obs, full_obs=None, wrapper=None):
    env = create_gym_env(args, env_name, entry_point, wrapper=wrapper)                                              
    env = create_partial_environment(args, env, part_obs, full_obs=full_obs)
    return env


def create_env_swimmer(args):
    env = create_gym_mujoco_env(args, args.env, 'gym.envs.mujoco:SwimmerEnv', [0,1,2],  wrapper='envs.mujoco.swimmer_env.SwimmerEnv')                                              
    #model_exp = ObservableModel(obs_dim = env.dimensions[0])  
    #pi_exp = policies.RandomGaussianPolicy(env.dimensions[1])      
    return env, None, None #model_exp, pi_exp 

def create_env_hopper(args): 
    env = create_gym_mujoco_env(args, 'Hopper-v1', 'gym.envs.mujoco:HopperEnv', [0,1,2,3,4], wrapper='envs.mujoco.hopper_env.HopperEnv')                                     
    return env, None, None

def create_env_walker(args): 
    env = create_gym_mujoco_env(args, 'Walker2d-v1', 'gym.envs.mujoco:Walker2d', [0,1,2,3,4,5,6,7])                                     
    return env, None, None
        
def create_env_ant(args): 
    env = create_gym_mujoco_env(args, 'Ant-v1', 'gym.envs.mujoco:AntEnv', np.arange(8,16), np.arange(16), wrapper='envs.mujoco.ant_env.AntEnv')                                     
    return env, None, None    

def create_env_Mcartpole(args): 
    env = create_gym_mujoco_env(args, 'InvertedPendulum-v1', 'gym.envs.mujoco:InvertedPendulum', [0,1])                                     
    return env, None, None

def create_env_reacher(args): 
    env = create_gym_mujoco_env(args, 'Reacher-v1', 'gym.envs.mujoco:Reacher', [2,3,4,5],[2,3,4,5,6,7])                                     
    return env, None, None

def create_env_doublependulum(args): 
    env = create_gym_mujoco_env(args, 'InvertedDoublePendulum-v1', 'gym.envs.mujoco:InvertedDoublePendulum', [0,1,2], [0,1,2,5,6,7], wrapper='envs.mujoco.inverted_double_pendulum_env.InvertedDoublePendulumEnv')                                     
    return env, None, None
    
env_dict = {'Swimmer-v1' : create_env_swimmer,
            'Swimmer-v2' : create_env_swimmer,
            'Swimmer-v3' : create_env_swimmer,
            'Walker2d-v1' : create_env_walker,
            'Hopper-v1' : create_env_hopper,
            'Hopper-v2' : create_env_hopper,
            'Ant-v1' : create_env_ant,
            'Ant-v2' : create_env_ant,
            'CartPole-v0' : create_env_cartpole,
            'Swimmer-dart': create_env_dartSwimmer,
            'CartPole-dart': create_env_dartCartPole,
            'Swimmer-RL': create_env_RLSwimmer,
            'MountainCar-v0' : create_env_MountainCar,
            'Acrobot-v0' : create_env_Acrobot,
            'Acrobot-v1' : create_env_doublependulum,
            'Acrobot-v2' : create_env_doublependulum,
            'Pendulum-v0' : create_env_Pendulum,
            'CartPole-v1' : create_env_Mcartpole,
            'Reacher-v1' : create_env_reacher
            }

def load_environment(args):
    if args.use_rllab:
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        from rllab.envs.normalized_env import normalize
        from rllab.envs.gym_env import GymEnv
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        #from rllab.envs.occlusion_env import OcclusionEnv
        from rllab_scripts.occlusion_env import OcclusionEnv
        from gym.envs.registration import register 
        from gym.wrappers import Monitor
        envs = {'Swimmer-v1' : ('Swimmer-v1', [0,1,2], 'gym.envs.mujoco:SwimmerEnv', 5000),
            'Hopper-v1' : ('Hopper-v1', [0,1,2,3,4], 'gym.envs.mujoco:HopperEnv', 1000),
            'CartPole-v1' : ('InvertedPendulum-v1', [0,1], 'gym.envs.mujoco:InvertedPendulum', 200),
            'Walker2d-v1' : ('Walker2d-v1', [0,1,2,3,4,5,6,7],'gym.envs.mujoco:Walker2d',1000),
            } 
        env_info = envs[args.env]
        def setup_envs():
            register(
            id=args.env+'-cap-v1',
            entry_point=env_info[2],
            tags={'wrapper_config.TimeLimit.max_episode_steps': env_info[3]},
            reward_threshold=1e5,
            )
        #setup_envs()
        if args.monitor is not None:
            env = GymEnv(env_info[0], record_video=True, \
                         video_schedule=lambda it: it%int(args.vrate) == 0,\
                        log_dir=args.file+args.monitor+'/')
        else:
            env = GymEnv(env_info[0])
        
        env = OcclusionEnv(env, env_info[1])   
        model_exp = None
        pi_exp = None
    else:
        env, model_exp, pi_exp = env_dict[args.env](args)
        
    if args.obsnoise > 0.0:
        env = NoisyEnvironment(env, args.obsnoise)
        
    if args.act_latency > 0 or args.obs_latency > 0:
        env = LatencyEnvironment(env, act_lat=args.act_latency, obs_lat = args.obs_latency)
    
    if args.normalize_act:
        # TODO: Query environment        
        env = NormalizingEnvironment(env, -1, 1, obs=args.normalize_obs, rwd=args.normalize_rwd)
     
    if args.dbg_len is not None:
        env = RandomStopEnvironment(env, args.dbg_len[0], args.dbg_len[1], rng=args.rng)

    if args.addrwd:
        env = ExtendedEnvironment(env)
        
    if args.dbg_reward<>0.0:
        env = RewardShapingEnv(env, fwd_coeff=args.dbg_reward)
        
    if args.render:
        env = Renderer(env, render=lambda x: x%args.vrate==0)
        
    if args.p_obs_fail is not None:
        env = SensorFailureEnvironment(env, obs_T=args.T_obs_fail, failure_obs_p=args.p_obs_fail)
        
    
            
    return env, model_exp, pi_exp

