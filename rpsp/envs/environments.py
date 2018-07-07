"""
Created on Mon Nov 28 10:10:43 2016

@author: ahefny, zmarinho
"""

import numpy as np
import time
from IPython import embed

from rpsp.filters.models import Trajectory, ObservableModel


class Environment(object):
    """ Base environment class"""
    @property
    def dimensions(self):
        '''
        Return a tuple (obs_dimension, act_dimension)
        '''
        raise NotImplementedError

    @property
    def observation_info(self):
        '''
        Return a list of numbers. 
        If the ith observation is discrete, the ith element in the list is set to 
        cardinality of the observation, otherwise it is set to -1.
        '''
        # By default, assume all observations are continuous        
        return [-1] * self.dimensions[0]

    @property
    def action_info(self):
        '''
        Return a list of numbers. 
        If the ith action is discrete, the ith element in the list is set to 
        cardinality of the action, otherwise it is set to -1.
        '''
        # By default, assume all actions are continuous        
        return [-1] * self.dimensions[1]

    def render(self):
        raise NotImplementedError

    def reset(self):
        '''
        Reset the environment and return the first observation
        '''
        raise NotImplementedError

    def step(self, a):
        '''
        Given action a return a tuple:
            (observation, reward, is_episode_done)
        '''
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    @property
    def env_state(self):
        raise NotImplementedError

    def set_state(self, qpos, qvalue):
        raise NotImplementedError

    @property
    def monitor(self):
        return None

    def rng(self):
        raise NotImplementedError

    def set_rng(self):
        raise NotImplementedError

    def run(self, model, policy, max_traj_length, min_traj_length=0,
            num_trajs=0, num_samples=0, render=False):
        '''
        Generate trajectories of length up max_traj_length each.
        
        Returns:
        A list of trajectories (See models.Trajectory).
        
        Additional parameters:
        - model: An object that implements models.FilteringModel interface. 
            Used to track the state. If None, an ObservableModel is used,
            which returns the current observation.
        - policy: An object that implements policies.Policy interface. 
            Used to provide actions.
        - render: Whether to render generated trajectories in real-time. 
            This calls 'render' method which needs ot be implemented.
        - num_trajs: Number of trajectories to return.
        - num_samples: Total number of samples in generated trajectories.
            
        Must set num_trajs or num_samples (but not both) to a positive number.
        '''
        trajs = []
        rngs = []

        if model is None:
            model = ObservableModel(self.dimensions[0])

        if (num_samples > 0) == (num_trajs > 0):
            raise ValueError('Must specify exactly one of num_trajs and num_samples')

        done_all = False
        d_o, d_a = self.dimensions
        i_sample = 0
        tic = time.time()
        best_traj = [0, [0.0]]

        while not done_all:
            obs = np.empty((max_traj_length, d_o))
            act = np.empty((max_traj_length, d_a))
            rwd = np.empty((max_traj_length, 1))
            vel = np.empty((max_traj_length, 1))
            act_probs = np.empty((max_traj_length, 1))
            env_states = []
            states = np.empty((max_traj_length, model.state_dimension))
            dbg_info = {}
            rngs.append(self.rng().get_state())

            # Make a reset for each trajectory
            policy.reset()
            o = self.reset()
            q = model.reset(o)
            o0 = np.copy(o)
            q0 = np.copy(q)
            env_states.append(self.env_state)
            forward_pos = self.env_state[0][0]

            for j in xrange(max_traj_length):
                if render: self.render()
                a, p, info = policy.sample_action(q)
                o, r, done = self.step(a)
                env_states.append(self.env_state)
                q = model.update_state(o, a)
                act[j, :] = a
                obs[j, :] = o
                rwd[j] = r
                states[j, :] = q
                act_probs[j, :] = p
                vel[j] = (self.env_state[0][0] - forward_pos) / float(self.dt)
                forward_pos = self.env_state[0][0]
                for (k, v) in info.items():
                    if j == 0:
                        # Build arrays for diagnostic info
                        if type(v) is np.ndarray:
                            dbg_info[k] = np.empty((max_traj_length, v.size))
                        else:
                            dbg_info[k] = np.empty((max_traj_length, 1))

                    dbg_info[k][j, :] = v  # act variance

                if done: break

            j += 1
            drop_traj = False

            if j >= min_traj_length:
                # Check if we need to truncate trajectory to maintain num_samples
                if num_samples > 0 and i_sample + j >= num_samples:
                    j -= (i_sample + j - num_samples)
                    done_all = True
                # TODO: remove this will never happen because of outer if?
                if j < min_traj_length:
                    # Last trajectory is too short. Ignore it.
                    drop_traj = True

                if not drop_traj:
                    i_sample += j

                    new_traj = Trajectory(obs=obs[:j, :], states=states[:j, :], act=act[:j, :], rewards=rwd[:j, :],
                                          act_probs=act_probs[:j, :], obs0=o0, state0=q0, rng=rngs[-1],
                                          vel=vel[:j, :])

                    for (k, v) in dbg_info.iteritems():
                        dbg_info[k] = v[:j, :]
                    new_traj.dbg_info = dbg_info
                    trajs.append(new_traj)

                    if np.sum(rwd[:j, :]) >= np.sum(trajs[best_traj[0]].rewards):
                        best_traj[0] = len(trajs) - 1
                        best_traj[1] = env_states

                    if num_trajs > 0 and len(trajs) == num_trajs:
                        done_all = True
        print ('Gathering trajectories took:', time.time() - tic)
        # add best trajectory
        trajs[best_traj[0]].env_states = best_traj[1]  # save env states for best trajectory
        trajs[-1].bib = best_traj[0]  # save best in batch on last trajectory
        return trajs


class GymEnvironment(Environment):
    '''
    Create an Envinroment wrapper on top of an OpenAIGym environment.
    gym_env can be an OpenAIGYm environment object or a string specifying
    an OpenAIGym environment name such as:
    'CartPole-v0'
    'MountainCar-v0'
    '''

    def __init__(self, gym_env, discrete=True, rng=None):
        import gym
        if type(gym_env) is str:
            gym_env = gym.make(gym_env)

        self.env = gym_env.env
        self.discrete = discrete
        self.set_rng(rng)
        self.env.reset()
        self._orig_env = self.env
        assert not isinstance(self._orig_env, gym.wrappers.time_limit.TimeLimit), embed()

    @property
    def monitor(self):
        return self.env.monitor

    @property
    def dt(self):
        return self.env.dt

    @property
    def dimensions(self):
        if self.discrete is True:
            return (self.env.observation_space.shape[0], 1)
        else:
            return (self.env.observation_space.shape[0],
                    self.env.action_space.shape[0])

    @property
    def action_info(self):
        if self.discrete:
            return [self.env.action_space.n]
        else:
            return [self.env.action_space.shape[0]];

    def reset(self):
        if hasattr(self.env, 'stats_recorder'):
            self.env.stats_recorder.done = True
        return self.env.reset()

    def step(self, a):
        o, r, done, _ = self.env.step(a)
        return (o, r, done)

    def render(self):
        self.env.render()

    def close(self):
        return self.env.close()

    @property
    def env_state(self):
        return (np.copy(self.env.data.qpos.flatten()), np.copy(self.env.data.qvel.flatten()))

    def set_state(self, qpos, qvel):
        return self.env.set_state(qpos.flatten(), qvel.flatten())

    def rng(self):
        return self.env.np_random

    def set_rng(self, rng):
        self.env.np_random = rng
        return




class EnvironmentWrapper(Environment):
    '''
    A base class for environment wrappers which can alter the behavior of a given
    base environment. The base implementation maintains the behavior of 
    the base environment.
    '''

    def __init__(self, base_environment):
        self._base = base_environment
        self._orig_env = base_environment._orig_env

    @property
    def dt(self):
        return self._base.dt

    @property
    def monitor(self):
        return self._base.monitor

    @property
    def dimensions(self):
        return self._base.dimensions

    @property
    def observation_info(self):
        return self._base.observation_info

    @property
    def action_info(self):
        return self._base.action_info

    def render(self):
        return self._base.render()

    def reset(self):
        return self._base.reset()

    def step(self, a):
        return self._base.step(a)

    def close(self):
        return self._base.close()

    @property
    def action_limits(self):
        return [self._base.action_space.low, self._base.action_space.high]

    @property
    def obs_limits(self):
        return [self._base.observation_space.low, self._base.observation_space.high]

    @property
    def env_state(self):
        return self._base.env_state

    def set_state(self, qpos, qvalue):
        return self._base.set_state(qpos, qvalue)

    def rng(self):
        return self._base.rng()

    def set_rng(self, value):
        self._base.set_rng(value)
        return


class PartiallyObservableEnvironment(EnvironmentWrapper):
    '''
    A wrapper that emits a subset of observations of a base environment
    Example:
        PartiallyObservableEnvironment(ironment('CartPole-v0'), [0,2])
        is a cart pole environment where only the position of the cart and the angle of
        the pole are observed.
    '''

    def __init__(self, base_environment, visible_indices):
        EnvironmentWrapper.__init__(self, base_environment)
        self._visible_idx = np.array(visible_indices)

    @property
    def dimensions(self):
        base_dim = self._base.dimensions
        return (self._visible_idx.size, base_dim[1])

    @property
    def observation_info(self):
        base_info = self._base.observation_info
        return [base_info[i] for i in self._visible_idx]

    def reset(self):
        o = self._base.reset();
        return o[self._visible_idx];

    def step(self, a):
        o, r, done = self._base.step(a)
        return o[self._visible_idx], r, done


class NoisyEnvironment(EnvironmentWrapper):
    '''
    A wrapper to modify observations and/or actions of an environment.
    The modification cannot change the dimensionality of the observation and action.
    '''

    def __init__(self, base_environment, obs_noise=1.0, act_noise=None):
        '''
        Create a noisy environment given an existing base environment.
        obs_noise and act_noise specify observation and action modifiers. 
        Possible modifier are:
            - None
            - float number - Specifies the standard deviation of Gaussian noise applied to all coordinates.
            - function handle - Specifies a function f(x) that returns modified x
        '''
        EnvironmentWrapper.__init__(self, base_environment)

        def default_modifier(p, d):
            if p is None:
                return lambda x: x
            elif type(p) is float or type(p) is int:
                return lambda x, s=p: x + self.rng().randn(d) * s
            else:
                return p

        self._base = base_environment
        self._obs_noise = default_modifier(obs_noise, base_environment.dimensions[0])
        self._act_noise = default_modifier(act_noise, base_environment.dimensions[1])

    def reset(self):
        o = self._base.reset();
        return self._obs_noise(o)

    def step(self, a):
        a = self._act_noise(a)
        o, r, done = self._base.step(a)
        return self._obs_noise(o), r, done


class SensorFailureEnvironment(NoisyEnvironment):
    '''
    A wrapper to modify observations and/or actions of an environment, making random failures of a max preset window size T.
    This will be a uniform failure prob (failure_obs_p) for each length in T:t_i  and success probability of 1-Sum_i t_i in a categorical distribution.
    The modification cannot change the dimensionality of the observation and action.
    '''

    def __init__(self, base_environment, obs_T=None, act_T=None, failure_obs_p=None, failure_act_p=None):
        super(SensorFailureEnvironment, self).__init__(base_environment, obs_noise=None)
        if failure_obs_p is not None: assert failure_obs_p > 0.0
        if failure_act_p is not None: assert failure_act_p > 0.0

        dO = base_environment.dimensions[0]
        dA = base_environment.dimensions[1]
        self._obs_fail_buffer = np.zeros((dO), dtype=int)
        self._act_fail_buffer = np.zeros((dA), dtype=int)

        def default_modifier(d, win, pFail, fBuffer):
            if pFail is None or pFail is 0.0:
                print('not using failure environment')
                return lambda x: x
            elif type(pFail) is float:
                def fail(x, fB, T, pF):
                    fB -= 1
                    # fails = self.rng().choice(2, d ,p=[1.-pF,pF])*T#(T+1, d, p=[1.-pF]+[pF/T]*T)
                    fails = self.rng().choice(2, p=[1.0 - pF, pF]) * np.ones((d), dtype=int) * T
                    fails[fB >= 0] = fB[fB >= 0]
                    fB[:] = fails
                    x[fB > 0] = 0.0
                    # print fails,fB,x
                    return x

                return lambda x: fail(x, fBuffer, win, pFail)
            else:
                raise NotImplementedError
        self._obs_noise = default_modifier(dO, obs_T, failure_obs_p, self._obs_fail_buffer)
        self._act_noise = default_modifier(dA, act_T, failure_act_p, self._act_fail_buffer)


class RewardShaper(EnvironmentWrapper):
    '''
    A wrapper to modify the reward function.
    '''

    def __init__(self, base_environment):
        EnvironmentWrapper.__init__(self, base_environment)

    def reset(self):
        self.o_before = self._base.reset();
        return self.o_before

    def _reward_function(self, a, o_before, o_after, r):
        raise NotImplementedError

    def step(self, a):
        o, r, done = self._base.step(a)
        r = self._reward_function(a, self.o_before, o, r)
        self.o_before = o
        return o, r, done


class RewardShapingEnv(RewardShaper):
    '''
    A wrapper to modify the reward function.
    '''
    def __init__(self, base_environment, ctrl_coeff=1e-3, fwd_coeff=1.0, fwd_idx=0):
        EnvironmentWrapper.__init__(self, base_environment)
        self.rwd_dt = self._orig_env.dt
        self._fwd_coeff = fwd_coeff
        self._ctrl_coeff = ctrl_coeff
        self._alive_bonus = 1.0
        self._fwd_idx = fwd_idx

    def _reward_function(self, a, q_before, q_after, r_old):
        vel = (q_after[self._fwd_idx, 0] - q_before[self._fwd_idx, 0]) / float(self.rwd_dt)
        r = self._fwd_coeff * vel
        r += self._alive_bonus  # 1.0
        r -= self._ctrl_coeff * np.square(a).sum()  # 1e-3
        # r= r/(self._ctrl_coeff + self._alive_bonus + self._fwd_coeff)
        # print('v={} ctrl={} rv={} rc={} r={}'.format(vel, np.square(a).sum(), self._fwd_coeff*vel, -self._ctrl_coeff* np.square(a).sum(), r))
        return r

    def step(self, a):
        posbefore = self._orig_env.model.data.qpos
        out = self._base.step(a)
        o = out[0]
        r = out[1]
        done = out[2]
        posafter = self._orig_env.model.data.qpos
        r = self._reward_function(a, posbefore, posafter, r)
        return o, r, done


class Renderer(EnvironmentWrapper):
    '''
    A wrapper to render video.
    '''

    def __init__(self, base_environment, render=lambda x: False):
        EnvironmentWrapper.__init__(self, base_environment)
        self._render = render
        self._iter = 0

    def reset(self):
        self._iter += 1.0
        return self._base.reset();

    def step(self, a):
        out = self._base.step(a)
        o = out[0]
        r = out[1]
        done = out[2]
        if self._render(self._iter):
            self._orig_env.render()
        return o, r, done




class LatencyEnvironment(EnvironmentWrapper):
    '''
    Create an environment given an existing base environment with latency on the actions.
    obs_lat and act_lat specify observation and action modifiers. 
    Possible modifier are:
        - None
        - int number - Specifies the latency in terms of iterations.
        - function handle - Specifies a function f(x) that returns modified x
    '''

    def _get_latency(self, p, x, key):
        buffer = self._buffer.get(key, [])
        buffer.append(x)
        if len(buffer[1:p + 1]) == p:
            return buffer.pop(0)
        else:
            return np.zeros(x.shape)

    def __init__(self, base_environment, obs_lat=0, act_lat=0, rng=None):
        EnvironmentWrapper.__init__(self, base_environment)
        self.rng = rng
        self._buffer = {}

        def default_modifier(p, k):
            if p == 0:
                return lambda x: x
            elif p > 0:
                return lambda x: self._get_latency(p, x, k)
            else:
                return p

        self._base = base_environment
        self._obs_latency = default_modifier(obs_lat, 'obs')
        self._act_latency = default_modifier(act_lat, 'act')

    def reset(self):
        o = self._base.reset();
        return self._obs_latency(o)

    def step(self, a):
        a = self._act_latency(a)
        o, r, done = self._base.step(a)
        return self._obs_latency(o), r, done


class NormalizingEnvironment(EnvironmentWrapper):
    '''
    A wrapper to modify observations and/or actions of an environment.
    The modification cannot change the dimensionality of the observation and action.
    Normalize actions, observations and rewards uppon request.
    '''
    def __init__(self, base_environment, action_low, action_high, obs=False, rwd=False, scale_r=1.0):
        EnvironmentWrapper.__init__(self, base_environment)
        self._low = action_low
        self._high = action_high
        self._o_alpha = 0.001
        self._r_alpha = 0.001
        self._o_mean = np.zeros(self.dimensions[0])
        self._o_var = np.ones(self.dimensions[0])
        self._r_mean = 0.0
        self._r_var = 1.0
        self._normalize_obs = obs
        self._normalize_rwd = rwd
        self._scale_r = scale_r

    def step(self, a):
        a_scaled = self._low + (a + 1.) * 0.5 * (self._high - self._low)
        a_scaled = np.clip(a_scaled, self._low, self._high)

        o, r, done = self._base.step(a_scaled)
        o_scaled = self._get_normalize_obs(o) if self._normalize_obs else o
        r_scaled = self._get_normalize_rwd(r) if self._normalize_rwd else r
        return o_scaled, r_scaled * self._scale_r, done

    def reset(self):
        o = self._base.reset()
        return self._get_normalize_obs(o) if self._normalize_obs else o

    def _get_normalize_obs(self, o):
        self._update_obs(o)
        return (o - self._o_mean) / (np.sqrt(self._o_var) + 1e-8)

    def _get_normalize_rwd(self, rwd):
        self._update_rwd(rwd)
        return (rwd - self._r_mean) / (np.sqrt(self._r_var) + 1e-8)

    def _update_obs(self, o):
        self._o_mean = (1 - self._o_alpha) * self._o_mean + self._o_alpha * o
        self._o_var = (1 - self._o_alpha) * self._o_var + self._o_alpha * (o - self._o_mean) ** 2

    def _update_rwd(self, rwd):
        self._r_mean = (1 - self._r_alpha) * self._r_mean + self._r_alpha * rwd
        self._r_var = (1 - self._r_alpha) * self._r_var + self._r_alpha * (rwd - self._r_mean) ** 2


class ExtendedEnvironment(EnvironmentWrapper):
    """
    Environment wrapper to add reward and in the observation model
    """
    def __init__(self, base_environment):
        EnvironmentWrapper.__init__(self, base_environment)

    def step(self, a):
        o, r, done = self._base.step(a)
        extended_o = np.hstack([o, r])
        return extended_o, r, done

    def reset(self):
        o = self._base.reset()
        extended_o = np.hstack([o, 0])
        return extended_o

    @property
    def dimensions(self):
        return (self._base.dimensions[0] + 1, self._base.dimensions[1])


class ContinuousEnvironment(EnvironmentWrapper):
    '''
    create wrapper for continuous environment pass the continuous simulator.
    For classic control environments with continuous simulators. Alternative to mujoco environments.
    For reproducibility please try with mujoco environments.
    '''

    def __init__(self, base_environment, simulator, qpos_dim=[], qvel_dim=[]):
        self.sim = simulator
        self.qpos_dim = qpos_dim
        self.qvel_dim = qvel_dim
        EnvironmentWrapper.__init__(self, base_environment)
        self.discrete = False

    @property
    def dt(self):
        return self.sim.dt(self._orig_env)

    @property
    def dimensions(self):
        return (self._orig_env.observation_space.shape[0], self._orig_env.action_space.n)

    @property
    def action_info(self):
        return [self._orig_env.action_space.n]

    def reset(self):
        return self.sim.reset(self._base)

    def step(self, a):
        ''' receive a continuous action and get the 
        corresponding state change according to simulator '''
        o, r, done, _ = self.sim.simulate(a, self._orig_env)
        return o, r, done

    @property
    def env_state(self):
        return (np.copy(self._orig_env.state[self.qpos_dim]),
                np.copy(self._orig_env.state[self.qvel_dim]))
