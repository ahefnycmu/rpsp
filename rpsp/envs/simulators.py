"""
Created on Mon Nov 28 10:10:43 2016

@author: ahefny, zmarinho
"""

import numpy as np
import math
from IPython import embed

class Simulator(object): 
    def simulate(self, a, env):
        return "NotImplementedError"
    
    def reset(self,env):
        return env.reset()

class PredictedSimulator(Simulator):
    def __init__(self, model):
        self.model = model
        self.state=self.model._start
    
    def simulate(self, a, env):
        embed()
        self.model.predict(a)
        obs = env.state
        try:
            next_obs = self.model.predict(self.state, a=a).squeeze()
            self.state = self.model.filter(self.state, obs, a=a).squeeze()
        except ValueError:
            embed()
            
        env.state = next_obs
        (x,_, theta,_) = env.state
        done =  x < -env.x_threshold \
                or x > env.x_threshold \
                or theta < -env.theta_threshold_radians \
                or theta > env.theta_threshold_radians
        done = bool(done)
        if not done:
            reward = 1.0
        elif env.steps_beyond_done is None:
            # Pole just fell!
            env.steps_beyond_done = 0
            reward = 1.0
        else:
            if env.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            env.steps_beyond_done += 1
            reward = 0.0
        return np.array(env.state).squeeze(), reward, done, {}
    
    def reset(self,env):
        env.reset()
        self.state = self.model._start
        return
    
    
class CartpoleContinuousSimulator(Simulator):  
    def reset(self, env):
        Simulator.reset(self, env)
        return Simulator.reset(self, env)
    
    
    def simulate(self, force, env):               
        state = env.state
        try:
            x, x_dot, theta, theta_dot = env.state
        except ValueError:
            embed()
        #force = env.force_mag if action==1 else -env.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta) #env.force_mag
        temp = (force[0] + env.polemass_length * theta_dot * theta_dot * sintheta) / env.total_mass
        thetaacc = (env.gravity * sintheta - costheta* temp) / (env.length * (4.0/3.0 - env.masspole * costheta * costheta / env.total_mass))
        xacc  = temp - env.polemass_length * thetaacc * costheta / env.total_mass
        x  = x + env.tau * x_dot
        x_dot = x_dot + env.tau * xacc
        theta = theta + env.tau * theta_dot
        theta_dot = theta_dot + env.tau * thetaacc
        
        env.state = np.array((x,x_dot,theta,theta_dot))
        done =  x < -env.x_threshold \
                or x > env.x_threshold \
                or theta < -env.theta_threshold_radians \
                or theta > env.theta_threshold_radians
        done = bool(done)
        if not done:
            reward = 1.0
        elif env.steps_beyond_done is None:
            # Pole just fell!
            env.steps_beyond_done = 0
            reward = 1.0
        else:
            if env.steps_beyond_done == 0:
                print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            env.steps_beyond_done += 1
            reward = 0.0
        return np.array(env.state).squeeze(), reward, done, {}
    
    def dt(self, env):
        return env.tau
     
class MountainCarContinuousSimulator(Simulator):
    def simulate(self, action, env):
        position = env.state[0]
        velocity = env.state[1]
        force = min(max(action[0], -1.0), 1.0) #action between [-1,1]

        velocity += force -0.0025 * math.cos(3*position)
        velocity = np.clip(velocity, -env.max_speed, env.max_speed)

        position += velocity
        position = np.clip(position, env.min_position, env.max_position)
        if (position==env.min_position and velocity<0): velocity = 0

        done = bool(position >= env.goal_position)

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        env.state = np.array([position, velocity])
        return env.state, reward, done, {}
    
    def dt(self, env):
        return env.dt
     
     
class AcrobotContinuousSimulator(Simulator):
    def simulate(self, torque, env):
        from gym.envs.classic_control.acrobot import wrap, rk4, bound
        s = env.state
        #torque = env.AVAIL_TORQUE[a]
        torque = min(max(torque[0],-1), 1) # -np.ones(torque.shape)), np.ones(torque.shape))
        
        # Add noise to the force action
        if env.torque_noise_max > 0:
            torque += env.np_random.uniform(-env.torque_noise_max, env.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(env._dsdt, s_augmented, [0, env.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -env.MAX_VEL_1, env.MAX_VEL_1)
        ns[3] = bound(ns[3], -env.MAX_VEL_2, env.MAX_VEL_2)
        env.state = ns
        terminal = env._terminal()
        reward = -1. if not terminal else 0.
        return env.state, reward, terminal, {}

    def reset(self,env):
        env.reset()
        env.state = np.zeros(env.state.shape[0])
        return env.state  
    
    def dt(self, env):
        return env.dt 
        
class PendulumContinuousSimulator(Simulator):
    def simulate(self, u, env):
        from gym.envs.classic_control.pendulum import angle_normalize
        th, thdot = env.state # th := theta
        g = 10.
        m = 1.0
        l = 1.
        dt = env.dt
        env.scale_action = 100.0
        u = np.clip(u, -env.max_torque, env.max_torque)[0]
        
        env.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -env.max_speed, env.max_speed) #pylint: disable=E1111

        env.state = np.array([newth, newthdot])
        return env._get_obs(), -costs, False, {}
    
    def reset(self,env):
        env.reset()
        env.state = np.zeros(env.state.shape[0])
        env.state[2:] = env.max_torque #start with high velocity and up
        return env.state
    
    def dt(self, env):
        return env.dt