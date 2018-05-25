
from mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from IPython import  embed
class SwimmerEnv(MujocoEnv, Serializable):
    import mujoco_env
    import envs
    from rllab.misc.overrides import overrides
    from rllab.misc import autoargs
    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        from envs.base import Step
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        return Step(next_obs, reward, done)
    
    @overrides
    def log_diagnostics(self, paths):
        from rllab.misc import logger
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)
            
import numpy as np
import gym.envs.mujoco.mujoco_env
from gym import utils
import os

class SwimmerEnv_rllab(gym.envs.mujoco.mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, rng = None):
        gym.envs.mujoco.mujoco_env.MujocoEnv.__init__(self, os.getcwd()+'/envs/mujoco/swimmer_rllab.xml', 6)
        utils.EzPickle.__init__(self)
        self.np_random = rng
        
    
    @property
    def action_bounds(self):
        return self.model.actuator_ctrlrange.T
        
    def _step(self, a):
        from envs.base import Step
        ctrl_cost_coeff = 1e-2 #1e-2 #0.0001
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        #reward_fwd = self.get_body_comvel("torso")[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        reward_ctrl = -ctrl_cost_coeff * np.square(a/scaling).sum()
        #print 'reward_ctrl:', reward_ctrl
        #print 'reward_fwd:', reward_fwd
        #print self.model.data.qpos
        #embed()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
