from input_device import InputDevice
from policies import BasePolicy
import numpy as np

class InputPolicy(BasePolicy):
    '''
    A polic class the gets actions from an input device (a joystick)
    '''
    def __init__(self, input_device, num_actions, action_map):
        '''
        input_device must be an InputDevice object. This class does not take care fo opening 
        and closing the device.
        
        action_map is a list of tuples of the form (action index, axis index, bias, scale)
        '''
        
        if not isinstance(input_device, InputDevice):
            raise TypeError('input_device must be of type %s.' % str(InputDevice))
        
        self._input_device = input_device
        self._action_map = action_map
        self._num_actions = num_actions
        
    def sample_action(self, state):
        a = np.zeros(self._num_actions)          
        self._input_device.poll()
        
        for x in self._action_map:
            val = self._input_device.get_axis(x[1])            
            a[x[0]] += (val + x[2]) * x[3]
        
        return a,1.0,{}
    
if __name__ == '__main__':
    from envs.environments import GymEnvironment    
        
    dev = InputDevice('/dev/input/js0')
    env = GymEnvironment('Swimmer-v1', False, np.random)        
    action_map = [(0,3,0,1),
                  (1,3,0,0.6)]
    pi = InputPolicy(dev, env.dimensions[1], action_map)
            
    dev.open()
    trajs = env.run(None, pi, 1000, render=True, num_samples=1000)
    dev.close()
    