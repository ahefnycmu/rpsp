from environments import EnvironmentWrapper, GymEnvironment
from input_device import InputDevice
from input_policy import InputPolicy
import numpy as np

class ManualResetEnvironment(EnvironmentWrapper):
    '''
    A wrapper that causes the environment to reset only if a specified button on 
    an input device is pressed.
    '''        
    def __init__(self, base_env, input_device, reset_button_index):
        if not isinstance(input_device, InputDevice):
            raise TypeError('input_device must be of type %s' % InputDevice)
            
        super(ManualResetEnvironment, self).__init__(base_env)
        self._device = input_device
        self._reset_btn = reset_button_index
        
    def step(self, a):        
        o,r,done = self._base.step(a)
        
        self._device.poll()
        done = self._device.is_button_pressed(self._reset_btn)                
        return o,r,done
        
def create_manual_environment(env_name, rng, control_device_name='/dev/input/js0', reset_btn=0):
    '''
    Creates a ManualResetEnvironment and an associated InputPolicy.
    
    This function assumes an XBOX 360 controller:
        Axis 0: Left stick X
        Axis 1: Left stick Y
        Axis 2: Right stick X
        Axis 3: Right stick Y
        Axis 4: Left Throttle
        Axis 5: Right Throttle
    '''
    
    dev = InputDevice(control_device_name)
    reset_dev = dev.copy()
    
    if env_name == 'swimmer':
        env = GymEnvironment('Swimmer-v1', False, rng)        
        action_map = [(0,3,0,1),
                      (1,3,0,0.6)]        
    elif env_name == 'hopper':
        env = GymEnvironment('Hopper-v1', False, rng)               
        action_map = [(0,0,0,0.5),
                      (1,2,0,0.5),
                      (2,5,-0.5,-1.0),
                      ]
    elif env_name == 'cartpole':
        env = GymEnvironment('InvertedPendulum-v1', False, rng)               
        action_map = [(0,2,0,0.1)]
    
    dev.open()
    reset_dev.open()
    
    training_env = ManualResetEnvironment(env, reset_dev, reset_btn)
    pi = InputPolicy(dev, training_env.dimensions[1], action_map)
    return training_env, pi
    