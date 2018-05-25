from input_device import InputDevice
from envs.manual_env import create_manual_environment
import numpy as np

if __name__ == '__main__':
    from policies import SineWavePolicy, RandomGaussianPolicy
                
    training_env, pi = create_manual_environment('swimmer', np.random, '/dev/input/js0', 0)        
    #pi = SineWavePolicy(np.array([1.0,0.6]), np.array([100,100]), np.array([100.0,100.0]))
    #pi = SineWavePolicy(np.array([0.2]), np.array([30]), np.array([0])) 
    #pi = RandomGaussianPolicy(training_env.dimensions[1], np.random)
           
    trajs = training_env.run(None, pi, 10000, render=True, num_samples=10000)
    
    import matplotlib.pyplot as plt
    h1, = plt.plot(trajs[0].obs[:,1], label='joint1')
    plt.hold(True)
    h2, = plt.plot(trajs[0].obs[:,2], label='joint2')
    h3, = plt.plot(trajs[0].act[:,0], label='act')    
    plt.legend(handles=[h1,h2,h3])
    plt.show()
    
        

