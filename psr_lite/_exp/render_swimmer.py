import numpy as np
import matplotlib.pyplot as plt

def render_swimmer(theta,color):
    x,y,z = theta
    plt.plot(np.array([0,np.cos(x)]), np.array([0,np.sin(x)]), color=color)
    plt.hold(True)
    bx = -np.cos(y+x)
    by = -np.sin(y+x)
    plt.plot(np.array([0,bx]), np.array([0,by]), color=color)
    plt.plot(np.array([bx,bx-np.cos(y+x+z)]), np.array([by,by-np.sin(y+x+z)]), color=color)
    plt.xlim(-2.0,1.0)
    plt.ylim(-1.0,1.0) 
    