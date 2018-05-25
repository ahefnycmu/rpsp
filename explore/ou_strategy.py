from explore.base import ExplorationStrategy
import numpy as np
import numpy.random as nr
import NN_policies


class OUStrategy(ExplorationStrategy):
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    theta=0.15, sigma=0.3, mu=0
    where Wt denotes the Wiener process
    """

    def __init__(self, base, mu=0, theta=0.25, sigma=0.1, **kwargs): 
        super(OUStrategy, self).__init__(base)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self._base_policy.output_dim) * self.mu
        self.reset()
        self.d={}

    def __getstate__(self):
        return np.copy(self.state)

    def __setstate__(self, d):
        self.state = np.copy(d)

    #overrides
    def reset(self):
        self.state = np.ones(self._base_policy.output_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state, dx

    
    def get_action(self, t, out, **kwargs):
        action = out[0]
        ou_state, dx = self.evolve_state()     
        diagnostics={'act_var':dx}
        return action  + ou_state , out[1], diagnostics


if __name__ == "__main__":
    policy = NN_policies.ContinuousPolicy(6,5,1,16)
    ou = OUStrategy(policy, mu=0, theta=0.15, sigma=0.3)
    states = []
    for i in range(1000):
        states.append(ou.evolve_state()[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
