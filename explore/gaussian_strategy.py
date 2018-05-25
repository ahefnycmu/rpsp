
from explore.base import ExplorationStrategy
import numpy as np


class GaussianStrategy(ExplorationStrategy):
    """
    This strategy adds Gaussian noise to the action taken by the deterministic policy.
    """
    def __init__(self, base, max_sigma=1.0, min_sigma=0.01, decay_period=10^3):
        super(GaussianStrategy, self).__init__(base)
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period

    
    def get_action(self, t, out, **kwargs):
        action = out[0]
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(1.0, t * 1.0 / float(self._decay_period))
        diagnostics={'act_var':sigma}
        return action + np.random.normal(size=len(action)) * sigma, out[1], diagnostics
