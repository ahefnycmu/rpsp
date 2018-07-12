import numpy as np
from scipy.stats import norm

class Uniform:
    def __init__(self, low, high):
        assert high > low
        self._low = low
        self._high = high
        self._width = self._high - self._low

    def pdf(self, x):
        if x < self._low or x > self._high:
            return 0.0
        else:
            return 1.0 / self._width

    def sample(self):
        return np.random.rand() * self._width + self._low
        
class TruncatedNormal:
    def __init__(self, cutoff):
        self._cutoff = cutoff
        self._mass = norm.cdf(cutoff)-norm.cdf(-cutoff)

    def pdf(self, x):
        if abs(x) > self._cutoff:
            return 0.0
        else:
            return norm.pdf(x) / self._mass

    def cdf(self, x):
        if abs(x) > self._cutoff:
            return 0.0
        else:
            return (norm.cdf(x)-norm.cdf(-self._cutoff)) / self._mass
        
            
        
