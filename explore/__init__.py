class ExplorationStrategy(object):
    def __init__(self, base_policy):
        self._base_policy = base_policy
        
    def sample_action(self, t, observation, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass
