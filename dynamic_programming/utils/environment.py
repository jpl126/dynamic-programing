from abc import abstractmethod


class Environment:
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self, action, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def states_count(self):
        pass

    @property
    @abstractmethod
    def actions_count(self):
        pass

    @property
    @abstractmethod
    def observation(self):
        pass

    @observation.setter
    @abstractmethod
    def observation(self, *args, **kwargs):
        pass
