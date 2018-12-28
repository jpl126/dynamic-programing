from abc import abstractmethod


class Environment:
    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self):
        pass

    # @abstractmethod
    # def set_agent_state(self):
    #     pass
