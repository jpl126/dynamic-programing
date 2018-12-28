from typing import Tuple, Callable

from utils.policies import get_random_policy


class Agent:
    def policy(self):
        pass

    def iterative_policy_evaluation(self):
        pass

    def policy_iteration(self):
        pass

    def value_iteration(self):
        pass


class DiscreteAgent(Agent):
    def __init__(
            self,
            actions_no: int = 4,
            policy: Callable = None,
            env_states_grid: Tuple[int, ...] = (4, 4)):

        if policy:
            self.policy = policy
        else:
            self.policy = get_random_policy(actions_no)
        self.env_states_grid = env_states_grid

