"""
Simple script showing iterative policy evaluation based on policy
going always right.
"""
from typing import Callable

from environments.frozen_lake.custom_frozen_lake import FrozenLakeEnv, RIGHT
from utils.DP_agents import DiscreteAgent
from utils.environment import Environment


def right_policy(state: int) -> int:
    """
    Policy for moving always right.
    :param state: unused
    :return: int describing right direction
    """
    del state  # unused
    return RIGHT

from utils.policies import get_random_policy
right_policy = get_random_policy(4)

def find_value_function(
        policy: Callable = right_policy,
        gamma: float = 0.9,
        iterations: int = 10,
        env: Environment = FrozenLakeEnv()):
    """
    Finds state value function for agent's policy
    :param policy: function describing agent's behaviour
    :param gamma: discount factor
    :param iterations: number of policy evaluation steps in a loop
    :param env: environment used in example
    :return: approximate state value function
    """

    agent = DiscreteAgent(env=env, policy=policy, gamma=gamma)
    value_function = agent.iterative_policy_evaluation(iterations=iterations)
    return value_function


def main():
    environment = FrozenLakeEnv()
    print('This is our environment. On red is agent\'s position.')
    environment.render()

    approximate_value_function = find_value_function(gamma=0.9)

    print('\nThis is how value function looks for gamma = 0.9. '
          'Try to change it to 0 or 1\n')
    print(approximate_value_function.reshape((4, 4)))


if __name__ == '__main__':
    main()
