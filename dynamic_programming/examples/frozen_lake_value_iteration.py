"""
Simple script showing Value Iteration - starting from value function equal 0
for each state and ending with optimal one.
"""
import os
import time

from environments.frozen_lake.custom_frozen_lake import FrozenLakeEnv
from utils.DP_agents import DiscreteAgent
from utils.environment import Environment


def find_optimal_policy_by_value_iteration(
        gamma: float = 0.9,
        iterations: int = 10,
        env: Environment = FrozenLakeEnv()) -> DiscreteAgent:
    """
    Finds optimal policy for given discrete environment.
    :param gamma: discount factor
    :param iterations: number of policy iterations
    :param env: environment used in example
    :return: agent with optimal policy
    """

    agent = DiscreteAgent(env=env, gamma=gamma)
    agent.value_iteration(iterations=iterations)
    return agent


def main():
    os.system('cls' if os.name == 'nt' else 'clear')  # clear screen

    my_agent = find_optimal_policy_by_value_iteration()
    environment = FrozenLakeEnv()
    done = False
    i = 1

    print('Start')
    environment.render()

    start_state = environment.observation
    action = my_agent.policy(start_state)

    while not done:
        time.sleep(1)
        os.system('cls' if os.name == 'nt' else 'clear')  # clear screen
        state, reward, done, _ = environment.step(action)
        print(f'Step {i}')
        environment.render()
        action = my_agent.policy(state)
        i += 1


if __name__ == '__main__':
    main()
