from typing import Callable, Tuple

import copy
import numpy as np

from utils.policies import get_random_policy
from utils.environment import Environment

ITERATION_NO = 100  # TODO: remove to config


class Agent:
    def policy(self, *args, **kwargs):
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
            env: Environment,
            policy: Callable = None,
            gamma: float = 0.9):
        self.env = env
        self.states_count = self.env.states_count
        self.actions_count = self.env.actions
        self.value_function = np.array([0.0] * self.states_count)
        self.gamma = gamma
        if policy:
            self.policy = policy
        else:
            self.policy = get_random_policy(self.actions_count)

    def get_transition_matrix(
            self, iterations: int = ITERATION_NO) -> np.array:
        env = copy.deepcopy(self.env)
        states_no = env.states_count
        transition_matrix = np.array([[0.0] * states_no] * states_no)

        for state in range(states_no):
            for _ in range(iterations):
                env.observation = state
                action = self.policy(state=state)
                following_state, _, _, _ = env.step(action)

                transition_matrix[state][following_state] += 1.0

        return transition_matrix / iterations

    def get_expected_rewards(self, iterations: int = ITERATION_NO) -> np.array:
        env = copy.deepcopy(self.env)
        states_no = env.states_count
        expected_rewards = np.array([0.0] * states_no)
        good_actions = 0

        for state in range(states_no):


            for _ in range(iterations):
                env.observation = state
                action = self.policy(state=state)


                if state == 14 and action == 2:
                    good_actions += 1

                _, reward, _, _ = env.step(action)
                expected_rewards[state] += reward
        print(f'good_actions={good_actions}')
        return expected_rewards / iterations

    def apply_bellman_expectation_equation(self) -> np.array:
        r_pi = self.get_expected_rewards()
        # print(r_pi)
        p_pi = self.get_transition_matrix()
        # print(p_pi)

        self.value_function = r_pi + self.gamma * p_pi.dot(self.value_function)
        return self.value_function

    def find_value_function(self, iterations: int = 1):
        for _ in range(iterations):
            self.apply_bellman_expectation_equation()
        return self.value_function

    def create_greedy_policy(self):
        # TODO: we need action-value function to act greedy so this is a small cheat
        states_actions = []
        for i in range(self.states_count):
            states_actions.append(self._find_best_action_in_given_state(i))

        # print('\n')
        # print('new states_actions')
        # print(states_actions)
        # print('new states_actions')
        # print('\n')

        def new_policy(state: int) -> int:
            current_state_best_actions = states_actions[state]
            return np.random.choice(current_state_best_actions)

        return new_policy

    def apply_greedy_policy(self):
        self.policy = self.create_greedy_policy()
        # print('UP')
        # print(self.get_transition_matrix() * 10)
        # print('DOWN')

    def _find_best_action_in_given_state(self, state: int) -> Tuple[int, ...]:
        # TODO if we would like to have better accuracy we would have to iterate this process and get 'expected action-value funcion'
        env = copy.deepcopy(self.env)
        actions = []
        max_value_function = -np.inf
        for a in range(self.actions_count):
            env.observation = state
            following_state, _, _, _ = env.step(a)
            following_value_function = self.value_function[following_state]
            if following_value_function > max_value_function:
                max_value_function = following_value_function
                actions = [a]
            elif following_value_function == max_value_function:
                actions.append(a)
        print(f'For state: {state} best actions are/is: {actions}.')
        return tuple(actions)




"""gubie fakt że za przejście do celu dostaję nagrodę"""