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
            gamma: float = 0.9,
            iteration_no: int = ITERATION_NO):
        self.env = env
        self.states_count = self.env.states_count
        self.actions_count = self.env.actions_count
        self.value_function = np.array([0.0] * self.states_count)
        self.gamma = gamma
        self.iteration_no = iteration_no
        if policy:
            self.policy = policy
        else:
            self.policy = get_random_policy(self.actions_count)

    def iterative_policy_evaluation(self, iterations: int = 1):
        """
        Iteratively applies Bellman Expectation Equation to find value function
        for agent's policy.
        :param iterations: describe how many iterations will be proceeded
        :return: np.array[states_no] value function approximation
        """
        for _ in range(iterations):
            self._apply_bellman_expectation_equation()
        return self.value_function

    def policy_iteration(self, iterations: int = 10):
        """
        Applies Iterative Policy Evaluation and Greedy Policy Improvements to
        obtain optimal policy.
        Perform only one iteration of Policy Evaluation at each step.
        :param iterations: number of policy updates until stop.
        :return:
        """
        for _ in range(iterations):
            self.iterative_policy_evaluation()
            self.policy = self._create_greedy_policy()

    def _apply_bellman_expectation_equation(self) -> np.array:
        """
        Applies Bellman Expectation Equation to evaluates policy.
        :return: np.array[states_no] with new value function approximation
        """
        r_pi, p_pi = self._get_expected_rewards_and_transition_matrix()
        self.value_function = r_pi + self.gamma * p_pi.dot(self.value_function)
        return self.value_function

    def _get_expected_rewards_and_transition_matrix(
            self) -> Tuple[np.array, np.array]:
        """
        Approximates expected reward vector and transition probability matrix
        for agent's policy. They are used in the further
        Iterative Policy Evaluation or Policy Iteration.
        Precision is based on self.iteration_no variable.

        :return: expected_rewards, transition_matrix - where:
        + expected_rewards: np.array[states_no] with expected rewards
        + transition_matrix: np.array[states_no x states_no] with probabilities
        """
        env = copy.deepcopy(self.env)  # we cannot work on original environment
        states_no = env.states_count
        expected_rewards = np.array([0.0] * states_no)
        transition_matrix = np.array([[0.0] * states_no] * states_no)

        for state in range(states_no):
            for _ in range(self.iteration_no):
                env.observation = state
                action = self.policy(state=state)
                following_state, reward, _, _ = env.step(action)
                expected_rewards[state] += reward
                transition_matrix[state][following_state] += 1.0
        expected_rewards /= self.iteration_no
        transition_matrix /= self.iteration_no
        return expected_rewards, transition_matrix

    def _create_greedy_policy(self) -> Callable:
        """
        Based on value function creates new greedy policy.
        :return: function describing agent's behaviour
        """
        states_actions = []
        for i in range(self.states_count):
            states_actions.append(self._find_best_action_in_given_state(i))

        def new_policy(state: int) -> int:
            current_state_best_actions = states_actions[state]
            return np.random.choice(current_state_best_actions)

        return new_policy

    def _find_best_action_in_given_state(self, state: int) -> Tuple[int, ...]:
        """
        Finds the best possible action or actions to perform in a given state.

        WORKS ONLY WHEN ACTION CAN MOVE AGENT TO ONLY ONE STATE (DETERMINISTIC)

        :param state: int describing environment state
        :return: tuple with all optimal actions
        """
        env = copy.deepcopy(self.env)  # we cannot work on original environment
        actions = []
        max_expected_return = -np.inf
        for a in range(self.actions_count):
            env.observation = state
            following_state, reward, _, _ = env.step(a)
            following_value_function = self.value_function[following_state]
            expected_return = reward + self.gamma * following_value_function
            if expected_return > max_expected_return:
                max_expected_return = expected_return
                actions = [a]
            elif expected_return == max_expected_return:
                actions.append(a)
        return tuple(actions)
