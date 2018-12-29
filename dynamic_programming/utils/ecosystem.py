import numpy as np

from utils.agents import Agent
from utils.environment import Environment


class Ecosystem:
    def __init__(self, agent: Agent, environment: Environment,
                 gamma: float = 0.9):
        self.agent = agent
        self.environment = environment
        self.gamma = gamma

        self.value_function = np.array([0.0] * self.environment.states_count)

    def get_transition_probability_matrix_for_policy(
            self,
            iterations_per_state: int = 1000) -> np.array:

        states_no = self.environment.states_count
        probability_matrix = np.array([[0.0] * states_no] * states_no)

        for state in range(states_no):
            for _ in range(iterations_per_state):
                self.environment.observation = state
                action = self.agent.policy(state=state)
                following_state, _, _, _ = self.environment.step(action)
                probability_matrix[state][following_state] += 1.0

        return probability_matrix / iterations_per_state

    def get_expected_rewards_for_policy(
            self,
            iterations_per_state: int = 1000) -> np.array:

        states_no = self.environment.states_count
        expected_reward = np.array([0.0] * states_no)

        for state in range(states_no):
            for _ in range(iterations_per_state):
                self.environment.observation = state
                action = self.agent.policy(state=state)
                _, reward, _, _ = self.environment.step(action)
                expected_reward[state] += reward

        return expected_reward / iterations_per_state

    def bellman_expectation_equation(self) -> np.array:
        r_pi = self.get_expected_rewards_for_policy()
        p_pi = self.get_transition_probability_matrix_for_policy()

        self.value_function = r_pi + self.gamma * p_pi.dot(self.value_function)
        return self.value_function
