import numpy as np

from utils.agents import Agent
from utils.environment import Environment


class Ecosystem:
    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment

    def get_transition_probability_matrix_for_policy(
            self) -> np.array:
        pass
