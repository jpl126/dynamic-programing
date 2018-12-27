from .agent import Agent
from .environment import Environment


class Ecosystem:
    def __init__(self, agent: Agent, environment: Environment):
        self.agent = agent
        self.environment = environment
