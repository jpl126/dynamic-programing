from typing import Set, Tuple

import numpy as np

from utils.environment import Environment


class InvalidMoveException(ValueError):
    """
    Error raised when agent tries to take unpredicted action.
    """
    pass


MOVES = {
    0: 'LEFT',
    1: 'DOWN',
    2: 'RIGHT',
    3: 'UP'
}


class FrozenLakeEnv(Environment):
    _INT_TO_CHAR = {
        0: 'F',
        1: 'S',
        2: 'H',
        3: 'G'
    }

    def __init__(self, grid_size: Tuple[int, int] = (4, 4),
                 holes_positions: Tuple[Tuple[int, int]] = (
                     (1, 1), (1, 3), (2, 3), (3, 0)),
                 start_position: Tuple[int, int] = (0, 0),
                 goal_position: Tuple[int, int] = (3, 3)):

        self._agent_position = list(start_position)
        self._goal_position = list(goal_position)

        # grid: 0 - frozen, 1 - start, 2 - hole, 3 - goal
        self._grid = np.array([[0] * grid_size[0]] * grid_size[1])
        self._grid[start_position] = 1
        self._grid[goal_position] = 3
        for hole in holes_positions:
            self._grid[hole] = 2

    def render(self):
        """
        Prints frozen lake world. With highlighted agent's location.
        :return: None
        """
        text_grid = ''

        # + 1 is caused by '/n' at the end of each row
        state_no = (self._agent_position[0] * (self._grid.shape[0] + 1)
                    + self._agent_position[1])
        for row in self._grid:
            char_row = [self._INT_TO_CHAR[item] for item in row]
            text_grid += ''.join(char_row) + '\n'

        text_grid = (text_grid[:state_no] + '\033[41m' + text_grid[state_no]
                     + '\033[0m' + text_grid[state_no + 1:-1])

        print(text_grid)

    def step(self, action: int, *args,
             **kwargs) -> Tuple[int, float, bool, dict]:
        """
        Gives environment response to agent's action.
        :param action: move direction according to MOVES variable
        :return: (
        number of new agent's state,
        reward signal,
        True if episode is finished False otherwise,
        info: your info for debugging
        """
        del args, kwargs  # unused

        done = False
        info = {}
        reward = 0.0
        was_at_goal = self._is_agent_at_goal()
        nearby_walls = self._get_walls_next_to_agent()

        if action not in nearby_walls:
            self._change_agent_position(action)
            if self._is_agent_at_goal() and not was_at_goal:
                reward = 1.0
        if self._is_agent_in_hole() or self._is_agent_at_goal():
            done = True

        return self.observation, reward, done, info

    @property
    def observation(self):
        return self._get_observation()

    @observation.setter
    def observation(self, state: int, *args, **kwargs):
        del args, kwargs  # unused
        self._set_observation(state)

    @property
    def states_count(self):
        return sum(len(x) for x in self._grid)

    @property
    def actions(self) -> int:
        return 4

    def _get_walls_next_to_agent(self) -> Set[int]:
        """
        Gets walls next to the agent at his current position.
        Walls description:
            0: left wall,
            1: down wall,
            2: right wall,
            3: up wall
        :return: set with all walls next to the agent
        """
        walls = []
        if self._agent_position[0] == 0:
            walls.append(3)
        if self._agent_position[0] == self._grid.shape[0] - 1:
            walls.append(1)

        if self._agent_position[1] == 0:
            walls.append(0)
        if self._agent_position[1] == self._grid.shape[1] - 1:
            walls.append(2)

        return set(walls)

    def _get_goal_observation(self) -> int:
        """
        Returns goal state (grid cell) as an integer.
        :return: int describing current state
        """
        observation = (self._goal_position[0] * self._grid.shape[1]
                       + self._goal_position[1])
        return observation

    def _get_observation(self) -> int:
        """
        Returns current state (grid cell) as an integer.
        :return: int describing current state
        """
        observation = (self._agent_position[0] * self._grid.shape[0]
                       + self._agent_position[1])
        return observation

    def _set_observation(self, state: int):
        # TODO: add check if state is legit
        self._agent_position[0] = state // self._grid.shape[0]
        self._agent_position[1] = state % self._grid.shape[1]

    def _change_agent_position(self, direction: int):
        """
        Move agent to new position by one field. Rise an exception when agent
        leaves grid world.
        :param direction: direction according to MOVES variable
        :return: None
        """
        if direction in self._get_walls_next_to_agent():
            return

        if self._is_agent_in_hole() or self._is_agent_at_goal():
            return

        if direction == 0:
            self._agent_position[1] -= 1
        elif direction == 2:
            self._agent_position[1] += 1
        elif direction == 1:
            self._agent_position[0] += 1
        elif direction == 3:
            self._agent_position[0] -= 1
        else:
            raise InvalidMoveException

    def _is_agent_in_hole(self):
        """
        Check if agent felt into a hole
        :return: True if agent is in the hole, False otherwise
        """
        row_no = self._agent_position[0]
        col_no = self._agent_position[1]
        if self._grid[row_no, col_no] == 2:
            return True
        return False

    def _is_agent_at_goal(self):
        """
        Check if agent reached the goal
        :return: True if yes, False otherwise
        """
        return self.observation == self._get_goal_observation()
