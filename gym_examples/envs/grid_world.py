from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.spaces import Tuple, MultiDiscrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
class GridWorldEnv(Env):
    """
    The Rescue environment includes two agents controlled by a centralized source.
    The goal of the agents is to rescue all the victims in the environment and
    avoid hazards.


    ### Action Space
    Each agent can make one of four actions
    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP
    Since there are two agents, actions will be of the form (a1, a2) representing
    the action for a1 and the action for a2. Note, this increases the possible
    action space to 16.
    
    ### Observation Space
    The observation is a value representing the agent's current position as
    current_row * nrows + current_col (where both the row and col start at 0).
    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.
    For example, the 4x4 map has 16 possible observations.

    ### Rewards

    Reward schedule:
    - Agent collides with target (V): +5
    - Timestep: -1

    ### Arguments

    ```
    gym.make('Rescue-v0')
    ```
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.reset()
        self.nrow, self.ncol = self.desc.shape
        self.num_agents = 2
        self._action_to_direction = {
            # uses the same indexing as np arrays ( [0,0] is top left )
            # UP
            0: np.array([-1, 0]),
            # LEFT
            1: np.array([0, -1]),
            # DOWN
            2: np.array([1, 0]),
            # RIGHT
            3: np.array([0, 1]),
        }

        # each observation is the location of A1 and location of A2. Locations of
        # the hazards (H) and points of interest (V) remain constant
        self._agent_locations = np.array(list(zip(*np.where(self.desc == b'A'))))
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent1": spaces.Box(0, 5, shape=(2,), dtype=int),
                "agent2": spaces.Box(0, 5, shape=(2,), dtype=int)
            }
        )


        # two agents with 4 actions each
        self.action_space = spaces.MultiDiscrete([4] * self.num_agents)
        self.render_mode = render_mode
        
        

    def _get_obs(self):
        return {"agent1": self._agent_locations[0], "agent2": self._agent_locations[1]}

    def step(self, a):
        '''
        Given some action a, return the next state, reward
        
        '''
        # reward is -1 by default so the model can learn to visit the targets quickly
        reward = -1
        # for each agent
        for i in range(len(a)):
            # determine the agent position
            direction = self._action_to_direction[i]
            #overwrite wherever the agent was
            self.desc[self._agent_locations[i]] = b'.'
            # set location for agent i ( use clip so we don't go out of bounds )
            self._agent_locations[i] = np.clip(self._agent_locations[i] + direction, 0, self.nrow - 1)
            # if the agent reached a target, reward it with +5
            if (self._agent_locations[i] in np.array(list(zip(*np.where(self.desc == b'V'))))):
                reward += 5
            #overwrite wherever the agent is
            self.desc[self._agent_locations[i]] = b'A'
        # are there any more targets?
        terminated = b'V' not in self.desc.flatten()
        # agent locations are being used in place of state for the time being
        return (self._agent_locations, reward, terminated)

    def reset(self, seed: Optional[int] = None):
        desc = ["A...HV",
                "..V...",
                ".....V",
                "....HV",
                "..V...",
                ".....A"
                ]
        self.desc = np.asarray(desc, dtype="c")
        # use the ascii map to locate agents
        self._agent_locations = np.array(list(zip(*np.where(self.desc == b'A'))))
        super().reset(seed=seed)
        if self.render_mode == "ansi":
            self.render()
       

    def render(self):
        if self.render_mode is None:
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
        elif self.render_mode == "ansi":
            return self._render_text()

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
