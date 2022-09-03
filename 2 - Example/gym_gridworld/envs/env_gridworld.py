import gym
import sys
import os
import copy
import numpy as np

from gym import spaces

from PIL import Image as Image
import matplotlib.pyplot as plt

# Define colors
# 0: Black; 1: Gray; 2: Green; 3: Red, 4: Blue
COLORS = {0: [0.0, 0.0, 0.0], 1: [0.5, 0.5, 0.5],
          2: [0.0, 1.0, 0.0], 3: [1.0, 0.0, 0.0],
          4: [0.0, 0.0, 1.0]}


class GridworldEnv(gym.Env):
    num_env = 0

    def __init__(self):
        # Action space
        self.actions = [0, 1, 2, 3, 4]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {
            0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # Observation space
        self.obs_shape = [128, 128, 3]  # Observation space shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        # Construct grid map
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(file_path, 'map.txt')
        self.initial_map = self.read_grid_map(self.grid_map_path)
        self.current_map = copy.deepcopy(self.initial_map)
        self.observation = self.gridmap_to_observation(self.initial_map)
        self.grid_shape = self.initial_map.shape

        # Agent actions
        self.start_state, self.target_state = self.get_agent_states(
            self.initial_map)
        self.agent_state = copy.deepcopy(self.start_state)

        # Set other parameters
        GridworldEnv.num_env += 1
        self.fig_num = GridworldEnv.num_env
        self.fig = plt.figure(self.fig_num)
        plt.show(block=False)
        plt.axis('off')

    def step(self, action):
        # Return next observation, reward, finished, success
        action = int(action)
        next_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                      self.agent_state[1] + self.action_pos_dict[action][1])

        # Stay in place
        if action == 0:
            return (self.observation, 0, False)

        # Out of bounds condition
        if next_state[0] < 0 or next_state[0] >= self.grid_shape[0]:
            return (self.observation, 0, False)
        if next_state[1] < 0 or next_state[1] >= self.grid_shape[1]:
            return (self.observation, 0, False)

        # Successful behavior
        cur_color = self.current_map[self.agent_state[0], self.agent_state[1]]
        new_color = self.current_map[next_state[0], next_state[1]]
        if new_color == 0:  # Black - empty
            if cur_color == 3:  # Red - agent
                self.current_map[self.agent_state[0], self.agent_state[1]] = 0
                self.current_map[next_state[0], next_state[1]] = 3
            self.agent_state = copy.deepcopy(next_state)
        elif new_color == 1:    # Gray - obstacle
            return (self.observation, 0, False)
        elif new_color == 2:    # Green - target
            self.current_map[self.agent_state[0], self.agent_state[1]] = 0
            self.current_map[next_state[0], next_state[1]] = 4
            self.agent_state = copy.deepcopy(next_state)
        self.observation = self.gridmap_to_observation(self.current_map)
        if next_state[0] == self.target_state[0] and next_state[1] == self.target_state[1]:
            target_observation = copy.deepcopy(self.observation)
            return (target_observation, 1, True)
        else:
            return (self.observation, 0, False)

    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.current_map = copy.deepcopy(self.initial_map)
        self.observation = self.gridmap_to_observation(self.initial_map)
        return self.observation

    def render(self):
        img = self.observation
        plt.clf()
        plt.imshow(img)
        self.fig.canvas.draw()
        plt.pause(0.00001)

    def read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        grids = np.array(list(map(lambda x:
                                  list(map(lambda y: int(y),
                                           x.split(' '))), grid_map))
                         )
        return grids

    def get_agent_states(self, initial_map):
        start_state = None
        target_state = None
        start_state = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(initial_map == 3)
        ))
        target_state = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(initial_map == 2)
        ))
        if start_state == [None, None] or target_state == [None, None]:
            sys.exit('Start or target state not specified')
        return start_state, target_state

    def gridmap_to_observation(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0]/grid_map.shape[0])
        gs1 = int(observation.shape[1]/grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i*gs0:(i+1)*gs0, j*gs1:(j+1) *
                            gs1] = np.array(COLORS[grid_map[i, j]])
        return observation
