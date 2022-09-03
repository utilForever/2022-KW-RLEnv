from __future__ import unicode_literals

import gym
import gym_gridworld

# To make the environment 
env = gym.make('gridworld-v0')

while True:
    env.render()
    _ = env.step(env.action_space.sample())