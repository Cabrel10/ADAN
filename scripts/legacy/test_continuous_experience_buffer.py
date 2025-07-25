
import sys
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

import unittest
from unittest.mock import MagicMock
import sys
sys.modules['adan_trading_bot.environment.multi_asset_env'] = MagicMock()
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3 import PPO
from adan_trading_bot.online_learning_agent import OnlineLearningAgent, ExperienceBuffer

class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))

    def step(self, action):
        return self.observation_space.sample(), 0, False, {}

    def reset(self):
        return self.observation_space.sample()

class TestContinuousExperienceBuffer(unittest.TestCase):

    def test_continuous_buffer(self):
        env = DummyEnv()
        model = PPO("MlpPolicy", env)

        # Create the first agent and its experience buffer
        agent1 = OnlineLearningAgent(model, env, {})
        agent1.experience_buffer.add("state1", "action1", 1, "state2", False, 1.0)

        # Create a second agent with the same experience buffer
        agent2 = OnlineLearningAgent(model, env, {}, experience_buffer=agent1.experience_buffer)

        # Check if the buffer is the same and contains the experience
        self.assertIs(agent1.experience_buffer, agent2.experience_buffer)
        self.assertEqual(len(agent2.experience_buffer), 1)
        self.assertEqual(agent2.experience_buffer.memory[0].state, "state1")

if __name__ == '__main__':
    unittest.main()
