import sys
from unittest.mock import MagicMock

# Mock the OnlineLearningAgent and AdanTradingEnv before importing TrainingOrchestrator
sys.modules['adan_trading_bot.online_learning_agent'] = MagicMock()
sys.modules['adan_trading_bot.environment.multi_asset_env'] = MagicMock()
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

import unittest
from adan_trading_bot.training_orchestrator import TrainingOrchestrator

class TestTrainingOrchestrator(unittest.TestCase):

    def test_setup_environments(self):
        env_configs = [{'id': 'env1'}, {'id': 'env2'}]
        agent_config = {'param': 'value'}
        orchestrator = TrainingOrchestrator(env_configs, agent_config, shared_buffer_size=100000)
        orchestrator.setup_environments()
        self.assertEqual(len(orchestrator.agents), 2)
        # Verify that OnlineLearningAgent was called with the correct config
        for agent in orchestrator.agents:
            agent_mock = sys.modules['adan_trading_bot.online_learning_agent'].OnlineLearningAgent
            agent_mock.assert_called_with(model=None, env=None, config=agent_config, experience_buffer=orchestrator.shared_experience_buffer)

    def test_train(self):
        env_configs = [{'id': 'env1'}]
        agent_config = {'param': 'value'}
        orchestrator = TrainingOrchestrator(env_configs, agent_config, shared_buffer_size=100000)
        orchestrator.setup_environments()
        orchestrator.train()
        # Verify that the run method of the mocked agent was called
        orchestrator.agents[0].run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
