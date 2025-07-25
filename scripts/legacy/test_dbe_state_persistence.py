
import sys
sys.path.insert(0, '/home/morningstar/Documents/trading/ADAN/src')

import unittest
import os
import pickle
from pathlib import Path
from adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine

class TestDBEStatePersistence(unittest.TestCase):

    def setUp(self):
        self.test_save_path = Path("./test_dbe_state")
        self.test_save_path.mkdir(parents=True, exist_ok=True)
        self.dbe = DynamicBehaviorEngine(config={"state_persistence": {"save_path": str(self.test_save_path)}})

    def tearDown(self):
        for f in self.test_save_path.glob("*"):
            os.remove(f)
        os.rmdir(self.test_save_path)

    def test_save_and_load_state(self):
        # Modify the DBE state
        self.dbe.state['current_step'] = 123
        self.dbe.state['winrate'] = 0.75
        self.dbe.trade_history.append({"pnl_pct": 0.05})

        # Save the state
        save_file = self.test_save_path / "dbe_state.pkl"
        self.assertTrue(self.dbe.save_state(save_file))

        # Load the state into a new DBE instance
        new_dbe = DynamicBehaviorEngine.load_state(save_file)

        # Check if the state is restored correctly
        self.assertIsNotNone(new_dbe)
        self.assertEqual(new_dbe.state['current_step'], 123)
        self.assertEqual(new_dbe.state['winrate'], 0.75)
        self.assertEqual(len(new_dbe.trade_history), 1)
        self.assertEqual(new_dbe.trade_history[0]["pnl_pct"], 0.05)

if __name__ == '__main__':
    unittest.main()
