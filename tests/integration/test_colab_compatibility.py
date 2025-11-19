"""Integration tests for Colab compatibility."""
import unittest
import os
import tempfile
from stable_baselines3.common.vec_env import DummyVecEnv
from adan_trading_bot.utils.environment_detector import (
    is_colab,
    get_optimal_vec_env_type,
    get_optimal_num_workers,
)


class TestColabCompatibility(unittest.TestCase):
    """Test Colab compatibility."""

    def test_environment_detection_is_consistent(self):
        """Test that environment detection is consistent."""
        vec_env_type_1 = get_optimal_vec_env_type()
        num_workers_1 = get_optimal_num_workers()

        vec_env_type_2 = get_optimal_vec_env_type()
        num_workers_2 = get_optimal_num_workers()

        self.assertEqual(vec_env_type_1, vec_env_type_2)
        self.assertEqual(num_workers_1, num_workers_2)

    def test_dummy_vec_env_creation(self):
        """Test that DummyVecEnv can be created without pickle issues."""
        def dummy_env():
            """Create a dummy environment for testing."""
            import gymnasium as gym
            return gym.make("CartPole-v1")

        env_fns = [dummy_env for _ in range(2)]

        try:
            env = DummyVecEnv(env_fns)
            self.assertIsNotNone(env)
            env.close()
        except Exception as e:
            self.fail(f"DummyVecEnv creation failed: {e}")

    def test_environment_detector_no_pickle_errors(self):
        """Test that environment detector functions don't cause errors."""
        try:
            vec_env_type = get_optimal_vec_env_type()
            num_workers = get_optimal_num_workers()
            is_in_colab = is_colab()

            self.assertIsNotNone(vec_env_type)
            self.assertIsNotNone(num_workers)
            self.assertIsNotNone(is_in_colab)
        except Exception as e:
            self.fail(f"Environment detector failed: {e}")

    def test_temp_directory_handling(self):
        """Test that temporary directories are handled correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(os.path.isdir(tmpdir))
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")
            self.assertTrue(os.path.isfile(test_file))


if __name__ == "__main__":
    unittest.main()
