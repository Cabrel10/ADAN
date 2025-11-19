"""Tests for environment detection and GPU optimization."""
import unittest
import torch
from adan_trading_bot.utils.environment_detector import (
    is_colab,
    is_gpu_available,
    get_gpu_memory_gb,
    get_device,
    get_optimal_vec_env_type,
    get_optimal_num_workers,
    get_environment_config,
)


class TestEnvironmentDetector(unittest.TestCase):
    """Test environment detection functions."""

    def test_is_colab(self):
        """Test Colab detection."""
        result = is_colab()
        self.assertIsInstance(result, bool)

    def test_is_gpu_available(self):
        """Test GPU availability detection."""
        result = is_gpu_available()
        self.assertIsInstance(result, bool)

    def test_get_gpu_memory_gb(self):
        """Test GPU memory retrieval."""
        memory = get_gpu_memory_gb()
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0.0)

    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
        if is_gpu_available():
            self.assertEqual(str(device), "cuda")
        else:
            self.assertEqual(str(device), "cpu")

    def test_get_optimal_vec_env_type(self):
        """Test optimal vec env type detection."""
        vec_env_type = get_optimal_vec_env_type()
        self.assertIn(vec_env_type, ["dummy", "subproc"])

    def test_get_optimal_num_workers(self):
        """Test optimal number of workers."""
        num_workers = get_optimal_num_workers()
        self.assertIsInstance(num_workers, int)
        self.assertGreaterEqual(num_workers, 1)
        self.assertLessEqual(num_workers, 4)

    def test_environment_config(self):
        """Test complete environment configuration."""
        config = get_environment_config()
        self.assertIsInstance(config, dict)
        self.assertIn("in_colab", config)
        self.assertIn("has_gpu", config)
        self.assertIn("gpu_memory_gb", config)
        self.assertIn("vec_env_type", config)
        self.assertIn("num_workers", config)
        self.assertIn("device", config)
        self.assertIn("device_obj", config)

    def test_colab_detection_consistency(self):
        """Test that Colab detection is consistent."""
        result1 = is_colab()
        result2 = is_colab()
        self.assertEqual(result1, result2)

    def test_gpu_detection_consistency(self):
        """Test that GPU detection is consistent."""
        result1 = is_gpu_available()
        result2 = is_gpu_available()
        self.assertEqual(result1, result2)

    def test_vec_env_type_matches_workers(self):
        """Test that vec env type matches worker count."""
        vec_env_type = get_optimal_vec_env_type()
        num_workers = get_optimal_num_workers()

        # Colab should use DummyVecEnv with 1 worker
        if is_colab():
            self.assertEqual(vec_env_type, "dummy")
            self.assertEqual(num_workers, 1)
        # Local with GPU should use SubprocVecEnv with 4 workers
        elif is_gpu_available():
            self.assertEqual(vec_env_type, "subproc")
            self.assertEqual(num_workers, 4)
        # CPU only should use DummyVecEnv with 1 worker
        else:
            self.assertEqual(vec_env_type, "dummy")
            self.assertEqual(num_workers, 1)


class TestGPUOptimization(unittest.TestCase):
    """Test GPU optimization configuration."""

    def test_gpu_memory_positive(self):
        """Test that GPU memory is positive or zero."""
        memory = get_gpu_memory_gb()
        self.assertGreaterEqual(memory, 0.0)

    def test_device_is_valid_torch_device(self):
        """Test that device is a valid PyTorch device."""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_environment_config_device_obj_matches_device(self):
        """Test that device_obj in config matches get_device()."""
        config = get_environment_config()
        device = get_device()
        self.assertEqual(str(config["device_obj"]), str(device))


if __name__ == "__main__":
    unittest.main()
