#!/usr/bin/env python3
"""
Unit tests for Memory Manager.
"""

import unittest
import time
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from adan_trading_bot.common.memory_manager import (
    MemoryManager, MemoryPressureLevel, MemoryStats, 
    get_tensor_memory_usage, memory_profile
)


class TestMemoryManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'medium_threshold': 70.0,
            'high_threshold': 85.0,
            'critical_threshold': 95.0,
            'enable_monitoring': False,  # Disable for unit tests
            'monitoring_interval': 1.0,
            'enable_auto_gc': True,
            'enable_gpu_monitoring': True,
            'enable_mixed_precision': True,
            'enable_amp': True,
            'max_history_size': 10
        }
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.shutdown()
            
    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        self.memory_manager = MemoryManager(self.config)
        
        self.assertEqual(self.memory_manager.thresholds[MemoryPressureLevel.MEDIUM], 70.0)
        self.assertEqual(self.memory_manager.thresholds[MemoryPressureLevel.HIGH], 85.0)
        self.assertEqual(self.memory_manager.thresholds[MemoryPressureLevel.CRITICAL], 95.0)
        self.assertFalse(self.memory_manager.monitoring_enabled)
        self.assertTrue(self.memory_manager.auto_gc_enabled)
        
    def test_memory_stats_collection(self):
        """Test memory statistics collection."""
        self.memory_manager = MemoryManager(self.config)
        
        stats = self.memory_manager.get_memory_stats()
        
        self.assertIsInstance(stats, MemoryStats)
        self.assertGreater(stats.total_memory, 0)
        self.assertGreater(stats.available_memory, 0)
        self.assertGreater(stats.used_memory, 0)
        self.assertGreaterEqual(stats.memory_percent, 0)
        self.assertLessEqual(stats.memory_percent, 100)
        
    def test_pressure_level_detection(self):
        """Test memory pressure level detection."""
        self.memory_manager = MemoryManager(self.config)
        
        # Test different pressure levels
        self.assertEqual(
            self.memory_manager.get_pressure_level(50.0), 
            MemoryPressureLevel.LOW
        )
        self.assertEqual(
            self.memory_manager.get_pressure_level(75.0), 
            MemoryPressureLevel.MEDIUM
        )
        self.assertEqual(
            self.memory_manager.get_pressure_level(90.0), 
            MemoryPressureLevel.HIGH
        )
        self.assertEqual(
            self.memory_manager.get_pressure_level(98.0), 
            MemoryPressureLevel.CRITICAL
        )
        
    def test_garbage_collection(self):
        """Test garbage collection functionality."""
        self.memory_manager = MemoryManager(self.config)
        
        # Test normal garbage collection
        result = self.memory_manager.trigger_garbage_collection()
        self.assertIn('total', result)
        self.assertIsInstance(result['total'], int)
        
        # Test forced garbage collection
        result = self.memory_manager.trigger_garbage_collection(force=True)
        self.assertIn('gen0', result)
        self.assertIn('gen1', result)
        self.assertIn('gen2', result)
        
    def test_pressure_callbacks(self):
        """Test memory pressure callback registration and execution."""
        self.memory_manager = MemoryManager(self.config)
        
        callback_triggered = False
        callback_level = None
        
        def test_callback(level):
            nonlocal callback_triggered, callback_level
            callback_triggered = True
            callback_level = level
        
        # Register callback
        self.memory_manager.register_pressure_callback(
            MemoryPressureLevel.HIGH, test_callback
        )
        
        # Trigger pressure handling
        self.memory_manager.handle_memory_pressure(MemoryPressureLevel.HIGH)
        
        # Check callback was triggered
        self.assertTrue(callback_triggered)
        self.assertEqual(callback_level, MemoryPressureLevel.HIGH)
        
    def test_mixed_precision_context(self):
        """Test mixed precision context manager."""
        self.memory_manager = MemoryManager(self.config)
        
        # Test context manager creation
        context = self.memory_manager.create_mixed_precision_context()
        self.assertIsNotNone(context)
        
        # Test context usage
        with context:
            # Should not raise any errors
            pass
            
    def test_memory_summary(self):
        """Test memory usage summary generation."""
        self.memory_manager = MemoryManager(self.config)
        
        # Add some fake history
        stats = self.memory_manager.get_memory_stats()
        self.memory_manager.memory_history.append(stats)
        
        summary = self.memory_manager.get_memory_summary()
        
        self.assertIn('current', summary)
        self.assertIn('averages', summary)
        self.assertIn('thresholds', summary)
        self.assertIn('monitoring', summary)
        
        # Check current stats
        self.assertIn('memory_percent', summary['current'])
        self.assertIn('used_memory_gb', summary['current'])
        self.assertIn('pressure_level', summary['current'])
        
    def test_training_optimization(self):
        """Test memory optimization for training."""
        self.memory_manager = MemoryManager(self.config)
        
        original_interval = self.memory_manager.monitoring_interval
        
        # Test training optimization
        returned_interval = self.memory_manager.optimize_for_training()
        self.assertEqual(returned_interval, original_interval)
        
        # Interval should be adjusted for training
        self.assertLessEqual(self.memory_manager.monitoring_interval, 10.0)
        
        # Test restoration
        self.memory_manager.restore_monitoring_interval(original_interval)
        self.assertEqual(self.memory_manager.monitoring_interval, original_interval)
        
    def test_memory_efficient_decorator(self):
        """Test memory efficient decorator."""
        self.memory_manager = MemoryManager(self.config)
        
        # Create a test function with the decorator
        @self.memory_manager.memory_efficient_decorator()
        def test_function():
            return "test_result"
        
        # Test that the decorated function works
        result = test_function()
        self.assertEqual(result, "test_result")
        
    def test_loss_scaling(self):
        """Test loss scaling for mixed precision."""
        self.memory_manager = MemoryManager(self.config)
        
        # Create a test loss tensor
        loss = torch.tensor(1.0, requires_grad=True)
        
        # Test loss scaling
        scaled_loss = self.memory_manager.scale_loss(loss)
        
        # Should return a tensor (scaled or not depending on GPU availability)
        self.assertIsInstance(scaled_loss, torch.Tensor)
        
    def test_optimizer_step(self):
        """Test optimizer stepping with mixed precision."""
        self.memory_manager = MemoryManager(self.config)
        
        # Create a simple model and optimizer for testing
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test optimizer step (should not raise errors)
        self.memory_manager.step_optimizer(optimizer)
        
    def test_shutdown(self):
        """Test memory manager shutdown."""
        self.memory_manager = MemoryManager(self.config)
        
        # Test shutdown
        self.memory_manager.shutdown()
        
        # Should complete without errors
        self.assertTrue(True)
        
    @patch('psutil.virtual_memory')
    def test_memory_stats_with_mock(self, mock_memory):
        """Test memory stats with mocked system memory."""
        # Mock memory data
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB
            used=4 * 1024**3,  # 4GB
            percent=50.0
        )
        
        self.memory_manager = MemoryManager(self.config)
        stats = self.memory_manager.get_memory_stats()
        
        self.assertEqual(stats.total_memory, 8.0)  # 8GB
        self.assertEqual(stats.available_memory, 4.0)  # 4GB
        self.assertEqual(stats.used_memory, 4.0)  # 4GB
        self.assertEqual(stats.memory_percent, 50.0)
        
    def test_monitoring_thread_lifecycle(self):
        """Test monitoring thread start and stop."""
        config = self.config.copy()
        config['enable_monitoring'] = True
        config['monitoring_interval'] = 0.1  # Very short for testing
        
        self.memory_manager = MemoryManager(config)
        
        # Should start automatically
        self.assertTrue(self.memory_manager._monitoring_thread.is_alive())
        
        # Test manual stop
        self.memory_manager.stop_monitoring()
        
        # Wait a bit for thread to stop
        time.sleep(0.2)
        
        # Thread should be stopped
        self.assertFalse(self.memory_manager._monitoring_thread.is_alive())


class TestMemoryUtilities(unittest.TestCase):
    
    def test_tensor_memory_usage(self):
        """Test tensor memory usage utility."""
        usage = get_tensor_memory_usage()
        
        if torch.cuda.is_available():
            self.assertIn('allocated_gb', usage)
            self.assertIn('reserved_gb', usage)
            self.assertIsInstance(usage['allocated_gb'], float)
        else:
            self.assertIn('error', usage)
            
    def test_memory_profile_decorator(self):
        """Test memory profiling decorator."""
        
        @memory_profile
        def test_function():
            # Create some tensors to use memory
            tensors = [torch.randn(100, 100) for _ in range(5)]
            return sum(t.sum() for t in tensors)
        
        # Should complete without errors and return result
        result = test_function()
        self.assertIsInstance(result, torch.Tensor)


class TestMemoryStats(unittest.TestCase):
    
    def test_memory_stats_creation(self):
        """Test MemoryStats dataclass creation."""
        stats = MemoryStats(
            total_memory=8.0,
            available_memory=4.0,
            used_memory=4.0,
            memory_percent=50.0
        )
        
        self.assertEqual(stats.total_memory, 8.0)
        self.assertEqual(stats.available_memory, 4.0)
        self.assertEqual(stats.used_memory, 4.0)
        self.assertEqual(stats.memory_percent, 50.0)
        self.assertIsInstance(stats.timestamp, float)
        
    def test_memory_stats_with_gpu(self):
        """Test MemoryStats with GPU information."""
        stats = MemoryStats(
            total_memory=8.0,
            available_memory=4.0,
            used_memory=4.0,
            memory_percent=50.0,
            gpu_memory_used=2.0,
            gpu_memory_total=4.0,
            gpu_memory_percent=50.0,
            tensor_count=100,
            tensor_memory=1.5
        )
        
        self.assertEqual(stats.gpu_memory_used, 2.0)
        self.assertEqual(stats.gpu_memory_total, 4.0)
        self.assertEqual(stats.gpu_memory_percent, 50.0)
        self.assertEqual(stats.tensor_count, 100)
        self.assertEqual(stats.tensor_memory, 1.5)


if __name__ == '__main__':
    unittest.main()