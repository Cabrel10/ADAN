#!/usr/bin/env python3
"""
Unit tests for the SystemMetricsCollector class.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
from adan_trading_bot.common.system_metrics import SystemMetricsCollector

class TestSystemMetricsCollector(unittest.TestCase):
    """Test cases for SystemMetricsCollector class."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'enabled': True,
            'update_interval': 1.0
        }
        self.metrics_collector = SystemMetricsCollector(self.config)
    
    def test_initialization(self):
        """Test that the collector initializes correctly."""
        self.assertTrue(hasattr(self.metrics_collector, 'metrics'))
        self.assertTrue(self.metrics_collector.enabled)
        self.assertEqual(self.metrics_collector.update_interval, 1.0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_update_metrics(self, mock_swap, mock_virtual_mem, mock_cpu_percent):
        """Test updating metrics with mocked system calls."""
        # Setup mocks
        mock_cpu_percent.return_value = 25.5
        
        class VirtualMemory:
            total = 16 * 1024 * 1024 * 1024  # 16GB
            available = 8 * 1024 * 1024 * 1024  # 8GB
            percent = 50.0
            used = 8 * 1024 * 1024 * 1024
            free = 8 * 1024 * 1024 * 1024
            
        class SwapMemory:
            total = 2 * 1024 * 1024 * 1024  # 2GB
            used = 1 * 1024 * 1024 * 1024  # 1GB
            free = 1 * 1024 * 1024 * 1024  # 1GB
            percent = 50.0
            sin = 0
            sout = 0
            
        mock_virtual_mem.return_value = VirtualMemory()
        mock_swap.return_value = SwapMemory()
        
        # Test update
        metrics = self.metrics_collector.update_metrics()
        
        # Verify CPU metrics
        self.assertIn('cpu', metrics)
        self.assertEqual(metrics['cpu']['usage_percent'], 25.5)
        
        # Verify memory metrics
        self.assertIn('memory', metrics)
        self.assertEqual(metrics['memory']['virtual']['total'], 16 * 1024 * 1024 * 1024)
        self.assertEqual(metrics['memory']['virtual']['percent'], 50.0)
        self.assertEqual(metrics['memory']['swap']['total'], 2 * 1024 * 1024 * 1024)
    
    def test_metrics_summary(self):
        """Test getting a summary of metrics."""
        # First update metrics
        test_metrics = {
            'cpu': {'usage_percent': 25.5},
            'memory': {'virtual': {'percent': 50.0}},
            'system': {'timestamp': '2023-01-01T00:00:00'}
        }
        
        # Add GPU metrics only if GPU is available
        if self.metrics_collector.gpu_available:
            test_metrics['gpu'] = {
                'devices': [
                    {'load': 30.0, 'memory_util': 40.0, 'temperature': 65.0}
                ]
            }
        
        self.metrics_collector.metrics = test_metrics
        
        # Test summary
        summary = self.metrics_collector.get_metrics_summary()
        
        # Verify summary contains expected keys and values
        self.assertEqual(summary['cpu_usage'], 25.5)
        self.assertEqual(summary['memory_usage'], 50.0)
        
        # Check GPU metrics only if GPU is available
        if self.metrics_collector.gpu_available:
            self.assertEqual(summary['gpu_usage'], 30.0)
            self.assertEqual(summary['gpu_memory_usage'], 40.0)
            self.assertEqual(summary['gpu_temperature'], 65.0)
    
    def test_disabled_collector(self):
        """Test that the collector doesn't collect metrics when disabled."""
        self.metrics_collector.enabled = False
        metrics = self.metrics_collector.update_metrics()
        self.assertEqual(metrics, {})

if __name__ == '__main__':
    unittest.main()
