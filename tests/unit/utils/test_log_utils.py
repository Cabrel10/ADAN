"""
Unit tests for the log_utils module.
"""
import json
import logging
import os
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import yaml

from adan_trading_bot.utils.log_utils import LogManager, setup_log_management


class TestLogManager(unittest.TestCase):
    """Test cases for the LogManager class."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.test_dir / "logs"
        self.log_file = self.log_dir / "test.log"
        self.json_log_file = self.log_dir / "test.json"
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize LogManager
        self.log_manager = LogManager(
            log_dir=self.log_dir,
            log_file=self.log_file.name,
            json_log_file=self.json_log_file.name,
            max_size=1024,  # 1KB for testing
            backup_count=2,
            compress_backups=False
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_rotate_logs_creates_backup(self):
        """Test that log rotation creates a backup file."""
        # Create a log file that exceeds the max size
        with open(self.log_file, 'w') as f:
            f.write('x' * 2048)  # 2KB > 1KB max_size
        
        # Rotate logs
        self.log_manager.rotate_logs()
        
        # Check that a backup was created
        backups = list(self.log_dir.glob(f"{self.log_file.name}.*"))
        self.assertEqual(len(backups), 1)
    
    def test_rotate_logs_respects_backup_count(self):
        """Test that log rotation respects the backup count."""
        # Create multiple log files that would exceed the backup count
        for i in range(3):
            with open(self.log_file, 'w') as f:
                f.write(f"Test log entry {i}\n" * 300)  # Make it large enough to trigger rotation
            
            # Rotate logs
            self.log_manager.rotate_logs()
            
            # Add a small delay to ensure different modification times
            time.sleep(0.1)
        
        # Check that we only have the specified number of backups
        backups = sorted(
            self.log_dir.glob(f"{self.log_file.name}.*"),
            key=os.path.getmtime
        )
        self.assertLessEqual(len(backups), self.log_manager.backup_count)
    
    def test_search_logs_finds_matching_entries(self):
        """Test that search_logs finds matching log entries."""
        # Create a test log file with some entries
        test_entries = [
            {
                "timestamp": "2023-01-01T12:00:00Z",
                "level": "INFO",
                "message": "Test message 1",
                "module": "test_module",
                "function": "test_function"
            },
            {
                "timestamp": "2023-01-01T12:01:00Z",
                "level": "ERROR",
                "message": "Test error message",
                "module": "test_module",
                "function": "test_function"
            },
            {
                "timestamp": "2023-01-01T12:02:00Z",
                "level": "DEBUG",
                "message": "Debug message",
                "module": "test_module",
                "function": "test_function"
            }
        ]
        
        # Write entries to the log file
        with open(self.json_log_file, 'w') as f:
            for entry in test_entries:
                f.write(json.dumps(entry) + '\n')
        
        # Test text search
        results = self.log_manager.search_logs(query="error")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["message"], "Test error message")
        
        # Test level filter
        results = self.log_manager.search_logs(level="INFO")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["level"], "INFO")
        
        # Test time filter
        start_time = "2023-01-01T11:59:00Z"
        end_time = "2023-01-01T12:01:30Z"
        results = self.log_manager.search_logs(
            start_time=start_time,
            end_time=end_time
        )
        self.assertEqual(len(results), 2)  # Should match first two entries
    
    def test_cleanup_old_logs(self):
        """Test that old log files are cleaned up."""
        # Create some test log files with old modification times
        old_file = self.log_dir / "old_log.log"
        old_file.touch()
        old_time = time.time() - (60 * 60 * 24 * 31)  # 31 days ago
        os.utime(old_file, (old_time, old_time))
        
        # Create a recent log file
        recent_file = self.log_dir / "recent_log.log"
        recent_file.touch()
        
        # Clean up logs older than 30 days
        self.log_manager.cleanup_old_logs(days_to_keep=30)
        
        # Check that only the recent file remains
        self.assertFalse(old_file.exists())
        self.assertTrue(recent_file.exists())


class TestSetupLogManagement(unittest.TestCase):
    """Test cases for the setup_log_management function."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "logging_config.yaml"
        
        # Create a test config file
        config = {
            'log_dir': str(self.test_dir / 'custom_logs'),
            'log_file': 'custom.log',
            'json_log_file': 'custom.json',
            'max_log_size': 2048,  # 2KB
            'backup_count': 3,
            'compress_backups': False
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_setup_log_management_with_config(self):
        """Test setup_log_management with a config file."""
        # Get the expected log directory path
        expected_log_dir = self.test_dir / 'custom_logs'
        print(f"TEST - Expected log dir: {expected_log_dir}")
        print(f"TEST - Config file path: {self.config_file}")
        
        # Setup log management with config
        log_manager = setup_log_management(
            config_path=self.config_file,
            cleanup_days=30
        )
        
        print(f"TEST - Actual log dir: {log_manager.log_dir}")
        print(f"TEST - log_dir type: {type(log_manager.log_dir)}")
        print(f"TEST - log_dir absolute: {log_manager.log_dir.absolute()}")
        
        # Check that the log manager was configured correctly
        self.assertEqual(str(log_manager.log_dir.absolute()), str(expected_log_dir.absolute()))
        self.assertEqual(log_manager.log_file.name, 'custom.log')
        self.assertEqual(log_manager.json_log_file.name, 'custom.json')
        self.assertEqual(log_manager.max_size, 2048)
        self.assertEqual(log_manager.backup_count, 3)
        self.assertFalse(log_manager.compress_backups)
        
        # Verify the log directory was created
        self.assertTrue(expected_log_dir.exists())
        self.assertTrue(expected_log_dir.is_dir())
    
    def test_setup_log_management_without_config(self):
        """Test setup_log_management without a config file."""
        # Setup log management without config
        custom_dir = self.test_dir / 'default_logs'
        log_manager = setup_log_management(
            log_dir=str(custom_dir),
            log_file='default.log',
            json_log_file='default.json',
            max_size=1024,
            backup_count=5,
            compress_backups=True,
            cleanup_days=30
        )
        
        # Check that the log manager was configured with default values
        self.assertEqual(log_manager.log_dir.resolve(), custom_dir.resolve())
        self.assertEqual(log_manager.log_file.name, 'default.log')
        self.assertEqual(log_manager.json_log_file.name, 'default.json')
        self.assertEqual(log_manager.max_size, 1024)
        self.assertEqual(log_manager.backup_count, 5)
        self.assertTrue(log_manager.compress_backups)


if __name__ == '__main__':
    unittest.main()
