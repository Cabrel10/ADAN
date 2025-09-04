"""Unit tests for the checkpoint manager module."""
import os
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import numpy as np

from adan_trading_bot.environment.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata
)

class TestModel(nn.Module):
    """Simple test model for checkpoint testing."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))

class TestCheckpointMetadata(unittest.TestCase):
    """Test cases for CheckpointMetadata class."""

    def test_metadata_creation(self):
        """Test metadata initialization and conversion."""
        metrics = {"reward": 1.23, "loss": 0.456}
        custom_data = {"config": {"learning_rate": 0.001}, "notes": "test"}

        metadata = CheckpointMetadata(
            timestamp=1234567890.0,
            episode=42,
            total_steps=1000,
            metrics=metrics,
            custom_data=custom_data,
            version="1.0"
        )

        # Test to_dict
        metadata_dict = metadata.to_dict()
        self.assertEqual(metadata_dict["episode"], 42)
        self.assertEqual(metadata_dict["total_steps"], 1000)
        self.assertEqual(metadata_dict["metrics"], metrics)
        self.assertEqual(metadata_dict["custom_data"], custom_data)

        # Test from_dict
        new_metadata = CheckpointMetadata.from_dict(metadata_dict)
        self.assertEqual(new_metadata.episode, 42)
        self.assertEqual(new_metadata.total_steps, 1000)
        self.assertEqual(new_metadata.metrics, metrics)
        self.assertEqual(new_metadata.custom_data, custom_data)
        self.assertEqual(new_metadata.version, "1.0")

class TestCheckpointManager(unittest.TestCase):
    """Test cases for CheckpointManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, "checkpoints")
        self.model = TestModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize checkpoint manager
        self.manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            max_checkpoints=3,
            checkpoint_interval=10
        )

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        # Save initial checkpoint
        metrics = {"reward": 1.23, "loss": 0.456}
        checkpoint_path = self.manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            episode=1,
            total_steps=10,
            metrics=metrics,
            custom_data={"test": "data"}
        )

        self.assertIsNotNone(checkpoint_path)
        self.assertTrue(os.path.exists(checkpoint_path))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, 'model.pt')))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, 'optimizer.pt')))
        self.assertTrue(os.path.exists(os.path.join(checkpoint_path, 'metadata.json')))

        # Load the checkpoint
        loaded_model = TestModel()
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters())

        loaded_model, loaded_optimizer, metadata = self.manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=loaded_model,
            optimizer=loaded_optimizer
        )

        # Verify metadata
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.episode, 1)
        self.assertEqual(metadata.total_steps, 10)
        self.assertEqual(metadata.metrics["reward"], 1.23)
        self.assertEqual(metadata.metrics["loss"], 0.456)
        self.assertEqual(metadata.custom_data["test"], "data")

        # Verify model state
        input_tensor = torch.randn(1, 2)
        with torch.no_grad():
            output1 = self.model(input_tensor)
            output2 = loaded_model(input_tensor)
        self.assertTrue(torch.allclose(output1, output2))

    def test_checkpoint_interval(self):
        """Test checkpoint interval functionality."""
        # Should not save (step 5 % 10 != 0)
        path1 = self.manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            episode=1,
            total_steps=5,
            metrics={"reward": 1.0}
        )
        self.assertIsNone(path1)

        # Should save (step 10 % 10 == 0)
        path2 = self.manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            episode=1,
            total_steps=10,
            metrics={"reward": 1.0}
        )
        self.assertIsNotNone(path2)

        # Force save with is_final=True
        path3 = self.manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            episode=1,
            total_steps=15,
            metrics={"reward": 1.0},
            is_final=True
        )
        self.assertIsNotNone(path3)

    def test_max_checkpoints(self):
        """Test that only max_checkpoints are kept."""
        # Save several checkpoints
        for i in range(5):
            self.manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                episode=i+1,
                total_steps=(i+1)*10,
                metrics={"reward": float(i)},
                is_final=True  # Force save
            )

        # Should only keep the 3 most recent (max_checkpoints=3)
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 3)

        # Check that we have the expected number of checkpoints
        checkpoint_names = ",".join(checkpoints)
        # We should have checkpoints for episodes 3, 4, and 5 (the 3 most recent)
        self.assertIn("ep000003", checkpoint_names)
        self.assertIn("ep000004", checkpoint_names)
        self.assertIn("ep000005", checkpoint_names)

    def test_load_latest_checkpoint(self):
        """Test loading the latest checkpoint."""
        # Save multiple checkpoints
        for i in range(3):
            self.manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                episode=i+1,
                total_steps=(i+1)*10,
                metrics={"reward": float(i)},
                is_final=True  # Force save
            )

        # Load the latest checkpoint
        loaded_model = TestModel()
        loaded_optimizer = torch.optim.Adam(loaded_model.parameters())

        _, _, metadata = self.manager.load_latest_checkpoint(
            model=loaded_model,
            optimizer=loaded_optimizer
        )

        # Should have loaded the latest checkpoint (episode 3)
        self.assertEqual(metadata.episode, 3)
        self.assertEqual(metadata.total_steps, 30)
        self.assertEqual(metadata.metrics["reward"], 2.0)

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # No checkpoints initially
        self.assertEqual(len(self.manager.list_checkpoints()), 0)

        # Add some checkpoints
        for i in range(3):
            self.manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                episode=i+1,
                total_steps=(i+1)*10,
                metrics={"reward": float(i)},
                is_final=True
            )

        # Should list all checkpoints
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 3)

        # Check that all expected checkpoints exist (order doesn't matter for this test)
        checkpoint_names = "".join(checkpoints)
        self.assertIn("ep000001", checkpoint_names)
        self.assertIn("ep000002", checkpoint_names)
        self.assertIn("ep000003", checkpoint_names)

if __name__ == "__main__":
    unittest.main()
