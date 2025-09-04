#!/usr/bin/env python3
"""
Integration test for memory management optimizations.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.common.memory_manager import MemoryManager, MemoryPressureLevel


def test_memory_integration():
    """Test memory management integration."""
    print("🧪 Testing Memory Management Integration...")

    config = {
        'medium_threshold': 70.0,
        'high_threshold': 85.0,
        'critical_threshold': 95.0,
        'enable_monitoring': True,
        'monitoring_interval': 2.0,
        'enable_auto_gc': True,
        'enable_gpu_monitoring': True,
        'enable_mixed_precision': True,
        'enable_amp': True
    }

    try:
        # Initialize memory manager
        memory_manager = MemoryManager(config)
        print("✅ Memory Manager initialized")

        # Test basic functionality
        stats = memory_manager.get_memory_stats()
        print(f"  📊 Current memory usage: {stats.memory_percent:.1f}%")

        # Test training optimization
        print("\n🚀 Testing training optimization...")
        original_interval = memory_manager.optimize_for_training()

        # Simulate some training operations
        with memory_manager.create_mixed_precision_context():
            # Create some tensors
            tensors = []
            for i in range(5):
                tensor = torch.randn(500, 500)
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                tensors.append(tensor)

            # Simulate computation
            result = sum(t.sum() for t in tensors)
            print(f"  ✅ Training simulation completed: {result}")

        # Test memory cleanup
        print("\n🧹 Testing memory cleanup...")
        memory_manager.trigger_garbage_collection(force=True)

        # Get final stats
        final_stats = memory_manager.get_memory_stats()
        print(f"  📊 Final memory usage: {final_stats.memory_percent:.1f}%")

        # Test memory summary
        summary = memory_manager.get_memory_summary()
        print(f"  📋 Memory summary generated: {len(summary)} sections")

        # Restore settings
        memory_manager.restore_monitoring_interval(original_interval)

        # Cleanup
        memory_manager.shutdown()

        print("✅ Memory management integration test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run memory integration test."""
    print("🚀 Starting Memory Management Integration Test...\n")

    success = test_memory_integration()

    if success:
        print("\n🎉 Memory management integration test PASSED!")
        return True
    else:
        print("\n⚠️  Memory management integration test FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
