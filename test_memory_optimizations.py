#!/usr/bin/env python3
"""
Test script to verify memory management optimizations.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
import psutil
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adan_trading_bot.common.memory_manager import MemoryManager, MemoryPressureLevel, memory_profile
from adan_trading_bot.model.custom_cnn import CustomCNN
import gymnasium as gym


def create_test_observation_space():
    """Create a test observation space for CNN."""
    return gym.spaces.Box(
        low=0, high=255,
        shape=(4, 32, 32),  # 4 channels, 32x32 image
        dtype=np.uint8
    )


@memory_profile
def test_memory_intensive_operation():
    """Test function with memory profiling."""
    # Create some tensors to use memory
    tensors = []
    for i in range(10):
        tensor = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        tensors.append(tensor)

    # Simulate some computation
    result = sum(t.sum() for t in tensors)
    return result


def test_memory_manager():
    """Test MemoryManager functionality."""
    print("🧪 Testing Memory Manager...")

    config = {
        'medium_threshold': 60.0,
        'high_threshold': 80.0,
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

        # Test memory statistics
        stats = memory_manager.get_memory_stats()
        print(f"  📊 Current memory usage: {stats.memory_percent:.1f}%")
        print(f"  💾 Available memory: {stats.available_memory:.1f}GB")

        if stats.gpu_memory_percent:
            print(f"  🎮 GPU memory usage: {stats.gpu_memory_percent:.1f}%")

        # Test pressure level detection
        pressure = memory_manager.get_pressure_level(stats.memory_percent)
        print(f"  📈 Current pressure level: {pressure.value}")

        # Test garbage collection
        print("\n🧹 Testing garbage collection...")
        collected = memory_manager.trigger_garbage_collection()
        print(f"  ✅ Garbage collection completed: {collected}")

        # Test memory summary
        print("\n📋 Testing memory summary...")
        summary = memory_manager.get_memory_summary()
        print(f"  📊 Current usage: {summary['current']['memory_percent']:.1f}%")
        print(f"  🎯 Thresholds: {summary['thresholds']}")

        # Test mixed precision context
        print("\n🔄 Testing mixed precision context...")
        with memory_manager.create_mixed_precision_context():
            if torch.cuda.is_available():
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                result = torch.mm(x, y)
                print(f"  ✅ Mixed precision operation completed: {result.shape}")
            else:
                print("  ⚠️  CUDA not available, skipping GPU test")

        # Test memory optimization for training
        print("\n🚀 Testing training optimization...")
        original_interval = memory_manager.optimize_for_training()
        print(f"  ✅ Training optimization applied, interval: {memory_manager.monitoring_interval}s")

        # Restore interval
        memory_manager.restore_monitoring_interval(original_interval)
        print(f"  ✅ Monitoring interval restored: {memory_manager.monitoring_interval}s")

        # Test memory profiling decorator
        print("\n📊 Testing memory profiling...")
        result = test_memory_intensive_operation()
        print(f"  ✅ Memory profiled operation completed: {result}")

        # Test pressure callback
        print("\n🔔 Testing pressure callbacks...")
        callback_triggered = False

        def test_callback(level):
            nonlocal callback_triggered
            callback_triggered = True
            print(f"    🚨 Pressure callback triggered for level: {level.value}")

        memory_manager.register_pressure_callback(MemoryPressureLevel.MEDIUM, test_callback)
        print("  ✅ Pressure callback registered")

        # Wait a bit for monitoring
        time.sleep(3)

        # Cleanup
        memory_manager.shutdown()
        print("✅ Memory Manager tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Memory Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_cnn_memory_optimizations():
    """Test CustomCNN memory optimizations."""
    print("\n🧪 Testing CustomCNN Memory Optimizations...")

    try:
        # Create observation space
        obs_space = create_test_observation_space()

        # Test with memory optimizations enabled
        memory_config = {
            'enable_memory_efficient': True,
            'aggressive_cleanup': False,
            'enable_mixed_precision': True,
            'enable_gradient_checkpointing': True
        }

        cnn = CustomCNN(
            observation_space=obs_space,
            features_dim=128,
            memory_config=memory_config
        )
        print("✅ CustomCNN initialized with memory optimizations")

        # Test memory usage reporting
        if torch.cuda.is_available():
            cnn = cnn.cuda()
            memory_usage = cnn.get_memory_usage()
            print(f"  📊 Model parameters: {memory_usage['model_parameters_mb']:.1f}MB")
            print(f"  🎮 GPU allocated: {memory_usage['gpu_allocated_mb']:.1f}MB")
            print(f"  ⚙️  Memory efficient: {memory_usage['memory_efficient']}")
            print(f"  🔄 Mixed precision: {memory_usage['mixed_precision']}")
        else:
            print("  ⚠️  CUDA not available, skipping GPU memory tests")

        # Test forward pass with memory optimizations
        print("\n🔄 Testing forward pass with optimizations...")
        batch_size = 4
        test_input = torch.randn(batch_size, *obs_space.shape)

        if torch.cuda.is_available():
            test_input = test_input.cuda()

        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / (1024**2)  # MB

        # Forward pass
        with torch.no_grad():
            features = cnn(test_input)

        # Measure memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated() / (1024**2)  # MB
            memory_diff = memory_after - memory_before
            print(f"  📊 Forward pass completed: {features.shape}")
            print(f"  💾 Memory change: {memory_diff:.1f}MB")
        else:
            print(f"  📊 Forward pass completed: {features.shape}")

        # Test mixed precision toggle
        print("\n🔄 Testing mixed precision toggle...")
        cnn.enable_mixed_precision()
        cnn.disable_mixed_precision()
        print("  ✅ Mixed precision toggle working")

        # Test inference optimization
        print("\n🚀 Testing inference optimization...")
        cnn.optimize_for_inference()
        print("  ✅ Inference optimization applied")

        # Test memory cleanup
        print("\n🧹 Testing memory cleanup...")
        cnn.cleanup_memory()
        print("  ✅ Memory cleanup completed")

        # Test attention map with memory efficiency
        print("\n🎯 Testing attention map generation...")
        with torch.no_grad():
            attention_map = cnn.get_attention_map(test_input)
            print(f"  ✅ Attention map generated: {attention_map.shape}")

        print("✅ CustomCNN memory optimization tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ CustomCNN memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_pressure_simulation():
    """Test memory pressure handling by simulating high memory usage."""
    print("\n🧪 Testing Memory Pressure Simulation...")

    config = {
        'medium_threshold': 50.0,  # Lower thresholds for testing
        'high_threshold': 60.0,
        'critical_threshold': 70.0,
        'enable_monitoring': True,
        'monitoring_interval': 1.0,
        'enable_auto_gc': True
    }

    try:
        memory_manager = MemoryManager(config)

        # Track callback triggers
        callbacks_triggered = []

        def pressure_callback(level):
            callbacks_triggered.append(level)
            print(f"    🚨 Pressure callback: {level.value}")

        # Register callbacks for all levels
        for level in [MemoryPressureLevel.MEDIUM, MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            memory_manager.register_pressure_callback(level, pressure_callback)

        print("  ✅ Pressure callbacks registered")

        # Simulate memory allocation (be careful not to crash the system)
        print("  📈 Simulating memory allocation...")
        memory_hogs = []

        try:
            # Allocate memory gradually
            for i in range(5):
                # Allocate 100MB chunks
                chunk = np.random.randn(100 * 1024 * 1024 // 8)  # 100MB
                memory_hogs.append(chunk)

                # Check current memory
                stats = memory_manager.get_memory_stats()
                print(f"    📊 Memory usage: {stats.memory_percent:.1f}%")

                # Wait for monitoring to detect changes
                time.sleep(2)

                # Stop if we're getting too high
                if stats.memory_percent > 80:
                    print("    ⚠️  Stopping allocation to prevent system issues")
                    break

        except MemoryError:
            print("    ⚠️  Memory allocation limit reached")

        # Clean up allocated memory
        print("  🧹 Cleaning up allocated memory...")
        memory_hogs.clear()
        memory_manager.trigger_garbage_collection(force=True)

        # Final stats
        final_stats = memory_manager.get_memory_stats()
        print(f"  📊 Final memory usage: {final_stats.memory_percent:.1f}%")
        print(f"  🔔 Callbacks triggered: {len(callbacks_triggered)}")

        memory_manager.shutdown()
        print("✅ Memory pressure simulation completed!")
        return True

    except Exception as e:
        print(f"❌ Memory pressure simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all memory optimization tests."""
    print("🚀 Starting Memory Management Optimization Tests...\n")

    tests = [
        ("Memory Manager", test_memory_manager),
        ("CustomCNN Memory Optimizations", test_custom_cnn_memory_optimizations),
        ("Memory Pressure Simulation", test_memory_pressure_simulation)
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)

        success = test_func()
        results.append((test_name, success))

        if success:
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All memory optimization tests completed successfully!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
