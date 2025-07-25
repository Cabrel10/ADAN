#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for batch loading configuration with memory optimizations.

This script validates that the batch loading configurations work correctly
with the memory constraints and optimization settings.
"""

import sys
import logging
import yaml
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_batch_configurations():
    """Test that batch configurations are properly optimized for memory."""
    config_dir = Path(__file__).parent.parent / 'config'
    
    print("Testing Batch Loading Configurations")
    print("=" * 50)
    
    # Test training configuration
    train_config_path = config_dir / 'train_config.yaml'
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    print("Training Configuration:")
    print(f"  - Batch size: {train_config.get('batch_size', 'Not set')}")
    print(f"  - Number of environments: {train_config.get('n_envs', 'Not set')}")
    print(f"  - Frame stack enabled: {train_config.get('use_frame_stack', 'Not set')}")
    print(f"  - Memory efficient training: {train_config.get('memory_efficient_training', 'Not set')}")
    print(f"  - Gradient accumulation steps: {train_config.get('gradient_accumulation_steps', 'Not set')}")
    
    # Validate training config
    batch_size = train_config.get('batch_size', 64)
    n_envs = train_config.get('n_envs', 1)
    use_frame_stack = train_config.get('use_frame_stack', True)
    
    issues = []
    if batch_size > 32:
        issues.append(f"Batch size {batch_size} may be too large for memory optimization (recommended: ≤32)")
    if n_envs > 1:
        issues.append(f"Multiple environments ({n_envs}) may cause memory issues (recommended: 1)")
    if use_frame_stack:
        issues.append("Frame stacking enabled may increase memory usage")
    
    if issues:
        print("  ⚠️  Training config issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Training config optimized for memory")
    
    print()
    
    # Test data loading configuration
    data_config_path = config_dir / 'data_config_cpu.yaml'
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("Data Loading Configuration:")
    data_loader_config = data_config.get('data_loader', {})
    chunk_config = data_loader_config.get('chunk_config', {})
    memory_opts = data_config.get('memory_optimizations', {})
    
    print(f"  - Data loader batch size: {data_loader_config.get('batch_size', 'Not set')}")
    print(f"  - Number of workers: {data_loader_config.get('num_workers', 'Not set')}")
    print(f"  - Chunk size: {chunk_config.get('chunk_size', 'Not set')}")
    print(f"  - Max chunks in memory: {chunk_config.get('max_chunks_in_memory', 'Not set')}")
    print(f"  - Aggressive cleanup: {chunk_config.get('aggressive_cleanup', 'Not set')}")
    print(f"  - Caching disabled: {memory_opts.get('disable_caching', 'Not set')}")
    
    # Validate data config
    data_batch_size = data_loader_config.get('batch_size', 64)
    num_workers = data_loader_config.get('num_workers', 4)
    max_chunks = chunk_config.get('max_chunks_in_memory', 3)
    
    data_issues = []
    if data_batch_size > 32:
        data_issues.append(f"Data batch size {data_batch_size} may be too large (recommended: ≤32)")
    if num_workers > 1:
        data_issues.append(f"Multiple workers ({num_workers}) may cause memory overhead (recommended: 1)")
    if max_chunks > 1:
        data_issues.append(f"Multiple chunks in memory ({max_chunks}) defeats single-chunk strategy (recommended: 1)")
    
    if data_issues:
        print("  ⚠️  Data config issues:")
        for issue in data_issues:
            print(f"    - {issue}")
    else:
        print("  ✅ Data config optimized for memory")
    
    print()
    
    # Test memory configuration
    memory_config_path = config_dir / 'memory_config.yaml'
    if memory_config_path.exists():
        with open(memory_config_path, 'r') as f:
            memory_config = yaml.safe_load(f)
        
        print("Memory Configuration:")
        hw_constraints = memory_config.get('hardware_constraints', {})
        chunk_loader = memory_config.get('chunk_loader', {})
        
        print(f"  - Total RAM: {hw_constraints.get('total_ram_gb', 'Not set')} GB")
        print(f"  - Training RAM: {hw_constraints.get('training_ram_gb', 'Not set')} GB")
        print(f"  - CPU cores: {hw_constraints.get('cpu_cores', 'Not set')}")
        print(f"  - Max chunks in memory: {chunk_loader.get('max_chunks_in_memory', 'Not set')}")
        print(f"  - Aggressive cleanup: {chunk_loader.get('aggressive_cleanup', 'Not set')}")
        print(f"  - Force GC: {chunk_loader.get('force_gc', 'Not set')}")
        
        # Check consistency
        memory_issues = []
        if chunk_loader.get('max_chunks_in_memory', 1) != 1:
            memory_issues.append("Memory config should enforce single chunk strategy")
        if not chunk_loader.get('aggressive_cleanup', False):
            memory_issues.append("Aggressive cleanup should be enabled")
        if not chunk_loader.get('force_gc', False):
            memory_issues.append("Force garbage collection should be enabled")
        
        if memory_issues:
            print("  ⚠️  Memory config issues:")
            for issue in memory_issues:
                print(f"    - {issue}")
        else:
            print("  ✅ Memory config properly configured")
    else:
        print("Memory Configuration: ❌ Not found")
    
    print()
    print("=" * 50)
    
    # Overall assessment
    total_issues = len(issues) + len(data_issues) + (len(memory_issues) if 'memory_issues' in locals() else 0)
    if total_issues == 0:
        print("✅ All batch loading configurations are optimized for memory!")
        return True
    else:
        print(f"⚠️  Found {total_issues} configuration issues that may affect memory performance")
        return False

def test_memory_calculation():
    """Test memory usage calculations for different batch sizes."""
    print("\nMemory Usage Estimation")
    print("=" * 30)
    
    # Rough estimates based on typical data sizes
    # These are approximations for planning purposes
    
    timeframes = 3  # 5m, 1h, 4h
    features_per_tf = 6  # OHLCV + minutes_since_update
    window_size = 100
    float_size = 8  # bytes for float64
    
    print("Estimated memory usage per batch:")
    for batch_size in [16, 32, 64, 128]:
        # Memory for one batch of observations
        obs_memory_mb = (batch_size * timeframes * window_size * features_per_tf * float_size) / (1024 * 1024)
        
        # Add overhead for gradients, optimizer states, etc. (rough 3x multiplier)
        total_memory_mb = obs_memory_mb * 3
        
        print(f"  - Batch size {batch_size:3d}: ~{obs_memory_mb:.1f} MB (obs) + overhead = ~{total_memory_mb:.1f} MB total")
    
    print("\nRecommendation: Use batch_size ≤ 32 for 7GB memory constraint")

def main():
    """Main test function."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    success = test_batch_configurations()
    test_memory_calculation()
    
    if success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())