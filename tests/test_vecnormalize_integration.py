#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VecNormalize Integration Test

Verifies that:
1. vecnormalize.pkl is saved after training
2. RNG states are persisted
3. Stats can be loaded and inspected
"""

import os
import sys
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_vecnormalize_persistence(checkpoint_dir="models/rl_agents"):
    """Test that VecNormalize stats are saved correctly."""
    vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
    
    logger.info("=" * 60)
    logger.info("VecNormalize Persistence Test")
    logger.info("=" * 60)
    
    # Check file exists
    if not os.path.exists(vec_path):
        logger.error(f"❌ vecnormalize.pkl not found at {vec_path}")
        return False
    
    file_size = os.path.getsize(vec_path)
    logger.info(f"✅ vecnormalize.pkl exists ({file_size:,} bytes)")
    
    if file_size == 0:
        logger.error("❌ vecnormalize.pkl is empty!")
        return False
    
    # Try to load using pickle (basic check)
    try:
        import pickle
        with open(vec_path, 'rb') as f:
            vec_data = pickle.load(f)
        logger.info(f"✅ vecnormalize.pkl is valid pickle file")
        
        # Check for expected keys
        if hasattr(vec_data, 'obs_rms'):
            logger.info(f"✅ obs_rms found (observation normalization stats)")
        if hasattr(vec_data, 'ret_rms'):
            logger.info(f"✅ ret_rms found (return normalization stats)")
            
    except Exception as e:
        logger.error(f"❌ Failed to load vecnormalize.pkl: {e}")
        return False
    
    return True


def test_rng_states(checkpoint_dir="models/rl_agents"):
    """Test that RNG states are saved."""
    rng_path = os.path.join(checkpoint_dir, "rng_states.json")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("RNG States Persistence Test")
    logger.info("=" * 60)
    
    if not os.path.exists(rng_path):
        logger.error(f"❌ rng_states.json not found at {rng_path}")
        return False
    
    logger.info(f"✅ rng_states.json exists")
    
    try:
        with open(rng_path, 'r') as f:
            states = json.load(f)
        
        if 'seed' not in states:
            logger.error("❌ Seed not found in RNG states")
            return False
        
        logger.info(f"✅ RNG states loaded (seed={states['seed']})")
        
        # Check for other expected keys
        if 'python_random_state' in states:
            logger.info(f"✅ Python random state saved")
        if 'numpy_random_state' in states:
            logger.info(f"✅ NumPy random state saved")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load rng_states.json: {e}")
        return False


def test_model_checkpoints(checkpoint_dir="models/rl_agents/final"):
    """Test that model checkpoints exist."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Model Checkpoints Test")
    logger.info("=" * 60)
    
    if not os.path.exists(checkpoint_dir):
        logger.error(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    logger.info(f"✅ Checkpoint directory exists: {checkpoint_dir}")
    
    # Look for worker models
    workers = ["w1", "w2", "w3", "w4"]
    found_models = []
    
    for worker in workers:
        model_path = os.path.join(checkpoint_dir, f"{worker}_final.zip")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            logger.info(f"✅ {worker}_final.zip found ({size:,} bytes)")
            found_models.append(worker)
        else:
            logger.warning(f"⚠️ {worker}_final.zip not found")
    
    if not found_models:
        logger.error("❌ No model checkpoints found!")
        return False
    
    logger.info(f"✅ Found {len(found_models)}/4 worker models")
    return len(found_models) > 0


def main():
    """Run all integration tests."""
    logger.info("\n")
    logger.info("🧪 ADAN 2.0 - VecNormalize Integration Tests")
    logger.info("\n")
    
    results = {
        "vecnormalize": test_vecnormalize_persistence(),
        "rng_states": test_rng_states(),
        "model_checkpoints": test_model_checkpoints(),
    }
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("")
        logger.info("VecNormalize integration is working correctly.")
        logger.info("You can now:")
        logger.info("  1. Run backtests using frozen normalization stats")
        logger.info("  2. Resume training from checkpoints")
        logger.info("  3. Proceed to Phase 3.2 (Optuna Optimization)")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("")
        logger.error("Action required:")
        logger.error("  1. Run a short training session (5K steps)")
        logger.error("  2. Check train_parallel_agents.py logs for errors")
        logger.error("  3. Verify VecNormalize wrapper is active")
        sys.exit(1)


if __name__ == "__main__":
    main()
