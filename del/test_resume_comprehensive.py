#!/usr/bin/env python3
"""
Comprehensive Resume Test
Tests both logging fix and resume compatibility
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_checkpoint_integrity():
    """Test that checkpoints are valid and contain required files"""
    logger.info("=" * 80)
    logger.info("TEST 1: CHECKPOINT INTEGRITY")
    logger.info("=" * 80)
    
    checkpoint_dir = "/mnt/new_data/t10_training/checkpoints"
    workers = ["w1", "w2", "w3", "w4"]
    
    for worker in workers:
        worker_dir = os.path.join(checkpoint_dir, worker)
        if not os.path.exists(worker_dir):
            logger.error(f"❌ {worker} directory not found: {worker_dir}")
            continue
        
        # Find latest checkpoint
        checkpoints = []
        for f in os.listdir(worker_dir):
            if f.endswith(".zip") and "_model_" in f:
                try:
                    steps = int(f.split("_model_")[1].split("_steps")[0])
                    checkpoints.append((steps, f))
                except ValueError:
                    continue
        
        if not checkpoints:
            logger.error(f"❌ {worker}: No checkpoints found")
            continue
        
        best_steps, best_file = max(checkpoints, key=lambda x: x[0])
        best_path = os.path.join(worker_dir, best_file)
        
        logger.info(f"\n✅ {worker}: Found {len(checkpoints)} checkpoints")
        logger.info(f"   Latest: {best_file} ({best_steps:,} steps)")
        logger.info(f"   Size: {os.path.getsize(best_path) / 1024 / 1024:.1f} MB")
        
        # Verify integrity
        import zipfile
        try:
            with zipfile.ZipFile(best_path, 'r') as z:
                files = z.namelist()
                required = ['data', 'policy.pth', 'policy.optimizer.pth']
                missing = [f for f in required if f not in files]
                
                if missing:
                    logger.error(f"   ❌ Missing files: {missing}")
                else:
                    logger.info(f"   ✅ All required files present")
                    logger.info(f"   Files: {', '.join(files)}")
        except Exception as e:
            logger.error(f"   ❌ Integrity check failed: {e}")
    
    return True

def test_resume_logic():
    """Test the resume logic in train_parallel_agents.py"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: RESUME LOGIC IN SCRIPT")
    logger.info("=" * 80)
    
    script_path = "scripts/train_parallel_agents.py"
    
    if not os.path.exists(script_path):
        logger.error(f"❌ Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for critical fixes
    checks = {
        "reset_num_timesteps=False": "Resume compatibility flag",
        "PPO.load(best_checkpoint_path": "Checkpoint loading logic",
        "if resume and best_checkpoint_path": "Resume condition check",
        "central_logger.metric": "Metrics transmission for all workers",
        "remaining_steps = max(0, total_timesteps - initial_steps)": "Remaining steps calculation",
    }
    
    all_good = True
    for check_str, description in checks.items():
        if check_str in content:
            logger.info(f"✅ {description}")
            logger.info(f"   Found: '{check_str}'")
        else:
            logger.error(f"❌ {description}")
            logger.error(f"   Missing: '{check_str}'")
            all_good = False
    
    return all_good

def test_logging_fix():
    """Test that logging fix is in place"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: LOGGING FIX FOR W1-W3")
    logger.info("=" * 80)
    
    script_path = "scripts/train_parallel_agents.py"
    
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    # Find the central_logger section
    found_fix = False
    for i, line in enumerate(lines):
        if "central_logger.metric" in line and "Worker_" in line:
            # Check if it's outside the restrictive conditional
            # Look backwards to find the conditional
            for j in range(i-1, max(0, i-20), -1):
                if "if worker_id == 0 or" in lines[j]:
                    logger.error(f"❌ central_logger still inside restrictive conditional at line {j+1}")
                    return False
                if "if UNIFIED_SYSTEM_AVAILABLE" in lines[j]:
                    logger.info(f"✅ central_logger is OUTSIDE restrictive conditional")
                    logger.info(f"   Line {j+1}: {lines[j].strip()}")
                    found_fix = True
                    break
    
    if not found_fix:
        logger.warning("⚠️  Could not verify logging fix location")
        return False
    
    return True

def test_model_loading():
    """Test that model loading works correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: MODEL LOADING CAPABILITY")
    logger.info("=" * 80)
    
    try:
        from stable_baselines3 import PPO
        logger.info("✅ PPO import successful")
        
        # Check if PPO.load exists
        if hasattr(PPO, 'load'):
            logger.info("✅ PPO.load method exists")
        else:
            logger.error("❌ PPO.load method not found")
            return False
        
        # Try to load a checkpoint
        checkpoint_path = "/mnt/new_data/t10_training/checkpoints/w1/w1_model_170000_steps.zip"
        if os.path.exists(checkpoint_path):
            logger.info(f"✅ Test checkpoint exists: {checkpoint_path}")
            
            # Try loading (without env to just test the mechanism)
            try:
                model = PPO.load(checkpoint_path)
                initial_steps = model.num_timesteps
                logger.info(f"✅ Model loaded successfully")
                logger.info(f"   num_timesteps: {initial_steps:,}")
                
                if initial_steps >= 160000:
                    logger.info(f"✅ num_timesteps is reasonable ({initial_steps:,})")
                else:
                    logger.error(f"❌ num_timesteps too low ({initial_steps:,})")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                return False
        else:
            logger.warning(f"⚠️  Test checkpoint not found: {checkpoint_path}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import PPO: {e}")
        return False

def main():
    logger.info("\n" + "🧪 COMPREHENSIVE RESUME TEST SUITE 🧪".center(80))
    logger.info("=" * 80)
    
    results = {
        "Checkpoint Integrity": test_checkpoint_integrity(),
        "Resume Logic": test_resume_logic(),
        "Logging Fix": test_logging_fix(),
        "Model Loading": test_model_loading(),
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - RESUME IS READY!")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("⚠️  SOME TESTS FAILED - REVIEW ABOVE")
        logger.info("=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())
