#!/usr/bin/env python3
"""
Validate complete ADAN setup before starting optimization.
Checks all critical components and configurations.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.adan_trading_bot.common.config_loader import ConfigLoader


class ValidationChecker:
    """Comprehensive validation checker for ADAN setup."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
    
    def check_file_exists(self, path: str, description: str) -> bool:
        """Check if file exists."""
        if Path(path).exists():
            logger.info(f"✅ {description}: {path}")
            self.checks_passed += 1
            return True
        else:
            logger.error(f"❌ {description} NOT FOUND: {path}")
            self.checks_failed += 1
            self.errors.append(f"Missing: {path}")
            return False
    
    def check_directory_exists(self, path: str, description: str) -> bool:
        """Check if directory exists."""
        if Path(path).is_dir():
            logger.info(f"✅ {description}: {path}")
            self.checks_passed += 1
            return True
        else:
            logger.error(f"❌ {description} NOT FOUND: {path}")
            self.checks_failed += 1
            self.errors.append(f"Missing directory: {path}")
            return False
    
    def check_directory_empty(self, path: str, description: str) -> bool:
        """Check if directory is empty."""
        if not Path(path).exists():
            logger.warning(f"⚠️  {description} does not exist: {path}")
            return False
        
        items = list(Path(path).iterdir())
        if not items:
            logger.info(f"✅ {description} is empty: {path}")
            self.checks_passed += 1
            return True
        else:
            logger.warning(f"⚠️  {description} contains {len(items)} items (will be overwritten)")
            self.warnings.append(f"Directory not empty: {path}")
            return False
    
    def check_config_value(self, config_path: str, key: str, expected_value: any, 
                          description: str) -> bool:
        """Check configuration value."""
        try:
            cfg = ConfigLoader.load_config(config_path)
            
            # Access value directly from the loaded config dictionary
            path_parts = key.split('.')
            actual_value = cfg
            for part in path_parts:
                if isinstance(actual_value, dict):
                    actual_value = actual_value.get(part)
                else:
                    actual_value = None # Key not found at this level
                    break
            
            if actual_value == expected_value:
                logger.info(f"✅ {description}: {key} = {actual_value}")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"❌ {description}: {key} = {actual_value} (expected {expected_value})")
                self.checks_failed += 1
                self.errors.append(f"Config mismatch: {key}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to check config: {e}")
            self.checks_failed += 1
            self.errors.append(f"Config error: {str(e)}")
            return False
    
    def check_data_available(self, data_dir: str, description: str) -> bool:
        """Check if data files are available."""
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                logger.error(f"❌ {description}: Data directory not found: {data_dir}")
                self.checks_failed += 1
                self.errors.append(f"Missing data: {data_dir}")
                return False
            
            # Count data files
            files = list(data_path.glob("**/*.parquet")) + list(data_path.glob("**/*.csv"))
            if files:
                logger.info(f"✅ {description}: {len(files)} data files found")
                self.checks_passed += 1
                return True
            else:
                logger.error(f"❌ {description}: No data files found in {data_dir}")
                self.checks_failed += 1
                self.errors.append(f"No data files: {data_dir}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to check data: {e}")
            self.checks_failed += 1
            self.errors.append(f"Data check error: {str(e)}")
            return False
    
    def print_summary(self):
        """Print validation summary."""
        total = self.checks_passed + self.checks_failed
        
        print("
" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"✅ Passed: {self.checks_passed}/{total}")
        print(f"❌ Failed: {self.checks_failed}/{total}")
        
        if self.warnings:
            print(f"
⚠️  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"   - {w}")
        
        if self.errors:
            print(f"
❌ Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"   - {e}")
        
        print("="*70)
        
        if self.checks_failed == 0:
            print("✅ ALL CHECKS PASSED - READY FOR OPTIMIZATION")
            return True
        else:
            print("❌ SOME CHECKS FAILED - FIX ERRORS BEFORE PROCEEDING")
            return False


def main():
    """Run all validation checks."""
    logger.info("Starting ADAN Final Setup Validation...")
    print()
    
    checker = ValidationChecker()
    
    # Phase 1: File Structure
    logger.info("=" * 70)
    logger.info("PHASE 1: FILE STRUCTURE")
    logger.info("=" * 70)
    
    checker.check_file_exists("config/config.yaml", "Main config")
    checker.check_file_exists("scripts/optimize_hyperparams.py", "Optuna script")
    checker.check_file_exists("scripts/train_parallel_agents.py", "Training script")
    checker.check_directory_exists("src/adan_trading_bot", "Source code")
    checker.check_directory_exists("data/processed/indicators", "Data directory")
    
    # Phase 2: Configuration
    logger.info("
" + "=" * 70)
    logger.info("PHASE 2: CONFIGURATION")
    logger.info("=" * 70)
    
    checker.check_config_value(
        "config/config.yaml",
        "environment.initial_balance",
        20.5,
        "Capital initial"
    )
    checker.check_config_value(
        "config/config.yaml",
        "agent.n_envs",
        4,
        "Number of workers"
    )
    checker.check_config_value(
        "config/config.yaml",
        "environment.max_chunks_per_episode",
        10,
        "Max chunks per episode"
    )
    
    # Phase 3: Directories
    logger.info("
" + "=" * 70)
    logger.info("PHASE 3: WORKING DIRECTORIES")
    logger.info("=" * 70)
    
    checker.check_directory_exists("checkpoints", "Checkpoints directory")
    checker.check_directory_empty("checkpoints", "Checkpoints (should be empty)")
    
    checker.check_directory_exists("logs", "Logs directory")
    
    # Phase 4: Data
    logger.info("
" + "=" * 70)
    logger.info("PHASE 4: DATA AVAILABILITY")
    logger.info("=" * 70)
    
    checker.check_data_available(
        "data/processed/indicators",
        "Processed indicators"
    )
    
    # Phase 5: Dependencies
    logger.info("
" + "=" * 70)
    logger.info("PHASE 5: DEPENDENCIES")
    logger.info("=" * 70)
    
    try:
        import optuna
        logger.info(f"✅ Optuna: {optuna.__version__}")
        checker.checks_passed += 1
    except ImportError:
        logger.error("❌ Optuna not installed")
        checker.checks_failed += 1
        checker.errors.append("Missing: optuna")
    
    try:
        import stable_baselines3
        logger.info(f"✅ Stable Baselines 3: {stable_baselines3.__version__}")
        checker.checks_passed += 1
    except ImportError:\
        logger.error("❌ Stable Baselines 3 not installed")
        checker.checks_failed += 1
        checker.errors.append("Missing: stable_baselines3")
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        checker.checks_passed += 1
    except ImportError:
        logger.error("❌ PyTorch not installed")
        checker.checks_failed += 1
        checker.errors.append("Missing: torch")
    
    # Print summary
    success = checker.print_summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
