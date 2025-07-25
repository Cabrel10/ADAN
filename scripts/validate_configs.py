#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to validate all configuration files for the ADAN trading bot.

This script checks that all configuration files are present and contain
the required parameters according to the design specifications.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from adan_trading_bot.common.config_validator import validate_config_directory
except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to basic validation...")
    
    import yaml
    
    def basic_validate_config_directory(config_dir):
        """Basic configuration validation without dependencies."""
        config_dir = Path(config_dir)
        
        required_files = [
            'main_config.yaml',
            'data_config.yaml', 
            'environment_config.yaml',
            'train_config.yaml',
            'dbe_config.yaml',
            'memory_config.yaml',
            'risk_config.yaml'
        ]
        
        all_valid = True
        errors = []
        
        for config_file in required_files:
            config_path = config_dir / config_file
            
            if not config_path.exists():
                errors.append(f"Missing required configuration file: {config_file}")
                all_valid = False
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    if not config_data:
                        errors.append(f"Empty configuration file: {config_file}")
                        all_valid = False
                    else:
                        print(f"✓ {config_file} - loaded successfully")
            except Exception as e:
                errors.append(f"Error loading {config_file}: {str(e)}")
                all_valid = False
        
        if errors:
            print("\nValidation errors:")
            for error in errors:
                print(f"  - {error}")
        
        return all_valid
    
    validate_config_directory = basic_validate_config_directory

def main():
    """Main validation function."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    config_dir = Path(__file__).parent.parent / 'config'
    
    print(f"Validating configurations in: {config_dir}")
    print("=" * 50)
    
    result = validate_config_directory(config_dir)
    
    print("=" * 50)
    if result:
        print("✅ All configuration files validated successfully!")
        return 0
    else:
        print("❌ Configuration validation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())