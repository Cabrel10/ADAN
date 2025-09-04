#!/usr/bin/env python
"""
Test script for worker configuration validations.
"""

import logging
import os
import sys
from pathlib import Path
import yaml

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.common.config_validator import ConfigValidator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_validation():
    """Test configuration validation."""
    validator = ConfigValidator()
    config_dir = Path(__file__).parent.parent / 'config'

    logger.info(f"Testing configuration validation in {config_dir}")

    try:
        valid = validator.validate_all_configs(config_dir)

        # Log results
        if validator.has_errors():
            logger.error("Validation errors found:")
            for error in validator.get_validation_errors():
                logger.error(f"  - {error}")

        if validator.has_warnings():
            logger.warning("Validation warnings found:")
            for warning in validator.get_validation_warnings():
                logger.warning(f"  - {warning}")

        if valid:
            logger.info("All configurations validated successfully!")
        else:
            logger.error("Configuration validation failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error reading configuration file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_config_validation()
