#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration validation utilities for the ADAN trading bot.

This module provides comprehensive validation for all configuration files
to ensure they meet the requirements specified in the design document.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ConfigValidator:
    """
    Validates configuration dictionaries against expected schemas.
    
    This validator ensures that all required configuration parameters are present
    and have valid values according to the ADAN trading bot specifications.
    """
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_all_configs(self, config_dir: Union[str, Path]) -> bool:
        """
        Validate all configuration files in the specified directory.
        
        Args:
            config_dir: Path to the configuration directory
            
        Returns:
            True if all configurations are valid, False otherwise
        """
        config_dir = Path(config_dir)
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        logger.info(f"Validating all configurations in {config_dir}")
        
        # Define required configuration files
        required_configs = {
            'main_config.yaml': self.validate_main_config,
            'data_config.yaml': self.validate_data_config,
            'environment_config.yaml': self.validate_environment_config,
            'train_config.yaml': self.validate_train_config,
            'dbe_config.yaml': self.validate_dbe_config,
            'memory_config.yaml': self.validate_memory_config,
            'risk_config.yaml': self.validate_risk_config
        }
        
        all_valid = True
        
        for config_file, validator_func in required_configs.items():
            config_path = config_dir / config_file
            
            if not config_path.exists():
                self.validation_errors.append(f"Required configuration file not found: {config_file}")
                all_valid = False
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                if not validator_func(config_data, config_file):
                    all_valid = False
                    
            except Exception as e:
                self.validation_errors.append(f"Error loading {config_file}: {str(e)}")
                all_valid = False
        
        # Log validation results
        if self.validation_errors:
            logger.error(f"Configuration validation failed with {len(self.validation_errors)} errors:")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
        
        if self.validation_warnings:
            logger.warning(f"Configuration validation completed with {len(self.validation_warnings)} warnings:")
            for warning in self.validation_warnings:
                logger.warning(f"  - {warning}")
        
        if all_valid:
            logger.info("All configuration files validated successfully")
        
        return all_valid
    
    def validate_main_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate main configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['data', 'agent', 'logging', 'training', 'environment', 'paths']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        # Validate data section
        if 'data' in config:
            data_config = config['data']
            required_data_keys = ['data_dir', 'assets', 'timeframes']
            for key in required_data_keys:
                if key not in data_config:
                    self.validation_errors.append(f"{filename}: Missing required data key '{key}'")
                    valid = False
            
            # Validate timeframes
            if 'timeframes' in data_config:
                expected_timeframes = ['5m', '1h', '4h']
                if data_config['timeframes'] != expected_timeframes:
                    self.validation_warnings.append(
                        f"{filename}: Timeframes {data_config['timeframes']} differ from expected {expected_timeframes}"
                    )
            
            # Validate assets
            if 'assets' in data_config:
                expected_assets = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA']
                if not all(asset in expected_assets for asset in data_config['assets']):
                    self.validation_warnings.append(
                        f"{filename}: Some assets not in expected list {expected_assets}"
                    )
        
        return valid
    
    def validate_data_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate data configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['feature_engineering', 'indicators', 'indicators_by_timeframe']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        # Validate feature engineering
        if 'feature_engineering' in config:
            fe_config = config['feature_engineering']
            
            # Check timeframes
            if 'timeframes' not in fe_config:
                self.validation_errors.append(f"{filename}: Missing timeframes in feature_engineering")
                valid = False
            else:
                expected_timeframes = ['4h', '1h', '5m']  # Order matters for processing
                if fe_config['timeframes'] != expected_timeframes:
                    self.validation_errors.append(
                        f"{filename}: Timeframes must be {expected_timeframes} in processing order"
                    )
                    valid = False
            
            # Check features per timeframe
            if 'features' not in fe_config:
                self.validation_errors.append(f"{filename}: Missing features in feature_engineering")
                valid = False
            else:
                required_features = ['open', 'high', 'low', 'close', 'volume', 'minutes_since_update']
                for tf in ['5m', '1h', '4h']:
                    if tf not in fe_config['features']:
                        self.validation_errors.append(f"{filename}: Missing features for timeframe {tf}")
                        valid = False
                    else:
                        tf_features = fe_config['features'][tf]
                        missing_features = [f for f in required_features if f not in tf_features]
                        if missing_features:
                            self.validation_errors.append(
                                f"{filename}: Missing required features for {tf}: {missing_features}"
                            )
                            valid = False
        
        return valid
    
    def validate_environment_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate environment configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['trading_rules', 'capital_tiers', 'reward_shaping']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        # Validate trading rules
        if 'trading_rules' in config:
            tr_config = config['trading_rules']
            required_keys = ['min_order_value_usdt', 'commission_pct', 'slippage_pct']
            for key in required_keys:
                if key not in tr_config:
                    self.validation_errors.append(f"{filename}: Missing trading rule '{key}'")
                    valid = False
        
        # Validate capital tiers
        if 'capital_tiers' in config:
            tiers = config['capital_tiers']
            if not isinstance(tiers, list) or len(tiers) == 0:
                self.validation_errors.append(f"{filename}: capital_tiers must be a non-empty list")
                valid = False
            else:
                # Check that tiers are in ascending order
                thresholds = [tier.get('threshold', 0) for tier in tiers]
                if thresholds != sorted(thresholds):
                    self.validation_errors.append(f"{filename}: capital_tiers must be in ascending threshold order")
                    valid = False
        
        return valid
    
    def validate_train_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate training configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required parameters
        required_params = ['total_timesteps', 'batch_size', 'learning_rate', 'n_envs']
        for param in required_params:
            if param not in config:
                self.validation_errors.append(f"{filename}: Missing required parameter '{param}'")
                valid = False
        
        # Validate memory-optimized settings
        if 'n_envs' in config and config['n_envs'] != 1:
            self.validation_warnings.append(
                f"{filename}: n_envs should be 1 for memory optimization (current: {config['n_envs']})"
            )
        
        if 'batch_size' in config and config['batch_size'] > 64:
            self.validation_warnings.append(
                f"{filename}: Large batch_size may cause memory issues (current: {config['batch_size']})"
            )
        
        return valid
    
    def validate_dbe_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate Dynamic Behavior Engine configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['risk_parameters', 'reward', 'learning']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        return valid
    
    def validate_memory_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate memory configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['hardware_constraints', 'chunk_loader', 'memory_monitoring']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        # Validate hardware constraints
        if 'hardware_constraints' in config:
            hw_config = config['hardware_constraints']
            required_keys = ['total_ram_gb', 'training_ram_gb', 'cpu_cores']
            for key in required_keys:
                if key not in hw_config:
                    self.validation_errors.append(f"{filename}: Missing hardware constraint '{key}'")
                    valid = False
        
        # Validate chunk loader settings
        if 'chunk_loader' in config:
            cl_config = config['chunk_loader']
            
            # Check single chunk strategy
            if cl_config.get('max_chunks_in_memory', 1) != 1:
                self.validation_warnings.append(
                    f"{filename}: max_chunks_in_memory should be 1 for memory optimization"
                )
            
            # Check aggressive cleanup
            if not cl_config.get('aggressive_cleanup', False):
                self.validation_warnings.append(
                    f"{filename}: aggressive_cleanup should be enabled for memory optimization"
                )
        
        return valid
    
    def validate_risk_config(self, config: Dict[str, Any], filename: str) -> bool:
        """Validate risk management configuration."""
        logger.debug(f"Validating {filename}")
        valid = True
        
        # Required sections
        required_sections = ['dbe_settings', 'risk_metrics', 'position_sizing']
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"{filename}: Missing required section '{section}'")
                valid = False
        
        return valid
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()
    
    def get_validation_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.validation_warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if there are validation errors."""
        return len(self.validation_errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are validation warnings."""
        return len(self.validation_warnings) > 0


def validate_config_directory(config_dir: Union[str, Path]) -> bool:
    """
    Convenience function to validate all configurations in a directory.
    
    Args:
        config_dir: Path to the configuration directory
        
    Returns:
        True if all configurations are valid, False otherwise
    """
    validator = ConfigValidator()
    return validator.validate_all_configs(config_dir)


def validate_single_config(config_path: Union[str, Path], config_type: str) -> bool:
    """
    Validate a single configuration file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration ('main', 'data', 'environment', etc.)
        
    Returns:
        True if configuration is valid, False otherwise
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        validator = ConfigValidator()
        
        # Map config types to validation methods
        validation_methods = {
            'main': validator.validate_main_config,
            'data': validator.validate_data_config,
            'environment': validator.validate_environment_config,
            'train': validator.validate_train_config,
            'dbe': validator.validate_dbe_config,
            'memory': validator.validate_memory_config,
            'risk': validator.validate_risk_config
        }
        
        if config_type not in validation_methods:
            logger.error(f"Unknown configuration type: {config_type}")
            return False
        
        return validation_methods[config_type](config_data, config_path.name)
        
    except Exception as e:
        logger.error(f"Error validating {config_path}: {str(e)}")
        return False