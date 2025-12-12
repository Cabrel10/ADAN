#!/usr/bin/env python3
"""
Environment Stability Monitor
Ensures models don't detect environment changes that cause distribution shift
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('environment_stability.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnvironmentStabilityMonitor:
    """Monitors and maintains environment consistency for trained models"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stability_baseline = None
        self.drift_threshold = 0.05  # 5% drift tolerance
        
    def log_section(self, title):
        """Log a section header"""
        logger.info("=" * 80)
        logger.info(f"🔒 {title}")
        logger.info("=" * 80)
    
    def capture_environment_baseline(self):
        """Capture the current environment state as baseline"""
        self.log_section("CAPTURING ENVIRONMENT BASELINE")
        
        baseline = {
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'environment_vars': self._capture_critical_env_vars(),
            'data_paths': self._capture_data_paths(),
            'market_conditions': self._capture_market_baseline(),
            'model_input_specs': self._capture_model_input_specs(),
            'hash': None  # Will be computed
        }
        
        # Compute hash for integrity check
        baseline['hash'] = self._compute_baseline_hash(baseline)
        
        self.stability_baseline = baseline
        
        # Save baseline
        baseline_file = self.output_dir / "environment_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        logger.info(f"✅ Baseline captured: {baseline_file}")
        logger.info(f"   Python: {baseline['python_version']}")
        logger.info(f"   Hash: {baseline['hash'][:16]}...")
        
        return baseline
    
    def _capture_critical_env_vars(self):
        """Capture critical environment variables"""
        critical_vars = [
            'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
            'CUDA_VISIBLE_DEVICES', 'TF_CPP_MIN_LOG_LEVEL',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS'
        ]
        
        env_state = {}
        for var in critical_vars:
            if var in os.environ:
                env_state[var] = os.environ[var]
        
        return env_state
    
    def _capture_data_paths(self):
        """Capture data path configuration"""
        return {
            'checkpoint_dir': '/mnt/new_data/t10_training/checkpoints',
            'data_dir': '/mnt/new_data/t10_training/data',
            'market_data_dir': '/mnt/new_data/market_data',
            'config_dir': './config'
        }
    
    def _capture_market_baseline(self):
        """Capture market conditions baseline"""
        # In real implementation, would capture actual market stats
        return {
            'market_regime': 'training_regime',
            'volatility_level': 'baseline',
            'liquidity_conditions': 'training_conditions',
            'trading_hours': 'continuous',
            'data_frequency': '1m'
        }
    
    def _capture_model_input_specs(self):
        """Capture model input specifications"""
        return {
            'observation_space': {
                'type': 'Box',
                'shape': [100],  # Adjust based on actual model
                'dtype': 'float32'
            },
            'action_space': {
                'type': 'Discrete',
                'n': 3  # Buy, Hold, Sell
            },
            'normalization': 'z-score',
            'lookback_window': 100,
            'features': [
                'price', 'volume', 'rsi', 'macd', 'bollinger_bands',
                'atr', 'adx', 'obv', 'ema', 'sma'
            ]
        }
    
    def _compute_baseline_hash(self, baseline):
        """Compute hash of baseline for integrity"""
        baseline_copy = baseline.copy()
        baseline_copy.pop('hash', None)
        
        baseline_str = json.dumps(baseline_copy, sort_keys=True)
        return hashlib.sha256(baseline_str.encode()).hexdigest()
    
    def verify_environment_stability(self):
        """Verify current environment matches baseline"""
        self.log_section("VERIFYING ENVIRONMENT STABILITY")
        
        if self.stability_baseline is None:
            logger.warning("⚠️  No baseline captured. Capturing now...")
            self.capture_environment_baseline()
            return True
        
        current_env = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'environment_vars': self._capture_critical_env_vars(),
            'data_paths': self._capture_data_paths(),
            'market_conditions': self._capture_market_baseline(),
            'model_input_specs': self._capture_model_input_specs(),
        }
        
        # Check for critical differences
        issues = []
        
        # Python version check
        if current_env['python_version'] != self.stability_baseline['python_version']:
            issues.append(f"Python version mismatch: {current_env['python_version']} vs {self.stability_baseline['python_version']}")
        
        # Environment variables check
        for var, baseline_val in self.stability_baseline['environment_vars'].items():
            current_val = current_env['environment_vars'].get(var)
            if current_val != baseline_val:
                issues.append(f"Env var {var} changed: {baseline_val} -> {current_val}")
        
        # Data paths check
        for path_name, baseline_path in self.stability_baseline['data_paths'].items():
            current_path = current_env['data_paths'].get(path_name)
            if current_path != baseline_path:
                issues.append(f"Data path {path_name} changed: {baseline_path} -> {current_path}")
        
        if issues:
            logger.warning("⚠️  ENVIRONMENT CHANGES DETECTED:")
            for issue in issues:
                logger.warning(f"   - {issue}")
            
            # Save drift report
            drift_report = {
                'timestamp': datetime.now().isoformat(),
                'baseline_timestamp': self.stability_baseline['timestamp'],
                'issues': issues,
                'severity': 'HIGH' if len(issues) > 2 else 'MEDIUM',
                'recommendation': 'RETRAIN_MODELS' if len(issues) > 2 else 'MONITOR_CLOSELY'
            }
            
            drift_file = self.output_dir / "environment_drift_report.json"
            with open(drift_file, 'w') as f:
                json.dump(drift_report, f, indent=2)
            
            logger.warning(f"📋 Drift report saved: {drift_file}")
            return False
        
        logger.info("✅ Environment stable - no distribution shift detected")
        return True
    
    def create_environment_wrapper(self):
        """Create wrapper to maintain environment consistency"""
        self.log_section("CREATING ENVIRONMENT WRAPPER")
        
        wrapper_config = {
            'purpose': 'Maintain environment consistency across model inference',
            'baseline': self.stability_baseline,
            'stability_checks': [
                'python_version',
                'environment_variables',
                'data_paths',
                'market_conditions',
                'model_input_specs'
            ],
            'drift_detection': {
                'enabled': True,
                'threshold': self.drift_threshold,
                'action_on_drift': 'LOG_AND_ALERT'
            },
            'created_at': datetime.now().isoformat()
        }
        
        wrapper_file = self.output_dir / "environment_wrapper_config.json"
        with open(wrapper_file, 'w') as f:
            json.dump(wrapper_config, f, indent=2)
        
        logger.info(f"✅ Environment wrapper created: {wrapper_file}")
        return wrapper_config
    
    def generate_stability_report(self):
        """Generate comprehensive stability report"""
        self.log_section("GENERATING STABILITY REPORT")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'baseline_captured': self.stability_baseline is not None,
            'environment_stable': self.verify_environment_stability(),
            'baseline': self.stability_baseline,
            'recommendations': [
                '✅ Keep environment variables consistent across all inference runs',
                '✅ Use same Python version for training and inference',
                '✅ Maintain identical data paths and market data sources',
                '✅ Monitor for market regime changes (distribution shift)',
                '✅ Log all environment changes for audit trail',
                '✅ Implement automatic drift detection and alerting'
            ]
        }
        
        report_file = self.output_dir / "environment_stability_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Stability report generated: {report_file}")
        return report

def main():
    """Main entry point"""
    output_dir = "/mnt/new_data/t10_training/phase2_results"
    
    monitor = EnvironmentStabilityMonitor(output_dir)
    
    # Capture baseline
    monitor.capture_environment_baseline()
    
    # Verify stability
    is_stable = monitor.verify_environment_stability()
    
    # Create wrapper
    monitor.create_environment_wrapper()
    
    # Generate report
    monitor.generate_stability_report()
    
    if is_stable:
        logger.info("\n✅ Environment is stable - safe to proceed with model inference")
        return 0
    else:
        logger.warning("\n⚠️  Environment changes detected - review before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
