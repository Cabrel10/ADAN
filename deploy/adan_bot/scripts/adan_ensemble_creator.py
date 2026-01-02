#!/usr/bin/env python3
"""
ADAN Ensemble Creator
Creates the final ADAN model by fusing all 4 trained workers
WITH ENVIRONMENT STABILITY PRESERVATION
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from environment_stability_monitor import EnvironmentStabilityMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adan_ensemble_creator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdanEnsembleCreator:
    """Creates ADAN ensemble model from trained workers with environment preservation"""
    
    def __init__(self, checkpoint_dir, output_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        
        # Environment stability
        self.env_monitor = EnvironmentStabilityMonitor(str(self.output_dir))
        
        # Ensemble configuration
        self.voting_strategy = 'majority'
        self.confidence_weighting = True
        self.hyperparameter_fusion = 'weighted_average'
        
    def log_section(self, title):
        """Log a section header"""
        logger.info("=" * 80)
        logger.info(f"🚀 {title}")
        logger.info("=" * 80)
    
    def verify_environment_before_fusion(self):
        """Verify environment is stable before fusing models"""
        self.log_section("VERIFYING ENVIRONMENT STABILITY")
        
        # Capture baseline if not exists
        baseline = self.env_monitor.capture_environment_baseline()
        
        # Verify stability
        is_stable = self.env_monitor.verify_environment_stability()
        
        if not is_stable:
            logger.warning("⚠️  ENVIRONMENT INSTABILITY DETECTED")
            logger.warning("   Models may experience distribution shift!")
            logger.warning("   Proceeding with caution...")
        else:
            logger.info("✅ Environment stable - safe to proceed")
        
        return is_stable
    
    def load_worker_checkpoint(self, worker):
        """Load a worker's checkpoint and extract model info"""
        logger.info(f"📦 Loading {worker.upper()} checkpoint...")
        
        try:
            checkpoint_path = self.checkpoint_dir / worker
            checkpoints = list(checkpoint_path.glob(f"{worker}_model_*.zip"))
            
            if not checkpoints:
                logger.error(f"❌ {worker}: No checkpoints found")
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            steps = int(latest.stem.split('_')[-2])
            
            # Extract model info
            worker_info = {
                'worker': worker,
                'checkpoint_path': str(latest),
                'steps': steps,
                'size_mb': latest.stat().st_size / (1024 * 1024),
                'last_modified': datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
                'hyperparameters': self.extract_hyperparameters(worker),
                'performance_metrics': self.extract_performance_metrics(worker),
                'confidence_score': self.calculate_confidence_score(worker, steps),
                'environment_hash': self.env_monitor.stability_baseline['hash'] if self.env_monitor.stability_baseline else None
            }
            
            logger.info(f"✅ {worker}: Loaded {steps:,} steps, confidence: {worker_info['confidence_score']:.3f}")
            return worker_info
            
        except Exception as e:
            logger.error(f"❌ {worker}: Failed to load - {e}")
            return None
    
    def extract_hyperparameters(self, worker):
        """Extract hyperparameters for a worker"""
        # Known Optuna-optimized hyperparameters from previous training
        hyperparams = {
            'w1': {
                'learning_rate': 1.08e-05,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gae_lambda': 0.95,
                'gamma': 0.99,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5
            },
            'w2': {
                'learning_rate': 9.5e-06,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gae_lambda': 0.96,
                'gamma': 0.99,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5
            },
            'w3': {
                'learning_rate': 1.15e-05,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gae_lambda': 0.94,
                'gamma': 0.99,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5
            },
            'w4': {
               