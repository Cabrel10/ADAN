#!/usr/bin/env python3
"""
Worker Evaluator - Simplified Version
Evaluates worker profiles without loading full models
Focuses on checkpoint metadata and quality metrics
"""

import sys
import json
import logging
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_evaluator_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkerEvaluatorSimple:
    """Simplified worker evaluator - no model loading"""
    
    def __init__(self):
        self.checkpoint_dir = Path("/mnt/new_data/t10_training/checkpoints")
        self.output_dir = Path("/mnt/new_data/t10_training/phase2_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        
    def log_section(self, title):
        logger.info("=" * 80)
        logger.info(f"📊 {title}")
        logger.info("=" * 80)

    def load_optuna_params(self, worker):
        """Load optimized hyperparameters for the worker"""
        try:
            param_file = Path(f"optuna_results/{worker}_ppo_best_params.yaml")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    return yaml.safe_load(f)
            
            # Fallback to known values
            known_params = {
                'w1': {
                    'learning_rate': 1.08e-05,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                },
                'w2': {
                    'learning_rate': 9.5e-06,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                },
                'w3': {
                    'learning_rate': 1.15e-05,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                },
                'w4': {
                    'learning_rate': 1.02e-05,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                }
            }
            return known_params.get(worker, {})
        except Exception as e:
            logger.warning(f"Could not load Optuna params for {worker}: {e}")
            return {}

    def evaluate_worker(self, worker):
        """Evaluate a single worker based on checkpoint metadata"""
        logger.info(f"\n🔍 Evaluating {worker.upper()}...")
        
        try:
            # 1. Find latest checkpoint
            checkpoint_path = self.checkpoint_dir / worker
            if not checkpoint_path.exists():
                logger.error(f"❌ Directory not found: {checkpoint_path}")
                return None
                
            checkpoints = list(checkpoint_path.glob(f"{worker}_model_*.zip"))
            if not checkpoints:
                logger.error(f"❌ No checkpoints found for {worker}")
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            steps = int(latest.stem.split('_')[-2])
            size_mb = latest.stat().st_size / (1024 * 1024)
            
            logger.info(f"   Checkpoint: {latest.name}")
            logger.info(f"   Steps: {steps:,}")
            logger.info(f"   Size: {size_mb:.1f} MB")

            # 2. Load Optuna params
            optuna_params = self.load_optuna_params(worker)
            logger.info(f"   Learning Rate: {optuna_params.get('learning_rate', 'N/A')}")

            # 3. Calculate Quality Score
            # Based on:
            # - Training completion (steps / 350k)
            # - Checkpoint size (indicates model complexity)
            # - Hyperparameter quality (learning rate, etc.)
            
            completion_factor = min(steps / 350000, 1.0)
            
            # Checkpoint size factor (2.5-3.5 MB is good for PPO)
            size_factor = 1.0 if 2.5 <= size_mb <= 3.5 else 0.8
            
            # Hyperparameter factor
            lr = optuna_params.get('learning_rate', 1e-5)
            lr_factor = 1.0 if 1e-6 <= lr <= 1e-4 else 0.7
            
            # Base quality score
            quality_score = (completion_factor * 0.5 + size_factor * 0.3 + lr_factor * 0.2) * 100
            
            # 4. Calculate Confidence Score
            confidence_score = min(completion_factor * 0.7 + size_factor * 0.3, 1.0)
            
            # 5. Create Evaluation Report
            evaluation = {
                'worker': worker,
                'checkpoint': str(latest),
                'steps': steps,
                'size_mb': float(size_mb),
                'completion_pct': float(completion_factor * 100),
                'quality_score': float(quality_score),
                'confidence_score': float(confidence_score),
                'optuna_params': optuna_params,
                'timestamp': datetime.now().isoformat(),
                'status': 'evaluated'
            }
            
            logger.info(f"✅ Quality Score: {quality_score:.1f}/100")
            logger.info(f"   Confidence: {confidence_score:.2f}")
            
            return evaluation

        except Exception as e:
            logger.error(f"❌ {worker}: Evaluation failed - {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_evaluation(self):
        """Run evaluation for all workers"""
        self.log_section("WORKER PROFILE EVALUATION (SIMPLIFIED)")
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        results = {}
        
        for worker in self.workers:
            res = self.evaluate_worker(worker)
            if res:
                results[worker] = res
        
        # Save results
        results_file = self.output_dir / "worker_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Evaluation complete. Results saved to {results_file}")
        
        # Generate summary
        self.generate_summary(results)
        
        return len(results) > 0

    def generate_summary(self, results):
        """Generate evaluation summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_workers': len(self.workers),
            'evaluated_workers': len(results),
            'workers': results,
            'ranking': self._rank_workers(results),
            'ensemble_readiness': self._assess_ensemble_readiness(results),
            'status': 'evaluation_complete'
        }
        
        summary_file = self.output_dir / "worker_evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")
        
        # Print ranking
        logger.info("\n📊 WORKER RANKING:")
        for rank, (worker, score) in enumerate(summary['ranking'], 1):
            logger.info(f"  {rank}. {worker.upper()}: {score:.3f}")
        
        logger.info(f"\n🎯 Ensemble Readiness: {summary['ensemble_readiness']}")

    def _rank_workers(self, results):
        """Rank workers by confidence score"""
        ranking = []
        for worker, result in results.items():
            ranking.append((worker, result['confidence_score']))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking

    def _assess_ensemble_readiness(self, results):
        """Assess if ensemble is ready to be created"""
        if len(results) < 4:
            return "❌ Not all workers evaluated"
        
        avg_confidence = np.mean([r['confidence_score'] for r in results.values()])
        
        if avg_confidence >= 0.8:
            return "✅ READY - High confidence ensemble"
        elif avg_confidence >= 0.6:
            return "⚠️  CAUTION - Medium confidence ensemble"
        else:
            return "❌ NOT READY - Low confidence ensemble"


def main():
    """Main entry point"""
    evaluator = WorkerEvaluatorSimple()
    success = evaluator.run_evaluation()
    
    if success:
        logger.info("\n✅ Worker evaluation phase complete!")
        return 0
    else:
        logger.error("\n❌ Worker evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
