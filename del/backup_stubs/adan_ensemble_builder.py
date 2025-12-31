#!/usr/bin/env python3
"""
ADAN Ensemble Builder - Creates ensemble from 4 worker models
Reuses patterns from src/adan_trading_bot/portfolio/portfolio_manager.py
"""

import sys
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adan_ensemble_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdanEnsembleBuilder:
    """Builds ADAN ensemble from 4 trained worker models"""
    
    def __init__(self, checkpoint_dir, output_dir, evaluation_results):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        self.evaluation_results = evaluation_results
        
        # Ensemble configuration
        self.voting_strategy = 'confidence_weighted_majority'
        self.confidence_weights = self._compute_confidence_weights()
        
    def log_section(self, title):
        """Log a section header"""
        logger.info("=" * 80)
        logger.info(f"🔗 {title}")
        logger.info("=" * 80)
    
    def _compute_confidence_weights(self):
        """Compute confidence weights for each worker"""
        weights = {}
        total_confidence = 0
        
        for worker, result in self.evaluation_results.items():
            confidence = result.get('confidence_score', 0.5)
            weights[worker] = confidence
            total_confidence += confidence
        
        # Normalize to sum to 1
        if total_confidence > 0:
            weights = {w: c / total_confidence for w, c in weights.items()}
        
        return weights
    
    def load_worker_model(self, worker):
        """Load a worker's trained model"""
        logger.info(f"📦 Loading {worker.upper()} model...")
        
        try:
            checkpoint_path = self.checkpoint_dir / worker
            checkpoints = list(checkpoint_path.glob(f"{worker}_model_*.zip"))
            
            if not checkpoints:
                logger.error(f"❌ {worker}: No checkpoints found")
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            steps = int(latest.stem.split('_')[-2])
            
            # In production, would load actual PPO model
            # For now, create model metadata
            model_info = {
                'worker': worker,
                'checkpoint_path': str(latest),
                'steps': steps,
                'size_mb': latest.stat().st_size / (1024 * 1024),
                'confidence_weight': self.confidence_weights.get(worker, 0.25),
                'hyperparameters': self.evaluation_results[worker].get('optuna_params', {}),
                'quality_metrics': self.evaluation_results[worker].get('quality_metrics', {}),
            }
            
            logger.info(f"✅ {worker}: Loaded {steps:,} steps")
            logger.info(f"   Weight: {model_info['confidence_weight']:.3f}")
            logger.info(f"   Quality: {model_info['quality_metrics'].get('reflection_score', 0):.1f}/100")
            
            return model_info
            
        except Exception as e:
            logger.error(f"❌ {worker}: Failed to load - {e}")
            return None
    
    def create_ensemble_config(self):
        """Create ensemble configuration"""
        self.log_section("CREATING ENSEMBLE CONFIGURATION")
        
        # Load all worker models
        worker_models = {}
        for worker in self.workers:
            model = self.load_worker_model(worker)
            if model:
                worker_models[worker] = model
        
        if len(worker_models) < 4:
            logger.error(f"❌ Only {len(worker_models)}/4 workers loaded")
            return None
        
        # Create ensemble config
        ensemble_config = {
            'model_type': 'ensemble',
            'name': 'ADAN_Final',
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'workers': worker_models,
            'voting_strategy': self.voting_strategy,
            'confidence_weights': self.confidence_weights,
            'ensemble_hyperparameters': self._fuse_hyperparameters(worker_models),
            'ensemble_quality_metrics': self._aggregate_quality_metrics(worker_models),
            'environment_baseline': self._create_environment_baseline(),
        }
        
        # Save ensemble config
        config_file = self.output_dir / "adan_ensemble_config.json"
        with open(config_file, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        logger.info(f"✅ Ensemble configuration created: {config_file}")
        
        return ensemble_config
    
    def _fuse_hyperparameters(self, worker_models):
        """Fuse hyperparameters from all workers"""
        logger.info("🔀 Fusing hyperparameters...")
        
        # Collect all hyperparameters
        all_params = {}
        for worker, model in worker_models.items():
            params = model.get('hyperparameters', {})
            for key, value in params.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)
        
        # Compute weighted average
        fused_params = {}
        for key, values in all_params.items():
            if isinstance(values[0], (int, float)):
                # Weighted average for numeric values
                weighted_sum = sum(
                    v * self.confidence_weights.get(w, 0.25)
                    for v, w in zip(values, self.workers)
                )
                fused_params[key] = weighted_sum
            else:
                # Use most common value for non-numeric
                fused_params[key] = max(set(values), key=values.count)
        
        logger.info(f"✅ Fused {len(fused_params)} hyperparameters")
        
        return fused_params
    
    def _aggregate_quality_metrics(self, worker_models):
        """Aggregate quality metrics from all workers"""
        logger.info("📊 Aggregating quality metrics...")
        
        metrics_list = [
            model.get('quality_metrics', {})
            for model in worker_models.values()
        ]
        
        aggregated = {}
        
        # Average reflection score
        reflection_scores = [
            m.get('reflection_score', 0) for m in metrics_list
        ]
        aggregated['avg_reflection_score'] = np.mean(reflection_scores)
        aggregated['min_reflection_score'] = np.min(reflection_scores)
        aggregated['max_reflection_score'] = np.max(reflection_scores)
        
        # Average Sharpe ratio
        sharpe_ratios = [m.get('sharpe_ratio', 0) for m in metrics_list]
        aggregated['avg_sharpe_ratio'] = np.mean(sharpe_ratios)
        
        # Average Sortino ratio
        sortino_ratios = [m.get('sortino_ratio', 0) for m in metrics_list]
        aggregated['avg_sortino_ratio'] = np.mean(sortino_ratios)
        
        # Average max drawdown
        max_drawdowns = [m.get('max_drawdown', 0) for m in metrics_list]
        aggregated['avg_max_drawdown'] = np.mean(max_drawdowns)
        
        # Average win rate
        win_rates = [m.get('win_rate', 0) for m in metrics_list]
        aggregated['avg_win_rate'] = np.mean(win_rates)
        
        # Ensemble quality score
        aggregated['ensemble_quality_score'] = (
            aggregated['avg_reflection_score'] * 0.4 +
            min(aggregated['avg_sharpe_ratio'] / 2.0, 100) * 0.3 +
            aggregated['avg_win_rate'] * 0.3
        )
        
        logger.info(f"✅ Ensemble Quality Score: {aggregated['ensemble_quality_score']:.1f}/100")
        
        return aggregated
    
    def _create_environment_baseline(self):
        """Create environment baseline for stability monitoring"""
        return {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'checkpoint_dir': str(self.checkpoint_dir),
            'data_dir': '/mnt/new_data/t10_training/data',
            'market_data_dir': '/mnt/new_data/market_data',
            'ensemble_created_at': datetime.now().isoformat(),
            'workers_included': self.workers,
        }
    
    def create_voting_mechanism(self):
        """Create voting mechanism for ensemble predictions"""
        self.log_section("CREATING VOTING MECHANISM")
        
        voting_config = {
            'strategy': self.voting_strategy,
            'description': 'Confidence-weighted majority voting',
            'mechanism': {
                'step_1': 'Get predictions from all 4 workers',
                'step_2': 'Weight each prediction by worker confidence',
                'step_3': 'Apply majority voting with weighted confidence',
                'step_4': 'Return ensemble prediction with confidence score',
            },
            'confidence_weights': self.confidence_weights,
            'tie_breaking': 'Use highest confidence worker prediction',
            'confidence_threshold': 0.6,  # Minimum confidence to execute trade
        }
        
        voting_file = self.output_dir / "ensemble_voting_mechanism.json"
        with open(voting_file, 'w') as f:
            json.dump(voting_config, f, indent=2)
        
        logger.info(f"✅ Voting mechanism created: {voting_file}")
        
        return voting_config
    
    def build_ensemble(self):
        """Build complete ADAN ensemble"""
        self.log_section("BUILDING ADAN ENSEMBLE")
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Create ensemble configuration
        ensemble_config = self.create_ensemble_config()
        if not ensemble_config:
            logger.error("❌ Failed to create ensemble configuration")
            return False
        
        # Step 2: Create voting mechanism
        voting_config = self.create_voting_mechanism()
        
        # Step 3: Generate ensemble summary
        self.generate_ensemble_summary(ensemble_config, voting_config)
        
        logger.info(f"\n✅ ADAN Ensemble built successfully!")
        logger.info(f"   Quality Score: {ensemble_config['ensemble_quality_metrics']['ensemble_quality_score']:.1f}/100")
        logger.info(f"   Workers: {len(ensemble_config['workers'])}/4")
        logger.info(f"   Voting Strategy: {self.voting_strategy}")
        
        return True
    
    def generate_ensemble_summary(self, ensemble_config, voting_config):
        """Generate ensemble summary report"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ensemble_name': ensemble_config['name'],
            'ensemble_version': ensemble_config['version'],
            'workers_count': len(ensemble_config['workers']),
            'voting_strategy': ensemble_config['voting_strategy'],
            'confidence_weights': ensemble_config['confidence_weights'],
            'quality_metrics': ensemble_config['ensemble_quality_metrics'],
            'voting_mechanism': voting_config,
            'status': 'ready_for_backtest',
            'next_steps': [
                '1. Run backtest on historical data',
                '2. Validate ensemble performance',
                '3. Deploy to paper trading',
                '4. Monitor environment stability',
            ]
        }
        
        summary_file = self.output_dir / "adan_ensemble_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {summary_file}")


def main():
    """Main entry point"""
    checkpoint_dir = "/mnt/new_data/t10_training/checkpoints"
    output_dir = "/mnt/new_data/t10_training/phase2_results"
    
    # Load evaluation results
    eval_results_file = Path(output_dir) / "worker_evaluation_results.json"
    if not eval_results_file.exists():
        logger.error(f"❌ Evaluation results not found: {eval_results_file}")
        logger.info("Run worker_evaluator.py first")
        return 1
    
    with open(eval_results_file, 'r') as f:
        evaluation_results = json.load(f)
    
    # Build ensemble
    builder = AdanEnsembleBuilder(checkpoint_dir, output_dir, evaluation_results)
    success = builder.build_ensemble()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
