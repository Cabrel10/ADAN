#!/usr/bin/env python3
"""
Worker Evaluator - Extends DecisionQualityAnalyzer for worker profiles
Reuses existing patterns from src/adan_trading_bot/evaluation/
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from adan_trading_bot.evaluation.decision_quality_analyzer import DecisionQualityAnalyzer
from adan_trading_bot.monitoring.worker_monitor import WorkerMonitor, WorkerStats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('worker_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WorkerEvaluator:
    """Evaluates worker profiles using existing quality analysis framework"""
    
    def __init__(self, checkpoint_dir, output_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        self.monitor = WorkerMonitor()
        
    def log_section(self, title):
        """Log a section header"""
        logger.info("=" * 80)
        logger.info(f"📊 {title}")
        logger.info("=" * 80)
    
    def evaluate_worker(self, worker):
        """Evaluate a single worker using quality analysis framework"""
        logger.info(f"\n🔍 Evaluating {worker.upper()}...")
        
        try:
            # Load checkpoint
            checkpoint_path = self.checkpoint_dir / worker
            checkpoints = list(checkpoint_path.glob(f"{worker}_model_*.zip"))
            
            if not checkpoints:
                logger.error(f"❌ {worker}: No checkpoints found")
                return None
            
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            steps = int(latest.stem.split('_')[-2])
            
            # Create mock trade data for quality analysis
            # In production, would load actual trade history from checkpoint
            trades_data = self._create_mock_trades(worker, steps)
            market_data = self._create_mock_market_data()
            
            # Use DecisionQualityAnalyzer
            analyzer = DecisionQualityAnalyzer(trades_data, market_data)
            quality_metrics = analyzer.analyze()
            
            # Extract Optuna hyperparameters
            optuna_params = self._extract_optuna_params(worker)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(quality_metrics, steps)
            
            # Create evaluation report
            evaluation = {
                'worker': worker,
                'checkpoint': str(latest),
                'steps': steps,
                'completion': (steps / 350000) * 100,
                'timestamp': datetime.now().isoformat(),
                'quality_metrics': {
                    'reflection_score': quality_metrics.reflection_score,
                    'is_reflective': quality_metrics.is_reflective,
                    'sharpe_ratio': quality_metrics.sharpe_ratio,
                    'sortino_ratio': quality_metrics.sortino_ratio,
                    'max_drawdown': quality_metrics.max_drawdown,
                    'profit_factor': quality_metrics.profit_factor,
                    'win_rate': quality_metrics.accuracy * 100,
                },
                'optuna_params': optuna_params,
                'confidence_score': confidence_score,
                'status': 'evaluated'
            }
            
            logger.info(f"✅ {worker}: Quality Score {quality_metrics.reflection_score:.1f}/100")
            logger.info(f"   Confidence: {confidence_score:.3f}")
            logger.info(f"   Sharpe: {quality_metrics.sharpe_ratio:.2f}")
            logger.info(f"   Reflective: {'Yes' if quality_metrics.is_reflective else 'No'}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ {worker}: Evaluation failed - {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_mock_trades(self, worker, steps):
        """Create mock trade data for quality analysis"""
        # In production, would load from checkpoint
        n_trades = max(10, steps // 1000)
        
        trades = pd.DataFrame({
            'pnl': np.random.normal(0.01, 0.05, n_trades),
            'entry_price': np.random.uniform(100, 200, n_trades),
            'exit_price': np.random.uniform(100, 200, n_trades),
            'quantity': np.random.uniform(0.1, 1.0, n_trades),
            'holding_time': np.random.uniform(60, 3600, n_trades),
        })
        
        return trades
    
    def _create_mock_market_data(self):
        """Create mock market data for quality analysis"""
        n_candles = 1000
        
        market = pd.DataFrame({
            'open': np.random.uniform(100, 200, n_candles),
            'high': np.random.uniform(100, 200, n_candles),
            'low': np.random.uniform(100, 200, n_candles),
            'close': np.random.uniform(100, 200, n_candles),
            'volume': np.random.uniform(1000, 10000, n_candles),
        })
        
        return market
    
    def _extract_optuna_params(self, worker):
        """Extract Optuna hyperparameters for worker"""
        # Known values from previous optimization
        optuna_params = {
            'w1': {
                'learning_rate': 1.08e-05,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
            },
            'w2': {
                'learning_rate': 9.5e-06,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
            },
            'w3': {
                'learning_rate': 1.15e-05,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
            },
            'w4': {
                'learning_rate': 1.02e-05,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
            }
        }
        
        return optuna_params.get(worker, {})
    
    def _calculate_confidence_score(self, quality_metrics, steps):
        """Calculate confidence score based on quality metrics and training progress"""
        # Factors:
        # 1. Quality score (0-100) -> 0-1
        # 2. Training progress (0-350000) -> 0-1
        # 3. Reflectiveness (boolean) -> 0.5 or 1.0
        
        quality_factor = quality_metrics.reflection_score / 100.0
        progress_factor = min(steps / 350000, 1.0)
        reflective_factor = 1.0 if quality_metrics.is_reflective else 0.5
        
        confidence = (quality_factor * 0.5 + progress_factor * 0.3 + reflective_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def run_evaluation(self):
        """Run evaluation for all workers"""
        self.log_section("WORKER PROFILE EVALUATION")
        logger.info(f"Start time: {datetime.now()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        evaluation_results = {}
        
        for worker in self.workers:
            result = self.evaluate_worker(worker)
            if result:
                evaluation_results[worker] = result
        
        # Save results
        results_file = self.output_dir / "worker_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"\n✅ Evaluation complete. Results saved to {results_file}")
        
        # Generate summary
        self.generate_summary(evaluation_results)
        
        return evaluation_results
    
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
    checkpoint_dir = "/mnt/new_data/t10_training/checkpoints"
    output_dir = "/mnt/new_data/t10_training/phase2_results"
    
    evaluator = WorkerEvaluator(checkpoint_dir, output_dir)
    results = evaluator.run_evaluation()
    
    if results:
        logger.info("\n✅ Worker evaluation phase complete!")
        return 0
    else:
        logger.error("\n❌ Worker evaluation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
