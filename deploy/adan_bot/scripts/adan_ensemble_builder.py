#!/usr/bin/env python3
"""
ADAN Ensemble Builder - Production Version
Creates ensemble configuration from evaluated worker models.
"""

import sys
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
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
    """Builds ADAN ensemble from evaluated worker models"""
    
    def __init__(self):
        self.output_dir = Path("/mnt/new_data/t10_training/phase2_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = ['w1', 'w2', 'w3', 'w4']
        
    def log_section(self, title):
        logger.info("=" * 80)
        logger.info(f"🔗 {title}")
        logger.info("=" * 80)

    def load_evaluation_results(self):
        results_file = self.output_dir / "worker_evaluation_results.json"
        if not results_file.exists():
            logger.error(f"❌ Results not found: {results_file}")
            return None
        with open(results_file, 'r') as f:
            return json.load(f)

    def compute_confidence_weights(self, results):
        """Compute normalized confidence weights"""
        weights = {}
        total = 0
        for worker, res in results.items():
            conf = res.get('confidence_score', 0.1)
            weights[worker] = conf
            total += conf
            
        # Normalize
        if total > 0:
            return {w: c / total for w, c in weights.items()}
        return {w: 0.25 for w in self.workers}

    def fuse_hyperparameters(self, results, weights):
        """Fuse hyperparameters (weighted average for numeric, mode for categorical)"""
        logger.info("🔀 Fusing hyperparameters...")
        fused = {}
        
        # Gather all params
        all_params = {}
        for worker, res in results.items():
            params = res.get('optuna_params', {})
            for k, v in params.items():
                if k not in all_params: all_params[k] = []
                all_params[k].append((v, weights.get(worker, 0)))
        
        # Fuse
        for k, values in all_params.items():
            if not values: continue
            val_type = type(values[0][0])
            
            if val_type in [int, float]:
                # Weighted Average
                weighted_sum = sum(v * w for v, w in values)
                total_weight = sum(w for v, w in values)
                fused[k] = weighted_sum / total_weight if total_weight > 0 else 0
                if val_type == int: fused[k] = int(fused[k])
            else:
                # Mode (weighted?) - Simplification: just take best worker's param
                # Or most frequent. Let's use most frequent for now.
                vals = [v for v, w in values]
                fused[k] = max(set(vals), key=vals.count)
                
        return fused

    def create_environment_baseline(self):
        """Capture environment state for stability monitoring"""
        return {
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat(),
            'platform': sys.platform,
            # In production, add library versions, etc.
        }

    def build_ensemble(self):
        self.log_section("BUILDING ADAN ENSEMBLE (PRODUCTION)")
        
        # 1. Load Evaluation Results
        results = self.load_evaluation_results()
        if not results: return False
        
        # 2. Compute Weights
        weights = self.compute_confidence_weights(results)
        logger.info(f"   Weights: {json.dumps(weights, indent=2)}")
        
        # 3. Fuse Hyperparameters
        fused_params = self.fuse_hyperparameters(results, weights)
        
        # 4. Create Config
        ensemble_config = {
            'name': 'ADAN_Ensemble_v1',
            'created_at': datetime.now().isoformat(),
            'workers': list(results.keys()),
            'weights': weights,
            'fused_hyperparameters': fused_params,
            'voting_strategy': 'confidence_weighted',
            'environment_baseline': self.create_environment_baseline(),
            'worker_metadata': results,
            'ensemble_quality_metrics': {
                'ensemble_quality_score': sum(results[w].get('confidence_score', 0) * weights[w] for w in results) * 100 if results else 0.0,
                'worker_count': len(results),
                'average_confidence': np.mean([r.get('confidence_score', 0) for r in results.values()]) if results else 0.0
            }
        }
        
        # 5. Save
        config_path = self.output_dir / "adan_ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
            
        logger.info(f"✅ Ensemble config saved to {config_path}")
        return True

def main():
    builder = AdanEnsembleBuilder()
    success = builder.build_ensemble()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
