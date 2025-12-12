#!/usr/bin/env python3
"""
Test script for worker evaluation - simplified version
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_worker_evaluation():
    """Test worker evaluation with mock data"""
    
    logger.info("=" * 80)
    logger.info("🧪 TESTING WORKER EVALUATION")
    logger.info("=" * 80)
    
    checkpoint_dir = Path("/mnt/new_data/t10_training/checkpoints")
    output_dir = Path("/mnt/new_data/t10_training/phase2_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    workers = ['w1', 'w2', 'w3', 'w4']
    results = {}
    
    for worker in workers:
        logger.info(f"\n📊 Checking {worker.upper()}...")
        
        worker_dir = checkpoint_dir / worker
        if not worker_dir.exists():
            logger.error(f"❌ Directory not found: {worker_dir}")
            continue
        
        checkpoints = list(worker_dir.glob(f"{worker}_model_*.zip"))
        if not checkpoints:
            logger.error(f"❌ No checkpoints found")
            continue
        
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        steps = int(latest.stem.split('_')[-2])
        
        logger.info(f"✅ Found checkpoint: {latest.name}")
        logger.info(f"   Steps: {steps:,}")
        logger.info(f"   Size: {latest.stat().st_size / (1024*1024):.1f} MB")
        
        results[worker] = {
            'checkpoint': str(latest),
            'steps': steps,
            'size_mb': latest.stat().st_size / (1024*1024),
            'timestamp': datetime.now().isoformat()
        }
    
    # Save results
    results_file = output_dir / "worker_check_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Check complete. Results saved to {results_file}")
    logger.info(f"   Found {len(results)}/4 workers")
    
    return len(results) == 4

if __name__ == "__main__":
    success = test_worker_evaluation()
    sys.exit(0 if success else 1)
