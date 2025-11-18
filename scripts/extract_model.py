#!/usr/bin/env python3
"""
Extract trained model and hyperparameters from checkpoint
Prepares model for deployment and analysis
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse

import torch
import numpy as np
from stable_baselines3 import PPO


def extract_model(checkpoint_path, output_dir=None):
    """Extract model from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if output_dir is None:
        output_dir = Path("models/extracted")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting model from: {checkpoint_path}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load model
        print("⏳ Loading model...")
        model = PPO.load(str(checkpoint_path))
        
        # Extract model architecture
        model_info = {
            "policy": str(model.policy),
            "learning_rate": float(model.learning_rate) if hasattr(model, 'learning_rate') else None,
            "n_steps": int(model.n_steps) if hasattr(model, 'n_steps') else None,
            "batch_size": int(model.batch_size) if hasattr(model, 'batch_size') else None,
            "n_epochs": int(model.n_epochs) if hasattr(model, 'n_epochs') else None,
            "gamma": float(model.gamma) if hasattr(model, 'gamma') else None,
            "gae_lambda": float(model.gae_lambda) if hasattr(model, 'gae_lambda') else None,
            "clip_range": float(model.clip_range) if hasattr(model, 'clip_range') else None,
            "ent_coef": float(model.ent_coef) if hasattr(model, 'ent_coef') else None,
            "vf_coef": float(model.vf_coef) if hasattr(model, 'vf_coef') else None,
            "max_grad_norm": float(model.max_grad_norm) if hasattr(model, 'max_grad_norm') else None,
        }
        
        # Save model info
        model_info_path = output_dir / "model_info.json"
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        print(f"✓ Model info saved: {model_info_path}")
        
        # Save policy network weights
        policy_path = output_dir / "policy.pt"
        torch.save(model.policy.state_dict(), policy_path)
        print(f"✓ Policy weights saved: {policy_path}")
        
        # Save value network weights
        value_path = output_dir / "value.pt"
        torch.save(model.policy.value_net.state_dict(), value_path)
        print(f"✓ Value network saved: {value_path}")
        
        # Save action network weights
        action_path = output_dir / "action.pt"
        torch.save(model.policy.action_net.state_dict(), action_path)
        print(f"✓ Action network saved: {action_path}")
        
        # Save full model for inference
        model_file = output_dir / "model.zip"
        model.save(str(model_file))
        print(f"✓ Full model saved: {model_file}")
        
        # Extract and save network architecture
        arch_info = {
            "policy_type": model.policy_class.__name__,
            "features_extractor": str(model.policy.features_extractor),
            "action_net": str(model.policy.action_net),
            "value_net": str(model.policy.value_net),
        }
        
        arch_path = output_dir / "architecture.json"
        with open(arch_path, 'w') as f:
            json.dump(arch_info, f, indent=2, default=str)
        print(f"✓ Architecture saved: {arch_path}")
        
        # Create extraction summary
        summary = {
            "extraction_date": datetime.now().isoformat(),
            "checkpoint_source": str(checkpoint_path),
            "output_directory": str(output_dir),
            "model_type": "PPO",
            "files_created": [
                "model_info.json",
                "policy.pt",
                "value.pt",
                "action.pt",
                "model.zip",
                "architecture.json",
                "extraction_summary.json"
            ],
            "model_parameters": model_info,
            "status": "success"
        }
        
        summary_path = output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Extraction summary saved: {summary_path}")
        
        print("\n✅ Model extraction completed successfully!")
        print(f"\nExtracted files in: {output_dir}")
        print("  - model.zip: Full model for inference")
        print("  - policy.pt: Policy network weights")
        print("  - value.pt: Value network weights")
        print("  - action.pt: Action network weights")
        print("  - model_info.json: Model hyperparameters")
        print("  - architecture.json: Network architecture")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find latest checkpoint in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find all .zip files (PPO checkpoints)
    checkpoints = list(checkpoint_dir.glob("*.zip"))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in: {checkpoint_dir}")
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Extract trained model from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (auto-detect if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/extracted",
        help="Output directory for extracted model"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use latest checkpoint from checkpoints/ directory"
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.latest or args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint
    
    print(f"Using checkpoint: {checkpoint_path}\n")
    
    # Extract model
    success = extract_model(checkpoint_path, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
