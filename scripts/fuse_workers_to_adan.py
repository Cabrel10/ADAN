#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fusionne 4 workers en 1 ADAN avec VecNormalize.
Crée un modèle unifié par moyenne des poids.
"""

import os
import shutil
import torch
from stable_baselines3 import PPO

import argparse

def fuse_workers(checkpoint_dir="models/baseline/final", output_dir=None):
    """Moyenne pondérée des poids des 4 workers"""
    
    print(f"🔄 Starting fusion from {checkpoint_dir}...")
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint dir not found: {checkpoint_dir}")
        return

    # Charger les 4 workers
    workers = []
    worker_ids = ["w1", "w2", "w3", "w4"]
    
    for worker_id in worker_ids:
        path = os.path.join(checkpoint_dir, f"{worker_id}_final.zip")
        if not os.path.exists(path):
             print(f"❌ Worker model not found: {path}")
             return
        workers.append(PPO.load(path))
        print(f"✅ Loaded {path}")
    
    # Créer modèle ADAN (base = worker 1)
    adan = workers[0]
    
    # Fusionner les poids (moyenne)
    print("🔄 Averaging weights...")
    state_dict = adan.policy.state_dict()
    for key in state_dict.keys():
        # Moyenne des 4 workers
        # Note: We must ensure all tensors are on same device or CPU
        weights = [w.policy.state_dict()[key].cpu() for w in workers]
        state_dict[key] = torch.stack(weights).mean(dim=0)
    
    # Appliquer les poids fusionnés
    adan.policy.load_state_dict(state_dict)
    
    # Sauvegarder
    if output_dir is None:
        output_dir = checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    
    adan_path = os.path.join(output_dir, "adan_model_final.zip")
    adan.save(adan_path)
    print(f"✅ ADAN model created: {adan_path}")
    
    # Copier vecnormalize.pkl
    # We expect it in models/baseline/vecnormalize.pkl (saved by w1 in main dir)
    # OR in models/baseline/final/w1_vecnormalize.pkl
    
    # Try to find global stats first (from parent dir)
    parent_dir = os.path.dirname(checkpoint_dir.rstrip('/'))
    main_vec_src = os.path.join(parent_dir, "vecnormalize.pkl")
    
    # Fallback to worker stats
    worker_vec_src = os.path.join(checkpoint_dir, "w1_vecnormalize.pkl")
    
    vec_dst = os.path.join(output_dir, "vecnormalize.pkl")
    
    if os.path.exists(main_vec_src):
        shutil.copy(main_vec_src, vec_dst)
        print(f"✅ Global VecNormalize stats copied to {vec_dst}")
    elif os.path.exists(worker_vec_src):
        shutil.copy(worker_vec_src, vec_dst)
        print(f"✅ w1 VecNormalize stats copied to {vec_dst}")
    else:
        print("⚠️ No vecnormalize.pkl found to copy!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse worker models into ADAN")
    parser.add_argument("--models-dir", type=str, default="models/baseline/final", help="Directory containing worker models")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: same as models-dir)")
    parser.add_argument("--config", type=str, help="Ignored (for compatibility)")
    
    args = parser.parse_args()
    
    fuse_workers(args.models_dir, args.output)
