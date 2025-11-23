#!/usr/bin/env python3
"""
ADAN Multi-Worker Parallel Launcher
Launches 4 independent training processes simultaneously using multiprocessing.
Each worker (w1-w4) trains its own PPO model in a separate process.
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adan_trading_bot.common.config_loader import ConfigLoader


def train_single_worker(worker_id: str, total_steps: int, checkpoint_base_dir: str, config_path: str):
    """
    Train a single worker in an isolated process.
    This function will be called by multiprocessing.Process.
    
    Args:
        worker_id: Worker identifier (w1, w2, w3, w4)
        total_steps: Number of training steps
        checkpoint_base_dir: Base directory for checkpoints
        config_path: Path to config.yaml
    """
    import copy
    import torch
    import torch.nn as nn
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    
    # Setup logging for this worker
    log_file = f"logs/worker_{worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f'[{worker_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(f"worker_{worker_id}")
    
    try:
        logger.info("="*80)
        logger.info(f"🤖 WORKER {worker_id} STARTED (PID: {os.getpid()})")
        logger.info("="*80)
        
        # Load config
        config = ConfigLoader().load_config(config_path)
        worker_config = config["workers"][worker_id]
        agent_config = worker_config.get("agent_config", {})
        
        logger.info(f"Learning Rate: {agent_config.get('learning_rate')}")
        logger.info(f"N Steps: {agent_config.get('n_steps')}")
        logger.info(f"Risk Multiplier: {worker_config.get('risk_multiplier')}")
        
        # Create worker-specific environment
        def make_env():
            def _init():
                wc = copy.deepcopy(worker_config)
                data_loader = ChunkedDataLoader(
                    config=config, worker_config=wc, worker_id=0
                )
                data = data_loader.load_chunk(0)
                
                env_worker_config = copy.deepcopy(wc)
                env_worker_config["worker_id"] = 0
                
                env_log_dir = os.path.join(config["paths"]["logs_dir"], f"{worker_id}_env")
                os.makedirs(env_log_dir, exist_ok=True)
                
                return MultiAssetChunkedEnv(
                    data=data,
                    timeframes=config["data"]["timeframes"],
                    window_sizes=config["environment"]["observation"]["window_sizes"],
                    features_config=config["data"]["features_config"]["timeframes"],
                    max_steps=config["environment"]["max_steps"],
                    initial_balance=config["portfolio"]["initial_balance"],
                    commission=config["environment"]["commission"],
                    reward_scaling=config["environment"]["reward_scaling"],
                    enable_logging=True,
                    log_dir=env_log_dir,
                    worker_config=env_worker_config,
                    config=config,
                    exploration_tutor=config.get("reward_shaping", {}).get("exploration_tutor", {})
                )
            return _init
        
        # Create SubprocVecEnv with single environment
        env = SubprocVecEnv([make_env()])
        logger.info("✅ Environment created")
        
        # Policy kwargs
        policy_kwargs = copy.deepcopy(config["agent"]["features_extractor_kwargs"]["policy_kwargs"])
        activation_fn_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "LeakyReLU": nn.LeakyReLU}
        if "activation_fn" in policy_kwargs:
            act_fn_name = policy_kwargs["activation_fn"].split(".")[-1]
            policy_kwargs["activation_fn"] = activation_fn_map.get(act_fn_name, nn.ReLU)
        
        # Create checkpoint directory
        worker_checkpoint_dir = os.path.join(checkpoint_base_dir, worker_id)
        os.makedirs(worker_checkpoint_dir, exist_ok=True)
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=config["training"]["checkpointing"]["save_freq"],
            save_path=worker_checkpoint_dir,
            name_prefix=f"{worker_id}_model",
        )
        
        # GPU setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Worker-specific hyperparameters
        learning_rate = agent_config.get("learning_rate", config["agent"]["learning_rate"])
        n_steps = agent_config.get("n_steps", config["agent"]["n_steps"])
        batch_size = agent_config.get("batch_size", config["agent"]["batch_size"])
        n_epochs = agent_config.get("n_epochs", config["agent"]["n_epochs"])
        gamma = agent_config.get("gamma", config["agent"]["gamma"])
        gae_lambda = agent_config.get("gae_lambda", config["agent"]["gae_lambda"])
        clip_range = agent_config.get("clip_range", config["agent"]["clip_range"])
        ent_coef = agent_config.get("ent_coef", config["agent"]["ent_coef"])
        
        # Unique seed per worker
        worker_idx = int(worker_id[1]) - 1  # w1->0, w2->1, etc.
        worker_seed = config["agent"]["seed"] + worker_idx
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        logger.info(f"🧠 Creating PPO model (seed={worker_seed})")
        
        # Create PPO model
        model = PPO(
            "MultiInputPolicy",
            env,
            device=device,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=config["agent"]["vf_coef"],
            max_grad_norm=config["agent"]["max_grad_norm"],
            tensorboard_log=os.path.join(config["paths"]["logs_dir"], f"tensorboard_{worker_id}"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=worker_seed,
        )
        
        logger.info(f"🚀 Starting training ({total_steps:,} steps)")
        
        # Train
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback],
            progress_bar=False,
            reset_num_timesteps=True,
        )
        
        # Save final model
        final_model_path = os.path.join(checkpoint_base_dir, "final", f"{worker_id}_final.zip")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        model.save(final_model_path)
        
        logger.info("="*80)
        logger.info(f"✅ {worker_id} TRAINING COMPLETE")
        logger.info(f"📁 Model saved: {final_model_path}")
        logger.info("="*80)
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ {worker_id} FAILED: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="ADAN Parallel Multi-Worker Training")
    parser.add_argument("--steps", type=int, default=50000, help="Steps per worker")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config path")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_final", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔥 ADAN PARALLEL MULTI-WORKER TRAINING")
    print("="*80)
    print(f"Workers: 4 (w1, w2, w3, w4)")
    print(f"Steps per worker: {args.steps:,}")
    print(f"Total training steps: {args.steps * 4:,}")
    print(f"Mode: TRUE PARALLEL (4 simultaneous processes)")
    print("="*80 + "\n")
    
    # Create processes for each worker
    processes = []
    worker_ids = ["w1", "w2", "w3", "w4"]
    
    for worker_id in worker_ids:
        p = mp.Process(
            target=train_single_worker,
            args=(worker_id, args.steps, args.checkpoint_dir, args.config)
        )
        p.start()
        processes.append((worker_id, p))
        print(f"✅ Started {worker_id} (PID: {p.pid})")
    
    print("\n⏳ Waiting for all workers to complete...\n")
    
    # Wait for all processes
    results = {}
    for worker_id, p in processes:
        p.join()
        results[worker_id] = p.exitcode == 0
        status = "✅ SUCCESS" if results[worker_id] else "❌ FAILED"
        print(f"{worker_id}: {status}")
    
    print("\n" + "="*80)
    successful = sum(results.values())
    print(f"🎯 TRAINING COMPLETE: {successful}/4 workers successful")
    print("="*80)
    
    if successful == 4:
        print("\n✅ All 4 models ready for ensemble inference!")
        print(f"Run: python scripts/ensemble_manager.py --strategy median")
    
    return successful == 4


if __name__ == "__main__":
    # Required for multiprocessing on Windows/MacOS
    mp.set_start_method('spawn', force=True)
    success = main()
    sys.exit(0 if success else 1)
