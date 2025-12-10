#!/usr/bin/env python3
"""Load Optuna results into config.yaml."""
import sys
import yaml
from pathlib import Path

OPTUNA_RESULTS_DIR = Path("/home/morningstar/Documents/trading/bot/optuna_results")
CONFIG_FILE = Path("/home/morningstar/Documents/trading/bot/config/config.yaml")


def load_yaml_results(worker: str) -> dict:
    """Load YAML results for a worker."""
    yaml_file = OPTUNA_RESULTS_DIR / f"{worker}_ppo_best_params.yaml"
    if not yaml_file.exists():
        print(f"❌ {yaml_file} not found")
        return None

    with open(yaml_file) as f:
        return yaml.safe_load(f)


def main():
    """Load all Optuna results into config.yaml."""
    print("🚀 LOADING OPTUNA RESULTS INTO CONFIG\n")

    # Load config
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    # Ensure workers section exists
    if "workers" not in config:
        config["workers"] = {}

    # Load each worker
    for worker in ["W1", "W2", "W3", "W4"]:
        worker_key = worker.lower()
        print(f"\n{'='*60}")
        print(f"Loading {worker}")
        print(f"{'='*60}")

        result = load_yaml_results(worker)
        if not result:
            print(f"⏭️  Skipping {worker}")
            continue

        # Extract PPO params
        ppo_params = result.get("ppo_parameters", {})
        trading_params = result.get("trading_parameters", {})
        metrics = result.get("metrics", {})

        # Update config
        if worker_key not in config["workers"]:
            config["workers"][worker_key] = {}

        config["workers"][worker_key]["agent_config"] = {
            "batch_size": ppo_params.get("batch_size", 64),
            "clip_range": ppo_params.get("clip_range", 0.2),
            "ent_coef": ppo_params.get("ent_coef", 0.01),
            "gamma": ppo_params.get("gamma", 0.99),
            "gae_lambda": ppo_params.get("gae_lambda", 0.95),
            "learning_rate": ppo_params.get("learning_rate", 3e-4),
            "max_grad_norm": ppo_params.get("max_grad_norm", 0.5),
            "n_epochs": int(ppo_params.get("n_epochs", 10)),
            "n_steps": ppo_params.get("n_steps", 2048),
            "vf_coef": ppo_params.get("vf_coef", 0.5),
        }

        config["workers"][worker_key]["trading_parameters"] = trading_params

        print(f"✅ Loaded {worker}")
        print(f"\n🔧 PPO Parameters:")
        for k, v in config["workers"][worker_key]["agent_config"].items():
            print(f"  {k}: {v}")
        print(f"\n📊 Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    # Save config
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print(f"✅ CONFIG UPDATED: {CONFIG_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
