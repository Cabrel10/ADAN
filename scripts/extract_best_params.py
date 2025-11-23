import optuna
import yaml
import os
import sys

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    storage = "sqlite:///optuna.db"
    config_path = "config/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    config = load_config(config_path)
    workers = ["w1", "w2", "w3", "w4"]
    
    report = []
    
    for w in workers:
        study_name = f"adan_final_v1_{w}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            
            # Filter for completed trials first
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not completed_trials:
                # Fallback to pruned trials if they have a value (score)
                pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                valid_pruned = []
                for t in pruned_trials:
                    score = t.value
                    if score is None:
                        score = t.user_attrs.get("score")
                    if score is not None:
                        # Monkey patch value for sorting
                        t.value = float(score)
                        valid_pruned.append(t)
                
                if valid_pruned:
                    print(f"Warning: No completed trials for {w}, using best pruned trial.")
                    best_trial = max(valid_pruned, key=lambda t: t.value)
                else:
                    print(f"Error: No valid trials for {w}")
                    report.append(f"{w}: FAILED (No trials)")
                    continue
            else:
                best_trial = study.best_trial
            
            print(f"Worker {w} Best Trial: {best_trial.number} (Score: {best_trial.value:.4f})")
            
            # Extract params
            params = best_trial.params
            
            # Update config for this worker
            if w not in config["workers"]:
                print(f"Warning: Worker {w} not found in config")
                continue
                
            worker_cfg = config["workers"][w]
            
            # Update Trading Params
            worker_cfg["position_size_pct"] = params.get("position_size_pct", worker_cfg.get("position_size_pct"))
            worker_cfg["risk_multiplier"] = params.get("risk_multiplier", worker_cfg.get("risk_multiplier"))
            
            # Update SL/TP per tier
            if "stop_loss_pct_by_tier" not in worker_cfg:
                worker_cfg["stop_loss_pct_by_tier"] = {}
            if "take_profit_pct_by_tier" not in worker_cfg:
                worker_cfg["take_profit_pct_by_tier"] = {}
                
            for tier in ["Micro", "Small", "Medium", "High", "Enterprise"]:
                worker_cfg["stop_loss_pct_by_tier"][tier] = params.get("stop_loss_pct", 0.05)
                worker_cfg["take_profit_pct_by_tier"][tier] = params.get("take_profit_pct", 0.10)
            
            # Update Reward Config
            if "reward_config" not in worker_cfg:
                worker_cfg["reward_config"] = {}
            
            reward_keys = ["pnl_weight", "win_rate_bonus", "stop_loss_penalty", "take_profit_bonus"]
            for k in reward_keys:
                if k in params:
                    worker_cfg["reward_config"][k] = params[k]
            
            # Update Agent Params (Global or Worker specific?)
            # The optimization script updated GLOBAL agent params.
            # But here we might want to keep them global or per worker if supported.
            # Config structure usually has 'agent' at top level.
            # We will update the global agent config with the best params from the BEST worker overall?
            # Or just average? 
            # For now, let's NOT update global agent params to avoid conflict between workers,
            # unless we pick the "best of best".
            # But the user asked to inject hyperparameters "for each worker".
            # Agent params (learning_rate, etc) are usually global in this config.
            # Let's leave agent params alone for now, or update them if w1 (primary) is best.
            
            report.append(f"{w}: SUCCESS (Trial {best_trial.number}, Score {best_trial.value:.4f})")
            
        except Exception as e:
            print(f"Error processing {w}: {e}")
            report.append(f"{w}: ERROR ({str(e)})")

    # Save updated config
    save_config(config, config_path)
    print("\nConfig updated successfully.")
    
    print("\n=== FINAL REPORT ===")
    for line in report:
        print(line)

if __name__ == "__main__":
    main()
