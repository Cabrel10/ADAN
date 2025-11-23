import optuna
import pandas as pd
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_study_summary(study_name, storage):
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        trials = study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
        
        if not completed:
            status = "Waiting..."
            if failed:
                status = f"Failed ({len(failed)})"
            elif pruned:
                last_pruned = pruned[-1]
                reason = last_pruned.user_attrs.get("reject_reason", "Unknown")
                status = f"Pruned ({len(pruned)}): {reason[:20]}..."
            
            return {
                "worker": study_name.split("_")[-1],
                "trials": 0,
                "pruned": len(pruned),
                "failed": len(failed),
                "best_score": 0.0,
                "last_score": 0.0,
                "last_trades": 0,
                "status": status
            }
            
        best_trial = study.best_trial
        last_trial = completed[-1]
        
        last_trades = last_trial.user_attrs.get("num_trades", "N/A")
        reject_reason = last_trial.user_attrs.get("reject_reason", "")
        
        status = "Running"
        if reject_reason:
            status = f"Rejected: {reject_reason[:20]}..."
        
        return {
            "worker": study_name.split("_")[-1],
            "trials": len(completed),
            "pruned": len(pruned),
            "failed": len(failed),
            "best_score": best_trial.value,
            "last_score": last_trial.value,
            "last_trades": last_trades,
            "status": status
        }
    except Exception as e:
        return {
            "worker": study_name.split("_")[-1],
            "trials": 0,
            "best_score": 0.0,
            "last_score": 0.0,
            "last_trades": 0,
            "status": f"Error: {str(e)}"
        }

def main():
    storage = "sqlite:///optuna.db"
    workers = ["w1", "w2", "w3", "w4"]
    
    print(f"{'WORKER':<10} | {'TRIALS':<10} | {'PRUNED':<10} | {'FAILED':<10} | {'BEST SCORE':<12} | {'LAST SCORE':<12} | {'LAST TRADES':<12} | {'STATUS':<30}")
    print("-" * 125)
    
    for w in workers:
        study_name = f"adan_final_v1_{w}"
        summary = get_study_summary(study_name, storage)
        print(f"{summary['worker']:<10} | {summary['trials']:<10} | {summary['pruned']:<10} | {summary['failed']:<10} | {summary['best_score']:<12.4f} | {summary['last_score']:<12.4f} | {str(summary['last_trades']):<12} | {summary['status']:<30}")

if __name__ == "__main__":
    main()
