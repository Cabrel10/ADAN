#!/usr/bin/env python3
"""Extract best Optuna results and save to YAML."""
import sys
import sqlite3
import json
import yaml
from pathlib import Path

OPTUNA_RESULTS_DIR = Path("/home/morningstar/Documents/trading/bot/optuna_results")
TRADING_PARAMS = {
    "w1": {
        "stop_loss_pct": 0.0776,
        "take_profit_pct": 0.1056,
        "risk_per_trade_pct": 0.02,
        "position_size_pct": 0.5,
        "max_concurrent_positions": 1,
        "min_holding_period_steps": 5,
    },
    "w2": {
        "stop_loss_pct": 0.0776,
        "take_profit_pct": 0.1056,
        "risk_per_trade_pct": 0.02,
        "position_size_pct": 0.5,
        "max_concurrent_positions": 1,
        "min_holding_period_steps": 5,
    },
    "w3": {
        "stop_loss_pct": 0.0922,
        "take_profit_pct": 0.1248,
        "risk_per_trade_pct": 0.02,
        "position_size_pct": 0.5,
        "max_concurrent_positions": 1,
        "min_holding_period_steps": 5,
    },
    "w4": {
        "stop_loss_pct": 0.0884,
        "take_profit_pct": 0.1456,
        "risk_per_trade_pct": 0.02,
        "position_size_pct": 0.5,
        "max_concurrent_positions": 1,
        "min_holding_period_steps": 5,
    },
}


def extract_best_trial(worker: str) -> dict:
    """Extract best trial from Optuna DB."""
    # Find DB file
    db_files = list(OPTUNA_RESULTS_DIR.glob(f"{worker.upper()}_ppo_*.db"))
    if not db_files:
        print(f"❌ No DB found for {worker}")
        return None

    db_path = db_files[-1]  # Latest DB
    print(f"📂 Using DB: {db_path.name}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get best trial
    cursor.execute("""
        SELECT t.trial_id, tv.objective
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state='COMPLETE'
        ORDER BY tv.objective DESC LIMIT 1;
    """)
    best = cursor.fetchone()

    if not best:
        print(f"❌ No COMPLETE trials for {worker}")
        conn.close()
        return None

    trial_id, score = best
    print(f"✅ Best trial: {trial_id}, score={score}")

    # Get params
    cursor.execute("""
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id=?;
    """, (trial_id,))

    params = {}
    for name, value in cursor.fetchall():
        if name == "n_steps_exp":
            params["n_steps"] = int(2 ** value)
        elif name == "batch_exp":
            params["batch_size"] = int(2 ** value)
        else:
            params[name] = value

    # Get metrics
    cursor.execute("""
        SELECT key, value_json
        FROM trial_user_attributes
        WHERE trial_id=?
        ORDER BY key;
    """, (trial_id,))

    metrics = {}
    for key, value_json in cursor.fetchall():
        try:
            val = json.loads(value_json)
        except Exception:
            val = value_json
        metrics[key] = val

    conn.close()

    # Build result
    result = {
        "worker": worker.upper(),
        "phase": "Phase 2 - PPO Hyperparams",
        "score": float(score),
        "ppo_parameters": params,
        "trading_parameters": TRADING_PARAMS.get(worker.lower(), {}),
        "metrics": metrics,
    }

    return result


def main():
    """Extract all results and save to YAML."""
    print("🚀 EXTRACTING OPTUNA RESULTS\n")

    for worker in ["W1", "W2", "W3", "W4"]:
        print(f"\n{'='*60}")
        print(f"Processing {worker}")
        print(f"{'='*60}")

        result = extract_best_trial(worker.lower())
        if not result:
            print(f"⏭️  Skipping {worker}")
            continue

        # Save to YAML
        output_file = OPTUNA_RESULTS_DIR / f"{worker}_ppo_best_params.yaml"
        with open(output_file, "w") as f:
            yaml.dump(result, f, default_flow_style=False)

        print(f"✅ Saved to: {output_file}")
        print(f"\n🔧 PPO Parameters:")
        for k, v in result["ppo_parameters"].items():
            print(f"  {k}: {v}")
        print(f"\n📊 Metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")

    print(f"\n{'='*60}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
