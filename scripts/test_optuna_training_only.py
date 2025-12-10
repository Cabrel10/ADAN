"""
Test d'intégration ciblé : vérifier si evaluate_ppo_params_robust
génère des trades PENDANT L'ENTRAÎNEMENT (pas l'éval).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.optuna_evaluation import evaluate_ppo_params_robust

print("=" * 80)
print("TEST INTÉGRATION: evaluate_ppo_params_robust (TRAINING ONLY)")
print("=" * 80)

# 1. Config avec limites augmentées
config = ConfigLoader.load_config("config/config.yaml")
config['trading_rules']['frequency'] = {
    'force_trade_steps': {'5m': 15, '1h': 30, '4h': 60},
    'daily_max_total': 500,
    'daily_max_by_tf': {'5m': 200, '1h': 200, '4h': 200},
}
config['trading_rules']['daily_max_forced_trades'] = 500

# 2. Créer env
env = MultiAssetChunkedEnv(config=config)
print(f"\n✅ Env créé (worker_id={env.worker_id})")

# 3. PPO params
ppo_params = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# 4. Évaluation avec TRAINING SEULEMENT (eval_steps=0)
print("\n[PHASE 1] Entraînement seul (training_steps=3000, eval_steps=0)...")
print("   (ceci prend environ 5-10 min...)\n")
try:
    metrics_training_only = evaluate_ppo_params_robust(
        env=env,
        ppo_params=ppo_params,
        training_steps=3_000,
        eval_steps=0,  # PAS D'ÉVALUATION
    )
except Exception as e:
    print(f"\n❌ ERREUR pendant evaluate_ppo_params_robust: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("📊 RÉSULTATS TRAINING ONLY:")
print("=" * 80)
for k, v in sorted(metrics_training_only.items()):
    print(f"  {k:20s}: {v}")

# 5. Vérifier directement dans env
print("\n" + "=" * 80)
print("🔍 INSPECTION ENV POST-TRAINING:")
print("=" * 80)

pm = env.portfolio_manager
print(f"  metrics.trades: {len(pm.metrics.trades)}")
print(f"  metrics.closed_positions: {len(pm.metrics.closed_positions)}")
print(f"  portfolio.equity: {pm.equity}")

# 6. Validation
print("\n" + "=" * 80)
print("✅ VALIDATION:")
print("=" * 80)

checks = {
    "metrics.trades > 0": len(pm.metrics.trades) > 0,
    "metrics.closed_positions > 0": len(pm.metrics.closed_positions) > 0,
    "total_trades (returned) > 0": metrics_training_only.get("total_trades", 0) > 0,
    "sharpe_ratio != 0": metrics_training_only.get("sharpe_ratio", 0) != 0,
}

all_pass = True
for check_name, result in checks.items():
    status = "✅" if result else "❌"
    print(f"  {status} {check_name}")
    if not result:
        all_pass = False

print("\n" + "=" * 80)
if all_pass:
    print("✅✅✅ TOUS LES TESTS PASSENT")
    print("\n💡 CONCLUSION: Le problème est dans la boucle d'évaluation séparée.")
    print("   SOLUTION: Utiliser directement metrics post-training pour Optuna.")
else:
    print("❌ CERTAINS TESTS ÉCHOUENT")
    print("\n💡 CONCLUSION: Le problème est déjà pendant l'entraînement.")
    print("   Il faut investiguer pourquoi model.learn() ne génère pas de trades.")
print("=" * 80)

sys.exit(0 if all_pass else 1)
