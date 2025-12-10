#!/bin/bash
# Diagnostic complet du projet ADAN avec environnement conda

set -e

echo "🔍 DIAGNOSTIC COMPLET DU PROJET ADAN"
echo "===================================="
echo ""

# Activer l'environnement conda
echo "📦 Activation de l'environnement conda trading_env..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trading_env
echo "✅ Environnement activé: $(python --version)"
echo ""

# Vérifier la configuration
echo "📋 1️⃣  VÉRIFICATION CONFIG.YAML"
echo "================================"
python3 << 'EOF'
import yaml
from pathlib import Path

config_path = Path("config/config.yaml")
if not config_path.exists():
    print(f"❌ config.yaml NOT FOUND")
    exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

print("✅ config.yaml loaded")
print("")

# Check critical sections
print("Sections critiques:")
for key in ["agent", "environment", "training", "trading_rules", "palier_tiers"]:
    if key in config:
        print(f"  ✅ {key}: present")
    else:
        print(f"  ❌ {key}: MISSING")

print("")
print("Hyperparamètres PPO:")
if "agent" in config and "ppo" in config["agent"]:
    ppo = config["agent"]["ppo"]
    lr = ppo.get("learning_rate", "N/A")
    mgn = ppo.get("max_grad_norm", "N/A")
    cr = ppo.get("clip_range", "N/A")
    print(f"  - learning_rate: {lr}")
    print(f"  - max_grad_norm: {mgn}")
    print(f"  - clip_range: {cr}")
    
    if isinstance(lr, (int, float)) and lr > 0.001:
        print(f"  ⚠️  Learning rate TOO HIGH: {lr} (should be < 0.001)")

print("")
print("Force Trade:")
if "trading_rules" in config and "force_trade" in config["trading_rules"]:
    ft = config["trading_rules"]["force_trade"].get("enabled", False)
    if ft:
        print(f"  ✅ force_trade.enabled = True")
    else:
        print(f"  ❌ force_trade.enabled = False (MUST BE TRUE)")
EOF

echo ""
echo "📊 2️⃣  VÉRIFICATION ENVIRONNEMENT"
echo "=================================="
timeout 60 python3 << 'EOF'
import sys
import logging

logging.basicConfig(level=logging.WARNING)

try:
    print("Importing modules...")
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.common.config_loader import ConfigLoader
    print("✅ Imports successful")
    
    print("Loading config...")
    config = ConfigLoader.load_config("config/config.yaml")
    print("✅ Config loaded")
    
    print("Creating environment...")
    env = MultiAssetChunkedEnv(
        config=config,
        worker_id=0,
        verbose=False
    )
    print("✅ Environment created")
    
    print("Resetting environment...")
    obs = env.reset()
    print(f"✅ Environment reset")
    
    print("Taking 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        step_result = env.step(action)
        # Handle both Gym (4 values) and Gymnasium (5 values)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        if i == 0:
            print(f"  Step 0: reward={reward:.4f}")
        if done:
            break
    
    print("✅ Environment stable")
    env.close()
    
except Exception as e:
    print(f"❌ Environment error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "💰 3️⃣  VÉRIFICATION REWARD FUNCTION"
echo "===================================="
timeout 60 python3 << 'EOF'
import sys
import logging

logging.basicConfig(level=logging.WARNING)

try:
    from adan_trading_bot.environment.reward_calculator import RewardCalculator
    from adan_trading_bot.common.config_loader import ConfigLoader
    
    config = ConfigLoader.load_config("config/config.yaml")
    print("✅ Config loaded")
    
    # RewardCalculator expects env_config with reward_shaping
    env_section = config.get("environment", {})
    reward_cfg = env_section.get("reward_config", {})
    env_config = {"reward_shaping": reward_cfg}
    
    rc = RewardCalculator(env_config=env_config)
    print("✅ RewardCalculator created")
    
    # Test reward calculation
    print("✅ RewardCalculator ready")
    
except Exception as e:
    print(f"❌ Reward function error: {e}")
    sys.exit(1)
EOF

echo ""
echo "📊 4️⃣  VÉRIFICATION LOGGING"
echo "============================"
timeout 60 python3 << 'EOF'
import sys
from pathlib import Path

try:
    from adan_trading_bot.common.central_logger import CentralLogger
    from adan_trading_bot.performance.unified_metrics_db import UnifiedMetricsDB
    
    print("✅ Imports successful")
    
    db_path = Path("metrics.db")
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        print(f"✅ metrics.db exists ({size_kb:.1f} KB)")
    else:
        print(f"⚠️  metrics.db not found (will be created)")
    
    db = UnifiedMetricsDB()
    print("✅ UnifiedMetricsDB created")
    
    cl = CentralLogger()
    print("✅ CentralLogger created")
    
    cl.metric("test_metric", 1.0, {"worker": "test"})
    print("✅ Metric logged")
    
except Exception as e:
    print(f"❌ Logging error: {e}")
    sys.exit(1)
EOF

echo ""
echo "🔧 5️⃣  VÉRIFICATION OPTUNA"
echo "=========================="
timeout 60 python3 << 'EOF'
import sys
from pathlib import Path

try:
    import optuna
    
    optuna_db = Path("optuna.db")
    if optuna_db.exists():
        print(f"✅ optuna.db exists")
        
        try:
            storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
            study_names = storage.get_study_names()
            print(f"  Found {len(study_names)} studies: {study_names}")
        except Exception as e:
            print(f"⚠️  Could not read optuna.db: {e}")
    else:
        print(f"⚠️  optuna.db not found")
    
except Exception as e:
    print(f"❌ Optuna error: {e}")
    sys.exit(1)
EOF

echo ""
echo "📈 6️⃣  VÉRIFICATION LOGS D'ENTRAÎNEMENT"
echo "======================================"
LATEST_LOG=$(ls -t /mnt/new_data/adan_logs/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "✅ Latest training log: $(basename $LATEST_LOG)"
    echo "  Size: $(du -h $LATEST_LOG | cut -f1)"
    echo ""
    echo "  Dernières lignes:"
    tail -5 "$LATEST_LOG" | sed 's/^/    /'
else
    echo "⚠️  No training logs found"
fi

echo ""
echo "🏁 DIAGNOSTIC TERMINÉ"
echo "====================="
echo ""
echo "📝 Voir DIAGNOSTIC_COMPLET.md pour plus de détails"
