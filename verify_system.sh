#!/bin/bash
set -e

echo "1️⃣ VÉRIFICATION DES CORRECTIONS CUSTOMCNN..."
grep -n "MultiScaleResidualBlock" src/adan_trading_bot/models/custom_cnn.py | head -5

echo -e "\n2️⃣ VÉRIFICATION CONFIG.YAML..."
grep -A 20 "^dbe:" config/config.yaml | head -25

echo -e "\n3️⃣ VÉRIFICATION SCORES OPTUNA..."
sqlite3 optuna.db << 'SQL'
SELECT 
    s.study_name as worker,
    t.number as trial,
    tv.value as score,
    t.state
FROM trials t
JOIN studies s ON t.study_id = s.study_id
LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE t.state = 'COMPLETE'
ORDER BY s.study_name, tv.value DESC
LIMIT 8;
SQL

echo -e "\n4️⃣ TEST SMOKE COMPLET..."
/home/morningstar/miniconda3/envs/trading_env/bin/python3.11 << 'TESTEOF'
import sys
import os
sys.path.insert(0, os.path.abspath('src'))

print("🧪 TEST SMOKE COMPLET")
print("=" * 60)

# Test 1: CustomCNN
try:
    from adan_trading_bot.model.custom_cnn import CustomCNN
    import torch
    import gymnasium as gym
    
    # Test forward pass
    # CustomCNN expects a Dict observation space usually, or Box.
    # Let's assume Box for simple test or construct Dict if needed.
    # Based on code, it handles Dict.
    
    spaces = {
        'images': gym.spaces.Box(low=0, high=1, shape=(3, 64, 64), dtype=float)
    }
    observation_space = gym.spaces.Dict(spaces)
    
    cnn = CustomCNN(observation_space=observation_space, features_dim=256)
    test_input = {'images': torch.randn(1, 3, 64, 64)}
    output = cnn(test_input)
    
    print(f"✅ CustomCNN: Input {test_input['images'].shape} -> Output {output.shape}")
except Exception as e:
    print(f"❌ CustomCNN: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Environment
try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from adan_trading_bot.common.config_loader import ConfigLoader
    
    config = ConfigLoader().load_config('config/config.yaml')
    # Ensure we don't crash on missing data by using a robust config or existing data
    env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    
    obs, info = env.reset()
    print(f"✅ Environment Reset: Obs keys {list(obs.keys()) if isinstance(obs, dict) else obs.shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Environment Step: Reward {reward}")
    
    env.close()
    
except Exception as e:
    print(f"❌ Environment: {e}")
    import traceback
    traceback.print_exc()

# Test 3: PPO Integration
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env():
        return MultiAssetChunkedEnv(config=config, worker_id=0, log_level="ERROR")
    
    vec_env = DummyVecEnv([make_env])
    
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=0.0003,
        n_steps=128,
        verbose=0
    )
    
    print("✅ PPO Model Created")
    
    # Test très court apprentissage
    model.learn(total_timesteps=100)
    print("✅ Short Training Successful")
    
    # Test prédiction
    obs = vec_env.reset()
    action, _states = model.predict(obs)
    print(f"✅ Prediction Successful - Action: {action}")
    
    vec_env.close()
    
except Exception as e:
    print(f"❌ PPO Integration: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
print("🎯 TEST SMOKE TERMINÉ")
TESTEOF
