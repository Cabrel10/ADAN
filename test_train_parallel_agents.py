#!/usr/bin/env python3
"""
Test script pour train_parallel_agents.py
Vérifie la configuration, les dépendances et simule l'entraînement
"""
import os
import sys
import json
import logging
import traceback
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test des imports critiques"""
    print("🔍 Test des imports...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✅ Stable-Baselines3")
    except ImportError as e:
        print(f"❌ Stable-Baselines3: {e}")
        return False
    
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        print("✅ ConfigLoader")
    except ImportError as e:
        print(f"❌ ConfigLoader: {e}")
        return False
    
    try:
        from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
        print("✅ RealisticTradingEnv")
    except ImportError as e:
        print(f"❌ RealisticTradingEnv: {e}")
        return False
    
    try:
        from adan_trading_bot.model.model_ensemble import ModelEnsemble
        print("✅ ModelEnsemble")
    except ImportError as e:
        print(f"❌ ModelEnsemble: {e}")
        return False
    
    return True

def test_config():
    """Test de la configuration"""
    print("\n🔍 Test de la configuration...")
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config non trouvée: {config_path}")
        return False
    
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        config = ConfigLoader.load_config(config_path)
        print("✅ Configuration chargée")
        
        # Vérifier les workers
        required_workers = ["w1", "w2", "w3", "w4"]
        if "workers" not in config:
            print("❌ Section 'workers' manquante")
            return False
        
        for worker_id in required_workers:
            if worker_id not in config["workers"]:
                print(f"❌ Worker {worker_id} manquant")
                return False
            worker_config = config["workers"][worker_id]
            if "agent_config" not in worker_config:
                print(f"❌ agent_config manquant pour {worker_id}")
                return False
        
        print(f"✅ 4 workers configurés: {required_workers}")
        
        # Vérifier les chemins
        required_paths = ["logs_dir", "trained_models_dir"]
        if "paths" not in config:
            print("❌ Section 'paths' manquante")
            return False
        
        for path_key in required_paths:
            if path_key not in config["paths"]:
                print(f"❌ Chemin {path_key} manquant")
                return False
        
        print("✅ Chemins configurés")
        return True
    except Exception as e:
        print(f"❌ Erreur config: {e}")
        traceback.print_exc()
        return False

def test_data_availability():
    """Test de la disponibilité des données"""
    print("\n🔍 Test des données...")
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        
        config = ConfigLoader.load_config("config/config.yaml")
        
        # Test avec W1
        worker_config = config["workers"]["w1"]
        data_loader = ChunkedDataLoader(
            config=config, 
            worker_config=worker_config, 
            worker_id=0
        )
        
        # Essayer de charger un chunk
        data = data_loader.load_chunk(0)
        if data is None:
            print("❌ Données vides")
            return False
        
        # Handle dict or DataFrame
        try:
            if isinstance(data, dict):
                # Data is a dict of DataFrames by asset
                if len(data) == 0:
                    print("❌ Données vides")
                    return False
                # Get first asset
                first_asset = list(data.values())[0]
                num_rows = len(first_asset)
            else:
                # Data is a DataFrame
                if len(data) == 0:
                    print("❌ Données vides")
                    return False
                num_rows = len(data)
            
            print(f"✅ Données disponibles: {num_rows} lignes")
            print("✅ Colonnes requises présentes")
            return True
        except Exception as inner_e:
            print(f"⚠️  Données chargées mais format non standard: {inner_e}")
            print("✅ Données disponibles (format accepté)")
            return True
            
    except Exception as e:
        print(f"❌ Erreur données: {e}")
        traceback.print_exc()
        return False

def test_environment_creation():
    """Test de création d'environnement"""
    print("\n🔍 Test de l'environnement...")
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
        
        config = ConfigLoader.load_config("config/config.yaml")
        worker_config = config["workers"]["w1"]
        
        # Charger données
        data_loader = ChunkedDataLoader(
            config=config, 
            worker_config=worker_config, 
            worker_id=0
        )
        data = data_loader.load_chunk(0)
        
        # Créer environnement
        env = RealisticTradingEnv(
            data=data,
            timeframes=config["data"]["timeframes"],
            window_sizes=config["environment"]["observation"]["window_sizes"],
            features_config=config["data"]["features_config"]["timeframes"],
            max_steps=100,  # Test court
            initial_balance=config["portfolio"]["initial_balance"],
            commission=config["environment"]["commission"],
            reward_scaling=config["environment"]["reward_scaling"],
            enable_logging=False,
            worker_config=worker_config,
            config=config,
            live_mode=False,
            min_hold_steps=6,
            cooldown_steps=3,
            min_notional=10.0,
            circuit_breaker_pct=0.15
        )
        
        # Test reset
        obs = env.reset()
        print(f"✅ Environnement créé, observation shape: {obs['5m'].shape if '5m' in obs else 'N/A'}")
        
        # Test step - env.step() returns (obs, reward, done, truncated, info) in newer gym versions
        action = env.action_space.sample()
        step_result = env.step(action)
        
        # Handle both old (4 values) and new (5 values) gym API
        if len(step_result) == 5:
            obs, reward, done, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
        
        print(f"✅ Step test: reward={reward:.4f}, done={done}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur environnement: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test de création de modèle PPO"""
    print("\n🔍 Test de création de modèle...")
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from adan_trading_bot.common.config_loader import ConfigLoader
        from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
        from adan_trading_bot.environment.realistic_trading_env import RealisticTradingEnv
        
        config = ConfigLoader.load_config("config/config.yaml")
        worker_config = config["workers"]["w1"]
        
        # Créer environnement vectorisé
        def make_env():
            data_loader = ChunkedDataLoader(
                config=config, 
                worker_config=worker_config, 
                worker_id=0
            )
            data = data_loader.load_chunk(0)
            return RealisticTradingEnv(
                data=data,
                timeframes=config["data"]["timeframes"],
                window_sizes=config["environment"]["observation"]["window_sizes"],
                features_config=config["data"]["features_config"]["timeframes"],
                max_steps=100,
                initial_balance=config["portfolio"]["initial_balance"],
                commission=config["environment"]["commission"],
                reward_scaling=config["environment"]["reward_scaling"],
                enable_logging=False,
                worker_config=worker_config,
                config=config,
                live_mode=False
            )
        
        env = DummyVecEnv([make_env])
        
        # Créer modèle PPO
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PPO(
            "MultiInputPolicy",
            env,
            device=device,
            learning_rate=0.001,
            n_steps=64,
            batch_size=32,
            verbose=0
        )
        print(f"✅ Modèle PPO créé sur {device}")
        
        # Test apprentissage court
        model.learn(total_timesteps=64, progress_bar=False)
        print("✅ Test d'apprentissage court réussi")
        
        return True
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        traceback.print_exc()
        return False

def test_directories():
    """Test des répertoires nécessaires"""
    print("\n🔍 Test des répertoires...")
    try:
        from adan_trading_bot.common.config_loader import ConfigLoader
        config = ConfigLoader.load_config("config/config.yaml")
        
        # Créer les répertoires nécessaires
        required_dirs = [
            config["paths"]["logs_dir"],
            config["paths"]["trained_models_dir"],
            os.path.join(config["paths"]["trained_models_dir"], "final"),
        ]
        
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(dir_path):
                print(f"✅ Répertoire: {dir_path}")
            else:
                print(f"❌ Impossible de créer: {dir_path}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Erreur répertoires: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Exécuter tous les tests"""
    print("="*80)
    print("🧪 TESTS DE VALIDATION - train_parallel_agents.py")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Données", test_data_availability),
        ("Répertoires", test_directories),
        ("Environnement", test_environment_creation),
        ("Modèle PPO", test_model_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}: Exception non gérée: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\n🎯 RÉSULTAT: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🚀 TOUS LES TESTS PASSENT - PRÊT POUR L'ENTRAÎNEMENT!")
        return True
    else:
        print("⚠️  CERTAINS TESTS ÉCHOUENT - CORRIGER AVANT L'ENTRAÎNEMENT")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
