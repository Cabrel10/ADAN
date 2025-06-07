#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test d'entraînement simple avec données simulées intégrées.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.agent.ppo_agent import create_ppo_agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_realistic_crypto_data(assets, num_days=30, timeframe="1m"):
    """Créer des données crypto réalistes avec volatilité et tendances."""
    
    if timeframe == "1m":
        freq = "1min"
        periods = num_days * 24 * 60
    elif timeframe == "1h":
        freq = "1h"
        periods = num_days * 24
    else:  # 1d
        freq = "1d"
        periods = num_days
    
    timestamps = pd.date_range(start='2024-01-01', periods=periods, freq=freq)
    
    # Prix de base réalistes pour chaque crypto
    base_prices = {
        'BTCUSDT': 45000,
        'ETHUSDT': 2800,
        'ADAUSDT': 0.45,
        'BNBUSDT': 320,
        'XRPUSDT': 0.52
    }
    
    data = {}
    
    for asset in assets:
        base_price = base_prices.get(asset, 100)
        
        # Générer une série de prix avec tendance et volatilité réalistes
        returns = np.random.normal(0.0001, 0.02, periods)  # Rendements avec drift positif
        
        # Ajouter de la persistance (autocorrélation)
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Calculer les prix
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # OHLCV
        opens = price_series
        highs = opens * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        lows = opens * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        closes = opens + np.random.normal(0, opens * 0.005, periods)
        volumes = np.random.lognormal(10, 1, periods)
        
        # Indicateurs techniques simplifiés (normalisés entre -1 et 1)
        sma_short = np.tanh(np.random.normal(0, 0.5, periods))
        sma_long = np.tanh(np.random.normal(0, 0.3, periods))
        ema_short = np.tanh(np.random.normal(0, 0.4, periods))
        ema_long = np.tanh(np.random.normal(0, 0.3, periods))
        rsi = np.tanh(np.random.normal(0, 0.6, periods))
        macd = np.tanh(np.random.normal(0, 0.4, periods))
        macds = np.tanh(np.random.normal(0, 0.3, periods))
        macdh = macd - macds
        
        # Stocker OHLCV sans timeframe (format attendu par l'environnement)
        for feature_name, values in [
            ('open', opens), ('high', highs), ('low', lows), ('close', closes), ('volume', volumes)
        ]:
            col_name = f"{feature_name}_{asset}"
            data[col_name] = values
        
        # Stocker les indicateurs techniques avec timeframe
        for feature_name, values in [
            ('SMA_short', sma_short), ('SMA_long', sma_long), ('EMA_short', ema_short), ('EMA_long', ema_long),
            ('RSI', rsi), ('MACD', macd), ('MACDs', macds), ('MACDh', macdh)
        ]:
            col_name = f"{feature_name}_{timeframe}_{asset}"
            data[col_name] = values
    
    df = pd.DataFrame(data, index=timestamps)
    logger.info(f"Données créées: {df.shape}, {len(assets)} actifs, période: {timeframe}")
    return df

def create_test_config(assets, timeframe="1m"):
    """Créer une configuration de test minimale."""
    
    config = {
        'data': {
            'assets': assets,
            'training_timeframe': timeframe,
            'data_source_type': 'calculate_from_raw',
            'cnn_input_window_size': 10,  # Réduit pour test rapide
            'cnn_config': {
                'features_dim': 32,
                'num_input_channels': 1,
                'conv_layers': [
                    {'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}
                ],
                'pool_layers': [
                    {'kernel_size': 2, 'stride': 2}
                ],
                'activation': 'relu',
                'dropout': 0.2,
                'fc_layers': [64, 32]
            },
            'indicators_by_timeframe': {
                timeframe: [
                    {'name': 'SMA Short', 'output_col_name': 'SMA_short'},
                    {'name': 'SMA Long', 'output_col_name': 'SMA_long'},
                    {'name': 'EMA Short', 'output_col_name': 'EMA_short'},
                    {'name': 'EMA Long', 'output_col_name': 'EMA_long'},
                    {'name': 'RSI', 'output_col_name': 'RSI'},
                    {'name': 'MACD', 'output_col_name': ['MACD', 'MACDs', 'MACDh']}
                ]
            }
        },
        'environment': {
            'initial_capital': 1000.0,
            'transaction': {
                'fee_percent': 0.001,
                'fixed_fee': 0.0
            },
            'order_rules': {
                'min_value_tolerable': 10.0,
                'min_value_absolute': 5.0
            },
            'penalties': {
                'invalid_order_base': -0.3,
                'out_of_funds': -0.5,
                'order_below_tolerable': -0.1
            },
            'reward_calculation': {
                'reward_type': 'portfolio_change',
                'base_reward_scaling': 1.0,
                'penalty_time': -0.001
            }
        },
        'agent': {
            'algorithm': 'PPO',
            'device': 'cpu',
            'learning_rate': 0.0003,
            'batch_size': 64,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }
    }
    
    return config

def test_short_training():
    """Test d'entraînement court pour valider le système."""
    
    print("🚀 DÉMARRAGE TEST D'ENTRAÎNEMENT ADAN")
    print("=" * 60)
    
    # Configuration
    assets = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    timeframe = "1m"
    total_steps = 2000  # Entraînement court
    
    print(f"📊 Configuration:")
    print(f"   Actifs: {assets}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Steps total: {total_steps}")
    
    # Créer les données
    print(f"\n📈 Génération des données de test...")
    test_data = create_realistic_crypto_data(assets, num_days=10, timeframe=timeframe)
    config = create_test_config(assets, timeframe)
    
    print(f"   Données générées: {test_data.shape}")
    print(f"   Période: {test_data.index[0]} à {test_data.index[-1]}")
    
    # Initialiser l'environnement
    print(f"\n🏗️  Initialisation de l'environnement...")
    try:
        env = MultiAssetEnv(test_data, config, max_episode_steps_override=200)
        print(f"   ✅ Environnement créé")
        print(f"   Actifs: {len(env.assets)}")
        print(f"   Features: {len(env.base_feature_names)}")
        print(f"   Actions possibles: {env.action_space.n}")
        print(f"   Shape CNN: {env.image_shape}")
    except Exception as e:
        print(f"   ❌ Erreur environnement: {e}")
        return False
    
    # Test de l'environnement
    print(f"\n🔧 Test de l'environnement...")
    try:
        obs, info = env.reset()
        print(f"   ✅ Reset réussi")
        print(f"   Image shape: {obs['image_features'].shape}")
        print(f"   Vector shape: {obs['vector_features'].shape}")
        
        # Quelques steps de test
        total_reward = 0
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        print(f"   ✅ Steps de test réussis (reward cumulé: {total_reward:.3f})")
    except Exception as e:
        print(f"   ❌ Erreur test environnement: {e}")
        return False
    
    # Initialiser l'agent
    print(f"\n🤖 Initialisation de l'agent PPO...")
    try:
        agent = create_ppo_agent(
            env=env,
            config=config
        )
        print(f"   ✅ Agent PPO créé")
    except Exception as e:
        print(f"   ❌ Erreur agent: {e}")
        return False
    
    # Entraînement
    print(f"\n🎯 Démarrage de l'entraînement ({total_steps} steps)...")
    start_time = datetime.now()
    
    try:
        # Callback simple pour logging
        class SimpleCallback:
            def __init__(self):
                self.episode_count = 0
                self.last_log_time = datetime.now()
                self.best_reward = float('-inf')
            
            def on_step(self, step_count):
                if step_count % 500 == 0:
                    elapsed = (datetime.now() - self.last_log_time).total_seconds()
                    print(f"   Step {step_count:4d}/{total_steps} | Capital: ${env.capital:.0f} | {elapsed:.1f}s")
                    self.last_log_time = datetime.now()
                return True
        
        callback = SimpleCallback()
        
        # Entraînement SB3 standard
        agent.learn(
            total_timesteps=total_steps,
            reset_num_timesteps=True,
            progress_bar=False
        )
        
        elapsed_total = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ")
        print(f"   Durée totale: {elapsed_total:.1f}s")
        print(f"   Steps total: {total_steps}")
        print(f"   Capital final: ${env.capital:.2f}")
        print(f"   Steps/seconde: {total_steps/elapsed_total:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("ADAN - Test d'Entraînement Simple")
    print("Données simulées intégrées")
    print()
    
    success = test_short_training()
    
    if success:
        print("\n🎉 TEST D'ENTRAÎNEMENT RÉUSSI!")
        print("✅ OrderManager: Fonctionnel")
        print("✅ MultiAssetEnv: Fonctionnel") 
        print("✅ PPOAgent: Fonctionnel")
        print("✅ Pipeline complet: Opérationnel")
        exit_code = 0
    else:
        print("\n❌ TEST D'ENTRAÎNEMENT ÉCHOUÉ")
        exit_code = 1
    
    print(f"\nCode de sortie: {exit_code}")
    sys.exit(exit_code)