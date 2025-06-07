#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'évaluation des performances pour les modèles ADAN entraînés.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Assurer que le package src est dans le PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from src.adan_trading_bot.common.utils import load_config, get_logger
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_performance_metrics(env, episode_rewards, episode_capitals, trades_history):
    """Calcule les métriques de performance trading."""
    
    if not episode_capitals or len(episode_capitals) < 2:
        return {}
    
    initial_capital = episode_capitals[0]
    final_capital = episode_capitals[-1]
    
    # Rendement total
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Calcul des rendements quotidiens
    returns = np.diff(episode_capitals) / episode_capitals[:-1]
    
    # Sharpe Ratio (annualisé, assumant 365 trading days)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)
    else:
        sharpe_ratio = 0
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(episode_capitals)
    drawdown = (episode_capitals - peak) / peak * 100
    max_drawdown = np.min(drawdown)
    
    # Volatilité annualisée
    volatility = np.std(returns) * np.sqrt(365) * 100 if len(returns) > 1 else 0
    
    # Analyse des trades
    total_trades = len(trades_history) if trades_history else 0
    winning_trades = 0
    losing_trades = 0
    total_pnl = 0
    
    if trades_history:
        for trade in trades_history:
            pnl = trade.get('pnl', 0)
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Rendement moyen par épisode
    avg_episode_return = np.mean(episode_rewards) if episode_rewards else 0
    
    return {
        'total_return_percent': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_percent': max_drawdown,
        'volatility_percent': volatility,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_percent': win_rate,
        'avg_episode_return': avg_episode_return,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_pnl': total_pnl
    }

def load_test_data(config):
    """Charge les données de test."""
    try:
        test_file = Path('data/processed/merged/1m_test_merged.parquet')
        if test_file.exists():
            df = pd.read_parquet(test_file)
            logger.info(f"✅ Données de test chargées: {df.shape}")
            return df
        else:
            logger.error(f"❌ Fichier de test introuvable: {test_file}")
            return None
    except Exception as e:
        logger.error(f"❌ Erreur chargement données test: {e}")
        return None

def evaluate_model(model_path, config, num_episodes=10, max_steps_per_episode=1000):
    """Évalue un modèle entraîné sur les données de test."""
    
    logger.info(f"🔍 ÉVALUATION DU MODÈLE: {model_path}")
    logger.info("=" * 60)
    
    # Charger les données de test
    test_df = load_test_data(config)
    if test_df is None:
        return None
    
    # Créer l'environnement
    try:
        env = MultiAssetEnv(test_df, config, max_episode_steps_override=max_steps_per_episode)
        logger.info(f"✅ Environnement créé: {len(env.assets)} actifs")
    except Exception as e:
        logger.error(f"❌ Erreur création environnement: {e}")
        return None
    
    # Charger le modèle
    try:
        model = PPO.load(model_path)
        logger.info(f"✅ Modèle chargé: {model_path}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle: {e}")
        return None
    
    # Évaluation
    logger.info(f"🎯 Démarrage évaluation: {num_episodes} épisodes")
    
    episode_rewards = []
    episode_capitals = []
    trades_history = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        initial_capital = env.capital
        
        for step in range(max_steps_per_episode):
            # Prédiction avec le modèle (mode déterministe pour évaluation)
            action, _ = model.predict(obs, deterministic=True)
            
            # Exécution de l'action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Enregistrer les trades si disponibles
            if 'trade_info' in info:
                trades_history.append(info['trade_info'])
            
            if terminated or truncated:
                break
        
        final_capital = env.capital
        episode_rewards.append(episode_reward)
        episode_capitals.append(final_capital)
        
        capital_change = (final_capital - initial_capital) / initial_capital * 100
        
        logger.info(f"Episode {episode+1:2d}/{num_episodes}: Reward={episode_reward:7.2f}, "
                   f"Capital=${final_capital:8.0f} ({capital_change:+.1f}%), Steps={episode_steps}")
    
    # Calcul des métriques
    metrics = calculate_performance_metrics(env, episode_rewards, episode_capitals, trades_history)
    
    return metrics

def print_performance_report(metrics, model_path):
    """Affiche un rapport de performance formaté."""
    
    logger.info("=" * 60)
    logger.info("📊 RAPPORT DE PERFORMANCE")
    logger.info("=" * 60)
    logger.info(f"Modèle évalué: {model_path}")
    logger.info(f"Date d'évaluation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    logger.info("💰 PERFORMANCE FINANCIÈRE")
    logger.info("-" * 30)
    logger.info(f"Capital initial:        ${metrics['initial_capital']:,.0f}")
    logger.info(f"Capital final:          ${metrics['final_capital']:,.0f}")
    logger.info(f"Rendement total:        {metrics['total_return_percent']:+.2f}%")
    logger.info(f"PnL total:              ${metrics['total_pnl']:+,.2f}")
    logger.info("")
    
    logger.info("📈 MÉTRIQUES DE RISQUE")
    logger.info("-" * 30)
    logger.info(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
    logger.info(f"Maximum Drawdown:       {metrics['max_drawdown_percent']:.2f}%")
    logger.info(f"Volatilité annuelle:    {metrics['volatility_percent']:.2f}%")
    logger.info("")
    
    logger.info("🎯 ANALYSE DES TRADES")
    logger.info("-" * 30)
    logger.info(f"Total trades:           {metrics['total_trades']}")
    logger.info(f"Trades gagnants:        {metrics['winning_trades']}")
    logger.info(f"Trades perdants:        {metrics['losing_trades']}")
    logger.info(f"Taux de réussite:       {metrics['win_rate_percent']:.1f}%")
    logger.info("")
    
    logger.info("⭐ RENDEMENT PAR ÉPISODE")
    logger.info("-" * 30)
    logger.info(f"Reward moyen:           {metrics['avg_episode_return']:.3f}")
    logger.info("")
    
    # Classification de la performance
    if metrics['total_return_percent'] > 10:
        performance_grade = "🎉 EXCELLENT"
    elif metrics['total_return_percent'] > 5:
        performance_grade = "✅ BON"
    elif metrics['total_return_percent'] > 0:
        performance_grade = "🟡 MODÉRÉ"
    else:
        performance_grade = "❌ FAIBLE"
    
    logger.info(f"🏆 ÉVALUATION GLOBALE: {performance_grade}")
    logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate ADAN trading model performance")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.zip file)')
    parser.add_argument('--exec_profile', type=str, default='cpu', choices=['cpu', 'gpu'],
                       help='Execution profile for configuration')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Modèle introuvable: {args.model_path}")
        return 1
    
    # Charger la configuration (même pattern que train_rl_agent.py)
    try:
        main_config_path = 'config/main_config.yaml'
        data_config_path = f'config/data_config_{args.exec_profile}.yaml'
        
        logger.info(f"📂 Chargement configs: {main_config_path}, {data_config_path}")
        
        main_config = load_config(main_config_path)
        data_config = load_config(data_config_path)
        
        # Fusionner les configurations
        config = {
            **main_config,
            'data': data_config,
            'environment': main_config.get('environment', {})
        }
        
        logger.info(f"✅ Configuration chargée - Actifs: {data_config.get('assets', [])}")
    except Exception as e:
        logger.error(f"❌ Erreur chargement configuration: {e}")
        return 1
    
    # Évaluation
    metrics = evaluate_model(
        args.model_path, 
        config, 
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps
    )
    
    if metrics is None:
        logger.error("❌ Évaluation échouée")
        return 1
    
    # Affichage du rapport
    print_performance_report(metrics, args.model_path)
    
    # Sauvegarde optionnelle
    if args.save_results:
        results_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results = pd.DataFrame([metrics])
        df_results.to_csv(results_file, index=False)
        logger.info(f"💾 Résultats sauvegardés: {results_file}")
    
    # Code de sortie basé sur la performance
    if metrics['total_return_percent'] > 0:
        return 0  # Succès si rendement positif
    else:
        return 1  # Échec si rendement négatif

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)