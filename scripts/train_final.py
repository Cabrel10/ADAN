#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'entraînement final optimisé pour ADAN.
Version Production - Interface améliorée, logs épurés, monitoring en temps réel.
"""
import os
import sys
import argparse
import time
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.training.trainer import train_agent

def main():
    """
    Script d'entraînement final avec interface optimisée.
    """
    parser = argparse.ArgumentParser(description='ADAN Trading Agent - Entraînement Final')
    
    # Profil d'exécution unifié
    parser.add_argument(
        '--profile', 
        type=str, 
        default='cpu',
        choices=['cpu', 'gpu'],
        help="Profil d'exécution unifié ('cpu' ou 'gpu')"
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Device pour l'entraînement ('auto', 'cpu', 'cuda')"
    )
    
    # Paramètres d'entraînement
    parser.add_argument('--initial_capital', type=float, default=15000,
                        help='Capital initial (défaut: 15000)')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                        help='Nombre total de timesteps (défaut: 50000)')
    parser.add_argument('--max_episode_steps', type=int, default=2000,
                        help='Maximum steps par épisode (défaut: 2000)')
    
    # Paramètres d'agent
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (défaut: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (défaut: 64)')
    
    # Options de sauvegarde
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Fréquence de sauvegarde (défaut: 10000)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Nom du modèle (défaut: auto-généré)')
    
    # Mode verbose
    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbose avec logs détaillés')
    
    args = parser.parse_args()
    
    # Configuration des logs
    if args.verbose:
        log_level = 'DEBUG'
    else:
        log_level = 'INFO'
    
    # Configurer le logger
    logger = setup_logging('config/logging_config.yaml', level=log_level)
    
    # Affichage de démarrage
    print("🚀 ADAN Trading Agent - Entraînement Final")
    print("=" * 60)
    print(f"📊 Profil: {args.profile.upper()}")
    print(f"🖥️  Device: {args.device}")
    print(f"💰 Capital initial: ${args.initial_capital:,.2f}")
    print(f"🎯 Timesteps: {args.total_timesteps:,}")
    print(f"📈 Max episode steps: {args.max_episode_steps:,}")
    print(f"🧠 Learning rate: {args.learning_rate}")
    print(f"📦 Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Construire les chemins de configuration
    config_paths = {
        'main': 'config/main_config.yaml',
        'data': f'config/data_config_{args.profile}.yaml',
        'environment': 'config/environment_config.yaml',
        'agent': f'config/agent_config_{args.profile}.yaml'
    }
    
    # Vérifier l'existence des fichiers de configuration
    for key, path in config_paths.items():
        if not os.path.exists(path):
            logger.error(f"❌ Fichier de configuration manquant: {path}")
            sys.exit(1)
        logger.info(f"✅ {key.capitalize()} config: {path}")
    
    # Paramètres d'override
    override_params = {
        'initial_capital': args.initial_capital,
        'total_timesteps': args.total_timesteps,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_episode_steps': args.max_episode_steps,
        'device': args.device,
        'save_freq': args.save_freq
    }
    
    # Générer nom de modèle si non spécifié
    if args.model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_name = f"adan_final_{args.profile}_{timestamp}"
    
    override_params['model_name'] = args.model_name
    
    logger.info(f"🎯 Modèle: {args.model_name}")
    logger.info(f"💾 Sauvegarde tous les {args.save_freq:,} steps")
    
    try:
        # Lancer l'entraînement
        logger.info("🚀 Démarrage de l'entraînement...")
        start_time = time.time()
        
        result = train_agent(
            config_paths=config_paths,
            override_params=override_params
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Affichage des résultats
        print("\n" + "=" * 60)
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"⏱️  Durée: {duration/3600:.2f} heures ({duration/60:.1f} minutes)")
        print(f"🎯 Timesteps: {args.total_timesteps:,}")
        print(f"💾 Modèle sauvegardé: models/{args.model_name}.zip")
        
        if result and 'best_mean_reward' in result:
            print(f"🏆 Meilleure récompense moyenne: {result['best_mean_reward']:.4f}")
        
        print("=" * 60)
        
        # Instructions de suivi
        print("\n📋 PROCHAINES ÉTAPES:")
        print(f"1. Évaluer le modèle:")
        print(f"   python scripts/evaluate_performance.py --model_path models/{args.model_name}.zip --exec_profile {args.profile}")
        print(f"2. Analyser les performances:")
        print(f"   python scripts/analyze_training.py --model_name {args.model_name}")
        print(f"3. Continuer l'entraînement:")
        print(f"   python scripts/train_final.py --profile {args.profile} --total_timesteps 100000")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Entraînement interrompu par l'utilisateur")
        print("\n🛑 Entraînement interrompu. Le modèle partiellement entraîné a été sauvegardé.")
        
    except Exception as e:
        logger.error(f"❌ Erreur pendant l'entraînement: {str(e)}")
        print(f"\n💥 ERREUR: {str(e)}")
        print("Consultez les logs pour plus de détails.")
        sys.exit(1)

if __name__ == "__main__":
    main()