#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de test pour l'entraînement ADAN avec affichage amélioré.
Test avec un nombre réduit de timesteps pour valider l'interface.
"""
import os
import sys
import argparse
import signal
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from src.adan_trading_bot.agent.ppo_agent import create_ppo_agent
from src.adan_trading_bot.training.callbacks import CustomTrainingInfoCallback, EvaluationCallback
import pandas as pd

console = Console()

class GracefulTrainingController:
    """Contrôleur pour arrêter l'entraînement proprement avec Ctrl+C"""
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        console.print("\n[bold yellow]⚠️  Arrêt demandé - Sauvegarde en cours...[/bold yellow]")
        self.should_stop = True

def main():
    parser = argparse.ArgumentParser(description='Test d\'entraînement ADAN avec affichage amélioré')
    parser.add_argument('--exec_profile', type=str, default='cpu',
                        help='Profil d\'exécution (cpu, gpu)')
    parser.add_argument('--total_timesteps', type=int, default=5000,
                        help='Nombre total de timesteps pour le test')
    parser.add_argument('--initial_capital', type=float, default=15000,
                        help='Capital initial pour l\'entraînement')
    parser.add_argument('--max_episode_steps', type=int, default=500,
                        help='Nombre maximum de steps par épisode')
    parser.add_argument('--verbose', action='store_true',
                        help='Affichage détaillé')
    
    args = parser.parse_args()
    
    # Initialiser le contrôleur d'arrêt gracieux
    controller = GracefulTrainingController()
    
    # Configuration du logger avec niveau INFO pour un affichage propre
    logger = setup_logging('config/logging_config.yaml')
    
    # Affichage d'introduction
    intro_text = Text()
    intro_text.append("🚀 Test d'Entraînement ADAN\n", style="bold cyan")
    intro_text.append(f"Profile: {args.exec_profile} | ", style="dim")
    intro_text.append(f"Timesteps: {args.total_timesteps:,} | ", style="dim")
    intro_text.append(f"Capital: ${args.initial_capital:,}", style="dim")
    
    console.print(Panel(intro_text, title="[bold green]Initialisation[/bold green]", expand=False))
    
    try:
        # Construire les chemins de configuration
        profile = args.exec_profile
        device_type = 'cpu' if profile.startswith('cpu') else 'gpu'
        
        config_paths = {
            'main': 'config/main_config.yaml',
            'data': f'config/data_config_{profile}.yaml',
            'environment': 'config/environment_config.yaml',
            'agent': f'config/agent_config_{device_type}.yaml'
        }
        
        console.print(f"📁 Chargement des configurations...")
        
        # Charger les configurations
        configs = {}
        for key, path in config_paths.items():
            try:
                configs[key] = load_config(path)
                console.print(f"   ✅ {key}: {path}")
            except Exception as e:
                console.print(f"   ❌ {key}: {path} - {str(e)}")
                return 1
        
        # Fusionner les configurations
        config = {
            'data': configs['data'],
            'environment': configs['environment'],
            'agent': configs['agent']
        }
        
        # Appliquer les overrides
        config['environment']['initial_capital'] = args.initial_capital
        config['agent']['total_timesteps'] = args.total_timesteps
        
        if args.max_episode_steps:
            config['environment']['max_steps'] = args.max_episode_steps
        
        console.print(f"📊 Chargement des données...")
        
        # Charger directement les données fusionnées existantes
        train_file = "data/processed/merged/unified/1m_train_merged.parquet"
        val_file = "data/processed/merged/unified/1m_val_merged.parquet"
        test_file = "data/processed/merged/unified/1m_test_merged.parquet"
        
        try:
            console.print(f"   📄 Chargement {train_file}...")
            train_df = pd.read_parquet(train_file)
            console.print(f"   ✅ Train: {train_df.shape[0]:,} échantillons, {train_df.shape[1]:,} features")
            
            val_df = None
            if os.path.exists(val_file):
                console.print(f"   📄 Chargement {val_file}...")
                val_df = pd.read_parquet(val_file)
                console.print(f"   ✅ Validation: {val_df.shape[0]:,} échantillons")
            
            test_df = None
            if os.path.exists(test_file):
                console.print(f"   📄 Chargement {test_file}...")
                test_df = pd.read_parquet(test_file)
                console.print(f"   ✅ Test: {test_df.shape[0]:,} échantillons")
                
        except Exception as e:
            console.print(f"[bold red]❌ Erreur lors du chargement des données: {str(e)}[/bold red]")
            console.print("[dim]Assurez-vous que les données ont été converties avec convert_real_data.py[/dim]")
            return 1
        
        if train_df is None or train_df.empty:
            console.print("[bold red]❌ Erreur: Aucune donnée d'entraînement trouvée[/bold red]")
            return 1
        
        # Créer l'environnement d'entraînement
        console.print(f"🏗️  Création de l'environnement...")
        train_env = MultiAssetEnv(train_df, config, max_episode_steps_override=args.max_episode_steps)
        
        # Créer l'environnement de validation (optionnel)
        val_env = None
        if val_df is not None and not val_df.empty:
            console.print(f"🏗️  Création de l'environnement de validation...")
            val_env = MultiAssetEnv(val_df, config, max_episode_steps_override=args.max_episode_steps)
        
        # Créer l'agent
        console.print(f"🤖 Création de l'agent PPO...")
        agent = create_ppo_agent(train_env, config)
        
        # Créer les callbacks avec affichage amélioré
        callbacks = []
        
        # Callback d'information d'entraînement (fréquence réduite pour le test)
        training_callback = CustomTrainingInfoCallback(check_freq=1, verbose=1)
        callbacks.append(training_callback)
        
        # Callback d'évaluation (si environnement de validation disponible)
        if val_env is not None:
            eval_callback = EvaluationCallback(
                eval_env=val_env,
                eval_freq=max(1000, args.total_timesteps // 5),  # Évaluer 5 fois pendant l'entraînement
                n_eval_episodes=3,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # Affichage de démarrage
        start_panel = Panel(
            f"[bold green]🎯 Démarrage de l'entraînement[/bold green]\n"
            f"Timesteps: {args.total_timesteps:,}\n"
            f"Capital initial: ${args.initial_capital:,}\n"
            f"Max steps/épisode: {args.max_episode_steps}\n"
            f"Actifs: {', '.join(config['data']['assets'])}\n"
            f"[dim]Ctrl+C pour arrêter proprement[/dim]",
            title="[bold cyan]Lancement[/bold cyan]",
            expand=False
        )
        console.print(start_panel)
        
        # Lancer l'entraînement
        start_time = time.time()
        
        try:
            agent.learn(
                total_timesteps=args.total_timesteps,
                callback=callbacks,
                progress_bar=False  # Nous utilisons notre propre barre de progression
            )
        except KeyboardInterrupt:
            console.print("\n[bold yellow]⚠️  Entraînement interrompu par l'utilisateur[/bold yellow]")
        except Exception as e:
            console.print(f"\n[bold red]❌ Erreur pendant l'entraînement: {str(e)}[/bold red]")
            return 1
        
        # Sauvegarder le modèle
        model_path = f"models/test_model_{int(time.time())}.zip"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save(model_path)
        
        console.print(f"\n[bold green]✅ Modèle sauvegardé: {model_path}[/bold green]")
        
        # Résumé final
        duration = time.time() - start_time
        final_panel = Panel(
            f"[bold green]🎉 Test d'entraînement terminé avec succès![/bold green]\n"
            f"Durée: {duration:.1f}s\n"
            f"Modèle: {model_path}\n"
            f"[dim]Le système d'affichage fonctionne correctement[/dim]",
            title="[bold cyan]Succès[/bold cyan]",
            expand=False
        )
        console.print(final_panel)
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]❌ Erreur fatale: {str(e)}[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())