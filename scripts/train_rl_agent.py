#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'entraînement optimisé pour l'agent ADAN.
Version Production avec barres de progression et gestion dynamique des flux.
"""
import os
import sys
import argparse
import time
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.training.trainer import train_agent

console = Console()

def main():
    """
    Fonction principale pour l'entraînement de l'agent ADAN avec interface améliorée.
    """
    parser = argparse.ArgumentParser(description='ADAN Trading Agent - Entraînement Optimisé')
    
    # Profil d'exécution et device
    parser.add_argument(
        '--exec_profile', 
        type=str, 
        default='cpu',
        choices=['cpu', 'gpu', 'smoke_cpu'], # Added 'smoke_cpu'
        help="Profil d'exécution unifié ('cpu' ou 'gpu') pour charger data_config_{profile}.yaml."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Appareil à utiliser pour l'entraînement ('auto', 'cpu', 'cuda')."
    )
    
    # Chemins des fichiers de configuration
    parser.add_argument('--main_config', type=str, default=None,
                        help='Chemin vers le fichier de configuration principal')
    parser.add_argument('--data_config', type=str, default=None,
                        help='Chemin vers le fichier de configuration des données')
    parser.add_argument('--env_config', type=str, default='config/environment_config.yaml',
                        help='Chemin vers le fichier de configuration de l\'environnement')
    parser.add_argument('--agent_config', type=str, default=None,
                        help='Chemin vers le fichier de configuration de l\'agent')
    parser.add_argument('--logging_config', type=str, default='config/logging_config.yaml',
                        help='Chemin vers le fichier de configuration des logs')
    
    # Paramètres d'entraînement optimisés
    parser.add_argument('--initial_capital', type=float, default=15.0,
                        help='Capital initial pour l\'entraînement (défaut: 15$)')
    parser.add_argument('--training_data_file', type=str, default=None,
                        help='Chemin vers le fichier de données d\'entraînement')
    parser.add_argument('--validation_data_file', type=str, default=None,
                        help='Chemin vers le fichier de données de validation')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                        help='Nombre total de timesteps (défaut: 50000)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Taux d\'apprentissage (défaut: 3e-4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Taille du batch (défaut: 64)')
    parser.add_argument('--max_episode_steps', type=int, default=2000,
                        help='Maximum steps par épisode (défaut: 2000)')
    
    # Paramètres d'affichage et monitoring
    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbose avec logs détaillés')
    parser.add_argument('--progress_bar', action='store_true', default=True,
                        help='Afficher la barre de progression (défaut: activé)')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Fréquence de sauvegarde du modèle (défaut: 10000)')
    parser.add_argument('--quiet_positive', action='store_true',
                        help='Réduire les logs pour les retours positifs')
    
    # Timeframe d'entraînement
    parser.add_argument(
        '--training_timeframe', 
        type=str, 
        default=None, 
        choices=['1m', '1h', '1d'],
        help="Timeframe d'entraînement (surcharge la valeur de data_config si spécifié)."
    )
    parser.add_argument('--n_steps', type=int, default=None,
                        help='PPO n_steps (rollout buffer size). Overrides agent_config.')
    parser.add_argument('--model_name_suffix', type=str, default=None,
                        help='Suffix to add to saved model filenames (e.g., final_model_{timeframe}_{suffix}.zip).')
    
    args = parser.parse_args()
    
    # Configuration du niveau de logs
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Configurer le logger
    logger = setup_logging(args.logging_config, default_level=log_level)
    
    # Affichage de démarrage avec Rich
    start_time = datetime.now()
    console.print(Panel.fit(
        "[bold blue]🚀 ADAN Trading Agent - Entraînement Optimisé[/bold blue]\n"
        f"[cyan]Démarré le: {start_time.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]\n"
        f"[yellow]Capital initial: ${args.initial_capital:.2f}[/yellow]\n"
        f"[green]Timesteps: {args.total_timesteps:,}[/green]",
        title="Initialisation ADAN"
    ))
    
    logger.info("Starting ADAN optimized training script")
    
    # Construire les chemins de configuration en fonction du profil d'exécution
    profile = args.exec_profile
    logger.info(f"Using execution profile: {profile}")
    
    # Définir les chemins par défaut en fonction du profil unifié
    main_config = args.main_config if args.main_config else 'config/main_config.yaml'
    data_config = args.data_config if args.data_config else f'config/data_config_{profile}.yaml'
    agent_config = args.agent_config if args.agent_config else f'config/agent_config_{profile}.yaml'
    
    config_paths = {
        'main': main_config,
        'data': data_config,
        'environment': args.env_config,
        'agent': agent_config
    }
    
    # Vérifier l'existence des fichiers de configuration
    missing_configs = []
    for key, path in config_paths.items():
        if not os.path.exists(path):
            missing_configs.append(path)
        else:
            logger.info(f"✅ {key.capitalize()} config: {path}")
    
    if missing_configs:
        console.print(Panel(
            f"[bold red]❌ Fichiers de configuration manquants:[/bold red]\n" +
            "\n".join(f"• {path}" for path in missing_configs),
            title="Erreur Configuration"
        ))
        return 1
    
    # Traitement du paramètre training_timeframe
    if args.training_timeframe:
        logger.info(f"Surcharge du training_timeframe par argument de ligne de commande: {args.training_timeframe}")
        # Charger le contenu de data_config pour le modifier
        data_config_content = load_config(config_paths['data'])
        data_config_content['training_timeframe'] = args.training_timeframe
        logger.info(f"Training timeframe défini sur: {args.training_timeframe}")
    
    # Afficher un tableau des paramètres d'entraînement
    params_table = Table(title="[bold magenta]Paramètres d'Entraînement[/bold magenta]")
    params_table.add_column("Paramètre", style="dim cyan")
    params_table.add_column("Valeur", style="bright_white")
    
    params_table.add_row("💰 Capital Initial", f"${args.initial_capital:.2f}")
    params_table.add_row("🎯 Total Timesteps", f"{args.total_timesteps:,}")
    params_table.add_row("📊 Profil d'Exécution", f"{profile.upper()}")
    params_table.add_row("🖥️ Device", f"{args.device}")
    params_table.add_row("🧠 Learning Rate", f"{args.learning_rate}")
    params_table.add_row("📦 Batch Size", f"{args.batch_size}")
    params_table.add_row("📈 Max Episode Steps", f"{args.max_episode_steps:,}")
    params_table.add_row("💾 Save Frequency", f"{args.save_freq:,}")
    
    console.print(params_table)
    
    # Extraire les paramètres d'entraînement qui peuvent outrepasser les configurations YAML
    override_params = {
        'initial_capital': args.initial_capital,
        'total_timesteps': args.total_timesteps,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_episode_steps': args.max_episode_steps,
        'device': args.device,
        'save_freq': args.save_freq,
        'quiet_positive': args.quiet_positive
    }
    
    # Ajouter le training_timeframe si spécifié
    if args.training_timeframe:
        override_params['training_timeframe'] = args.training_timeframe
    
    # Ajouter n_steps si spécifié
    if args.n_steps is not None:
        override_params['n_steps'] = args.n_steps
        logger.info(f"CLI override: PPO n_steps set to {args.n_steps}")

    # Ajouter model_name_suffix si spécifié
    if args.model_name_suffix is not None:
        override_params['model_name_suffix'] = args.model_name_suffix
        logger.info(f"CLI override: Model name suffix set to '{args.model_name_suffix}'")

    # Ajouter les chemins de fichiers si spécifiés
    if args.training_data_file is not None:
        override_params['training_data_file'] = args.training_data_file
    if args.validation_data_file is not None:
        override_params['validation_data_file'] = args.validation_data_file
    
    logger.info(f"Using device: {args.device}")
    
    # Lancer l'entraînement avec monitoring des performances
    training_start = time.time()
    
    try:
        console.print("\n[bold green]🎯 Démarrage de l'entraînement avec flux dynamiques...[/bold green]")
        logger.info("Starting agent training with dynamic flow management...")
        
        # Créer une barre de progression si activée
        if args.progress_bar:
            with Progress(
                TextColumn("[bold blue]ADAN Training"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                # Ajouter la tâche de progression
                task = progress.add_task("[cyan]Entraînement en cours...", total=args.total_timesteps)
                
                # Passer la barre de progression au trainer
                override_params['progress_bar_obj'] = (progress, task)
                
                agent, env = train_agent(config_paths, override_params=override_params)
        else:
            agent, env = train_agent(config_paths, override_params=override_params)
        
        training_duration = time.time() - training_start
        
        # Affichage des résultats finaux
        success_table = Table(title="[bold green]✅ Entraînement Terminé avec Succès![/bold green]")
        success_table.add_column("Métrique", style="dim cyan")
        success_table.add_column("Valeur", style="bright_white")
        
        success_table.add_row("⏱️ Durée Totale", f"{training_duration/3600:.2f}h ({training_duration/60:.1f}min)")
        success_table.add_row("🎯 Timesteps Effectués", f"{args.total_timesteps:,}")
        success_table.add_row("💰 Capital Initial", f"${args.initial_capital:.2f}")
        success_table.add_row("📊 Profil Utilisé", f"{profile.upper()}")
        success_table.add_row("💾 Modèle Final", "models/final_model.zip")
        success_table.add_row("🔄 Modèle Interrompu", "models/interrupted_model.zip")
        
        console.print(success_table)
        
        # Instructions de suivi avec flux dynamiques
        console.print(Panel(
            "[bold cyan]📋 Prochaines étapes avec flux optimisés:[/bold cyan]\n\n"
            f"[yellow]1. Évaluer les performances avec gestion dynamique:[/yellow]\n"
            f"   python scripts/evaluate_performance.py --model_path models/final_model.zip --exec_profile {profile}\n\n"
            f"[yellow]2. Analyser les flux monétaires:[/yellow]\n"
            f"   python scripts/analyze_cash_flow.py --model models/final_model.zip --capital {args.initial_capital}\n\n"
            f"[yellow]3. Continuer l'entraînement avec capital adaptatif:[/yellow]\n"
            f"   python scripts/train_rl_agent.py --exec_profile {profile} --total_timesteps 100000 --initial_capital 15.0\n\n"
            f"[yellow]4. Test en temps réel avec micro-capital:[/yellow]\n"
            f"   python scripts/live_trading.py --model models/final_model.zip --capital 15.0",
            title="Gestion Dynamique des Flux"
        ))
        
        logger.info("Training completed successfully with dynamic flow management!")
        
    except KeyboardInterrupt:
        training_duration = time.time() - training_start
        console.print(Panel(
            f"[bold yellow]⏹️ Entraînement interrompu après {training_duration/60:.1f} minutes[/bold yellow]\n\n"
            f"[cyan]Modèle partiellement entraîné sauvegardé:[/cyan]\n"
            f"models/interrupted_model.zip\n\n"
            f"[green]Capital géré dynamiquement: ${args.initial_capital:.2f}[/green]\n"
            f"Flux monétaires préservés dans l'état final.",
            title="Interruption Contrôlée"
        ))
        logger.info("Training interrupted by user - dynamic flows preserved")
        return 1
        
    except Exception as e:
        training_duration = time.time() - training_start
        console.print(Panel(
            f"[bold red]❌ Erreur pendant l'entraînement:[/bold red]\n"
            f"{str(e)}\n\n"
            f"[yellow]Durée avant erreur: {training_duration/60:.1f} minutes[/yellow]\n"
            f"[cyan]État des flux préservé dans les logs[/cyan]\n"
            f"Consultez les logs pour diagnostic détaillé.",
            title="Erreur avec Préservation des Flux"
        ))
        logger.error(f"Error during training with dynamic flow management: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())