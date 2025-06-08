#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de lancement complet automatisé pour ADAN v2.1.
Gère l'entraînement multi-timeframe, l'évaluation, le monitoring, paper trading et apprentissage continu.
Support Binance Testnet avec capital de 15$ + nouvelles fonctionnalités d'apprentissage en temps réel.
"""
import os
import sys
import subprocess
import argparse
import time
from datetime import datetime
from pathlib import Path

# Rich imports pour interface
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich not available, using basic interface")

console = Console() if RICH_AVAILABLE else None

def print_message(message, style=""):
    """Print message with Rich if available, otherwise basic print."""
    if RICH_AVAILABLE and console:
        console.print(message, style=style)
    else:
        print(message)

def print_panel(content, title="", style=""):
    """Print panel with Rich if available."""
    if RICH_AVAILABLE and console:
        console.print(Panel(content, title=title))
    else:
        print(f"\n=== {title} ===")
        print(content)
        print("=" * (len(title) + 8))

def run_command(cmd, description="", show_output=True):
    """Exécute une commande avec gestion d'erreurs."""
    if description:
        print_message(f"🔄 {description}...", "cyan")
    
    try:
        if show_output:
            result = subprocess.run(cmd, shell=True, check=True, text=True, 
                                  capture_output=False)
        else:
            result = subprocess.run(cmd, shell=True, check=True, text=True, 
                                  capture_output=True)
        return True, result.returncode if hasattr(result, 'returncode') else 0
    except subprocess.CalledProcessError as e:
        print_message(f"❌ Erreur lors de l'exécution: {e}", "red")
        return False, e.returncode

def check_environment():
    """Vérifie l'environnement et les dépendances."""
    print_message("🔍 Vérification de l'environnement...", "blue")
    
    # Vérifier conda
    success, _ = run_command("conda --version", show_output=False)
    if not success:
        print_message("❌ Conda non trouvé", "red")
        return False
    
    # Vérifier l'environnement trading_env
    success, _ = run_command("conda info --envs | grep trading_env", show_output=False)
    if not success:
        print_message("❌ Environnement trading_env non trouvé", "red")
        return False
    
    # Vérifier les fichiers de config
    config_files = [
        "config/data_config_cpu.yaml",
        "config/agent_config_cpu.yaml",
        "config/environment_config.yaml",
        "config/main_config.yaml"
    ]
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print_message(f"❌ Fichier de config manquant: {config_file}", "red")
            return False
    
    # Vérifier les données selon le timeframe
    timeframe = getattr(args, 'timeframe', '1m') if 'args' in locals() else '1m'
    data_files = [
        f"data/processed/merged/unified/{timeframe}_train_merged.parquet",
        f"data/processed/merged/unified/{timeframe}_val_merged.parquet",
        f"data/processed/merged/unified/{timeframe}_test_merged.parquet"
    ]
    
    missing_data = [f for f in data_files if not os.path.exists(f)]
    if missing_data:
        print_message(f"⚠️ Données manquantes: {missing_data}", "yellow")
        print_message("Les données seront générées automatiquement", "yellow")
    
    print_message("✅ Environnement vérifié", "green")
    return True

def prepare_data_if_needed(timeframe='1m'):
    """Prépare les données si nécessaire pour le timeframe spécifié."""
    data_file = f"data/processed/merged/unified/{timeframe}_train_merged.parquet"
    
    if not os.path.exists(data_file):
        print_message("📊 Préparation des données nécessaire...", "yellow")
        
        # Vérifier les données source
        source_files = [
            "data/new/ADAUSDT_features.parquet",
            "data/new/BTCUSDT_features.parquet",
            "data/new/ETHUSDT_features.parquet"
        ]
        
        missing_sources = [f for f in source_files if not os.path.exists(f)]
        if missing_sources:
            print_message(f"❌ Données source manquantes: {missing_sources}", "red")
            return False
        
        # Exécuter le traitement des données pour le timeframe
        cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/convert_real_data.py --exec_profile cpu"'
        success, _ = run_command(cmd, f"Conversion des données (tous timeframes configurés)") # Updated description
        
        if not success:
            return False
    
    print_message("✅ Données prêtes", "green")
    return True

def train_model(timesteps=50000, capital=15.0, profile="cpu", timeframe="1m", verbose=False):
    """Lance l'entraînement du modèle pour le timeframe spécifié."""
    print_message(f"🚀 Démarrage de l'entraînement {timeframe} ({timesteps:,} timesteps, capital ${capital:.2f})", "green")
    
    verbose_flag = "--verbose" if verbose else ""
    cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/train_rl_agent.py --exec_profile {profile} --training_timeframe {timeframe} --total_timesteps {timesteps} --initial_capital {capital} {verbose_flag}"'
    
    success, _ = run_command(cmd, "Entraînement du modèle")
    return success

def evaluate_model(model_path="models/final_model.zip", capital=15.0, profile="cpu", timeframe="1m"):
    """Évalue le modèle entraîné."""
    # Essayer le modèle spécifique au timeframe d'abord
    timeframe_model = f"models/final_model_{timeframe}.zip"
    if os.path.exists(timeframe_model):
        model_path = timeframe_model
    elif not os.path.exists(model_path):
        # Essayer le modèle interrompu
        model_path = "models/interrupted_model.zip"
        if not os.path.exists(model_path):
            print_message("❌ Aucun modèle trouvé pour l'évaluation", "red")
            return False
    
    print_message(f"📊 Évaluation du modèle avec capital ${capital:.2f}", "blue")
    
    cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/test_model_quick.py --model_path {model_path} --capital {capital} --exec_profile {profile} --training_timeframe {timeframe} --episodes 3"'
    
    success, _ = run_command(cmd, f"Évaluation des performances {timeframe}")
    return success

def test_exchange_connection():
    """Teste la connexion au Binance Testnet avec diagnostic complet."""
    print_message("🔌 Test de connexion au Binance Testnet...", "blue")
    
    # Vérifier les variables d'environnement
    api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret_key = os.environ.get("BINANCE_TESTNET_SECRET_KEY")
    
    if not api_key or not secret_key:
        print_message("❌ Variables d'environnement manquantes pour Binance Testnet", "red")
        print_message("Définissez BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_SECRET_KEY", "yellow")
        print_message("💡 Guide: https://testnet.binance.vision/", "cyan")
        return False
    
    # Test système complet avec exchange
    cmd = "python test_complete_system.py --exec_profile cpu"
    success, _ = run_command(cmd, "Test système complet", show_output=False)
    
    if success:
        print_message("✅ Système et connexion Binance Testnet validés", "green")
    else:
        print_message("❌ Échec des tests système ou connexion", "red")
        # Fallback: test basique
        cmd = "python test_exchange_connector.py"
        fallback_success, _ = run_command(cmd, "Test de base CCXT", show_output=False)
        if fallback_success:
            print_message("✅ Connexion basique OK (tests système échoués)", "yellow")
            return True
    
    return success

def run_paper_trading(model_path="models/final_model.zip", capital=15000, iterations=30, learning=False, profile="cpu", timeframe="1m"):
    """Lance le paper trading en temps réel avec options d'apprentissage continu."""
    mode = "avec apprentissage continu" if learning else "mode inférence"
    print_message(f"🌐 Démarrage du paper trading (${capital:.2f}, {iterations} itérations, {mode}, Profile: {profile}, Timeframe: {timeframe})", "green")

    # Determine model path (specific or generic)
    timeframe_model_path = f"models/final_model_{timeframe}.zip"
    if os.path.exists(timeframe_model_path):
        model_to_use = timeframe_model_path
        print_message(f"💡 Utilisation du modèle spécifique au timeframe: {model_to_use}", "yellow")
    elif os.path.exists(model_path):
        model_to_use = model_path
        print_message(f"💡 Utilisation du modèle générique: {model_to_use}", "yellow")
    else:
        # Try interrupted model as last resort for generic path
        interrupted_model_path = "models/interrupted_model.zip"
        if os.path.exists(interrupted_model_path):
            model_to_use = interrupted_model_path
            print_message(f"💡 Utilisation du modèle interrompu: {model_to_use}", "yellow")
        else:
            print_message(f"❌ Aucun modèle trouvé (ni {timeframe_model_path}, ni {model_path}, ni {interrupted_model_path})", "red")
            return False

    print_message(f"📁 Modèle sélectionné pour paper trading: {model_to_use}", "cyan")

    if learning:
        # Apprentissage continu conservateur
        cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/online_learning_agent.py --exec_profile {profile} --training_timeframe {timeframe} --model_path {model_to_use} --initial_capital {capital} --learning_rate 0.00001 --exploration_rate 0.1 --max_iterations {iterations}"'
    else:
        # Paper trading classique (inférence)
        cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/paper_trade_agent.py --exec_profile {profile} --training_timeframe {timeframe} --model_path {model_to_use} --initial_capital {capital} --max_iterations {iterations} --sleep_seconds 60"'
    
    success, _ = run_command(cmd, f"Paper trading {mode}")
    return success

def run_human_feedback_trading(model_path="models/final_model.zip", capital=15000, iterations=20, profile="cpu", timeframe="1m"):
    """Lance le trading avec feedback humain interactif."""
    print_message(f"🤝 Démarrage du trading avec feedback humain (${capital:.2f}, {iterations} décisions, Profile: {profile}, Timeframe: {timeframe})", "magenta")

    # Determine model path (specific or generic)
    timeframe_model_path = f"models/final_model_{timeframe}.zip"
    if os.path.exists(timeframe_model_path):
        model_to_use = timeframe_model_path
        print_message(f"💡 Utilisation du modèle spécifique au timeframe: {model_to_use}", "yellow")
    elif os.path.exists(model_path):
        model_to_use = model_path
        print_message(f"💡 Utilisation du modèle générique: {model_to_use}", "yellow")
    else:
        interrupted_model_path = "models/interrupted_model.zip"
        if os.path.exists(interrupted_model_path):
            model_to_use = interrupted_model_path
            print_message(f"💡 Utilisation du modèle interrompu: {model_to_use}", "yellow")
        else:
            print_message(f"❌ Aucun modèle trouvé (ni {timeframe_model_path}, ni {model_path}, ni {interrupted_model_path})", "red")
            return False

    print_message(f"📁 Modèle sélectionné pour feedback trading: {model_to_use}", "cyan")

    cmd = f'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/human_feedback_trading.py --exec_profile {profile} --training_timeframe {timeframe} --model_path {model_to_use} --initial_capital {capital} --interactive_mode true --max_iterations {iterations}"'
    
    success, _ = run_command(cmd, "Trading avec feedback humain")
    return success

def run_data_pipeline():
    """Lance le pipeline de données multi-timeframe."""
    print_message("📊 Démarrage du pipeline de données multi-timeframe...", "blue")
    
    cmd = 'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/convert_real_data.py --exec_profile cpu"'
    
    success, _ = run_command(cmd, "Pipeline de données unifié")
    
    if success:
        print_message("✅ Pipeline de données terminé avec succès", "green")
        # Fusion des données
        cmd_merge = 'bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate trading_env && python scripts/merge_processed_data.py --exec_profile cpu --timeframes 1m 1h 1d --splits train val test --training-timeframe 1m"'
        success_merge, _ = run_command(cmd_merge, "Fusion des données")
        return success_merge
    else:
        print_message("❌ Échec du pipeline de données", "red")
        return False

def main():
    parser = argparse.ArgumentParser(description='ADAN - Lancement Automatisé Complet')
    
    # Modes d'exécution
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'train', 'eval', 'check', 'paper_trading', 'exchange_test', 'online_learning', 'human_feedback', 'data_pipeline'],
                        help='Mode d\'exécution')
    
    # Paramètres d'entraînement
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Nombre de timesteps (défaut: 50000)')
    parser.add_argument('--capital', type=float, default=15.0,
                        help='Capital initial (défaut: 15$)')
    parser.add_argument('--profile', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='Profil d\'exécution')
    parser.add_argument('--timeframe', type=str, default='1m',
                        choices=['1m', '1h', '1d'],
                        help='Timeframe d\'entraînement (défaut: 1m)')
    
    # Options
    parser.add_argument('--quick', action='store_true',
                        help='Mode rapide (5000 timesteps)')
    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbose')
    parser.add_argument('--interactive', action='store_true',
                        help='Mode interactif avec confirmations')
    parser.add_argument('--testnet', action='store_true',
                        help='Utiliser Binance Testnet pour paper trading')
    parser.add_argument('--duration', type=int, default=30,
                        help='Durée du paper trading en minutes (défaut: 30)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Nombre d\'itérations pour paper trading/apprentissage (défaut: 50)')
    parser.add_argument('--learning', action='store_true',
                        help='Activer l\'apprentissage continu en paper trading')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Taux d\'apprentissage pour mode continu (défaut: 0.00001)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.timesteps = 5000
    
    # Affichage de démarrage
    start_time = datetime.now()
    
    print_panel(
        f"🚀 ADAN Trading Agent - Lancement Automatisé\n"
        f"Mode: {args.mode.upper()}\n"
        f"Capital: ${args.capital:.2f}\n"
        + (f"Timesteps: {args.timesteps:,}\n" if args.mode not in ['paper_trading', 'online_learning', 'human_feedback'] else f"Itérations: {args.iterations}\n")
        + (f"Timeframe: {args.timeframe.upper()}\n" if args.mode not in ['paper_trading', 'online_learning', 'human_feedback'] else f"Exchange: {'Testnet' if args.testnet else 'Live'}\n")
        + f"Profil: {args.profile.upper()}\n"
        + f"Démarré: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Initialisation ADAN"
    )
    
    try:
        # Étape 1: Vérification de l'environnement
        if not check_environment():
            print_message("❌ Échec de la vérification de l'environnement", "red")
            return 1
        
        if args.mode == 'check':
            print_message("✅ Vérification terminée avec succès", "green")
            return 0
        
        # Test spécifique de l'exchange
        if args.mode == 'exchange_test':
            success = test_exchange_connection()
            return 0 if success else 1
        
        # Pipeline de données
        if args.mode == 'data_pipeline':
            success = run_data_pipeline()
            return 0 if success else 1
        
        # Étape 2: Préparation des données
        if args.mode not in ['paper_trading']:
            if not prepare_data_if_needed(args.timeframe):
                print_message("❌ Échec de la préparation des données", "red")
                return 1
        
        # Étape 3: Entraînement (si nécessaire)
        if args.mode in ['auto', 'train']:
            if args.interactive and RICH_AVAILABLE:
                should_train = Confirm.ask(f"Démarrer l'entraînement {args.timeframe} ({args.timesteps:,} timesteps, ${args.capital:.2f})?")
                if not should_train:
                    print_message("Entraînement annulé par l'utilisateur", "yellow")
                    return 0
            
            train_success = train_model(args.timesteps, args.capital, args.profile, args.timeframe, args.verbose)
            
            if not train_success:
                print_message("❌ Échec de l'entraînement", "red")
                if args.mode == 'train':
                    return 1
                # En mode auto, continuer avec évaluation si modèle existe
        
        # Étape 4: Évaluation
        if args.mode in ['auto', 'eval']:
            if args.interactive and RICH_AVAILABLE:
                should_eval = Confirm.ask("Procéder à l'évaluation du modèle?")
                if not should_eval:
                    print_message("Évaluation annulée par l'utilisateur", "yellow")
                    return 0
            
            eval_success = evaluate_model(capital=args.capital, profile=args.profile, timeframe=args.timeframe)
            
            if not eval_success:
                print_message("❌ Échec de l'évaluation", "red")
                return 1
        
        # Étape 5: Paper Trading et modes avancés
        if args.mode == 'paper_trading':
            # Vérifier la connexion exchange d'abord
            if not test_exchange_connection():
                print_message("❌ Connexion exchange requise pour paper trading", "red")
                return 1
            
            if args.interactive and RICH_AVAILABLE:
                mode_desc = "avec apprentissage" if args.learning else "inférence"
                should_paper_trade = Confirm.ask(f"Démarrer le paper trading ({mode_desc}, ${args.capital:.2f}, {args.iterations} itérations)?")
                if not should_paper_trade:
                    print_message("Paper trading annulé par l'utilisateur", "yellow")
                    return 0
            
            paper_success = run_paper_trading(capital=args.capital, iterations=args.iterations, learning=args.learning, profile=args.profile, timeframe=args.timeframe)
            
            if not paper_success:
                print_message("❌ Échec du paper trading", "red")
                return 1
        
        # Apprentissage continu
        if args.mode == 'online_learning':
            if not test_exchange_connection():
                print_message("❌ Connexion exchange requise pour apprentissage continu", "red")
                return 1
            
            if args.interactive and RICH_AVAILABLE:
                should_learn = Confirm.ask(f"Démarrer l'apprentissage continu (${args.capital:.2f}, {args.iterations} itérations, lr={args.learning_rate})?")
                if not should_learn:
                    print_message("Apprentissage continu annulé par l'utilisateur", "yellow")
                    return 0
            
            learning_success = run_paper_trading(capital=args.capital, iterations=args.iterations, learning=True)
            
            if not learning_success:
                print_message("❌ Échec de l'apprentissage continu", "red")
                return 1
        
        # Feedback humain
        if args.mode == 'human_feedback':
            if not test_exchange_connection():
                print_message("❌ Connexion exchange requise pour feedback humain", "red")
                return 1
            
            if args.interactive and RICH_AVAILABLE:
                should_feedback = Confirm.ask(f"Démarrer le trading avec feedback humain (${args.capital:.2f}, {args.iterations} décisions)?")
                if not should_feedback:
                    print_message("Feedback humain annulé par l'utilisateur", "yellow")
                    return 0
            
            feedback_success = run_human_feedback_trading(capital=args.capital, iterations=args.iterations, profile=args.profile, timeframe=args.timeframe)
            
            if not feedback_success:
                print_message("❌ Échec du trading avec feedback humain", "red")
                return 1
        
        # Résumé final
        end_time = datetime.now()
        duration = end_time - start_time
        
        summary_table = Table(title="🎉 Exécution ADAN v2.1 Terminée")
        summary_table.add_column("Métrique", style="dim cyan")
        summary_table.add_column("Valeur", style="bright_green")
        
        summary_table.add_row("Mode", args.mode.upper())
        summary_table.add_row("Durée totale", str(duration).split('.')[0])
        summary_table.add_row("Capital", f"${args.capital:.2f}")
        
        if args.mode in ['paper_trading', 'online_learning', 'human_feedback']:
            summary_table.add_row("Itérations", str(args.iterations))
            if args.learning or args.mode == 'online_learning':
                summary_table.add_row("Apprentissage", "Activé")
                summary_table.add_row("Learning Rate", str(args.learning_rate))
        else:
            summary_table.add_row("Timesteps", f"{args.timesteps:,}")
            summary_table.add_row("Timeframe", args.timeframe.upper())
        
        summary_table.add_row("Status", "✅ SUCCÈS")
        
        if RICH_AVAILABLE and console:
            console.print(summary_table)
        else:
            print(f"\n🎉 ADAN v2.1 - Exécution terminée avec succès")
            print(f"Mode: {args.mode.upper()}")
            print(f"Durée: {duration}")
            print(f"Capital: ${args.capital:.2f}")
        
        return 0
    
    except KeyboardInterrupt:
        print_message("\n🛑 Exécution interrompue par l'utilisateur", "yellow")
        return 1
    except Exception as e:
        print_message(f"\n❌ Erreur inattendue: {str(e)}", "red")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)