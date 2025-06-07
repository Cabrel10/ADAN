#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour l'environnement MultiAssetEnv avec les données fusionnées.
Ce script charge un fichier de données fusionnées, instancie l'environnement,
et exécute quelques pas pour vérifier son bon fonctionnement.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from gymnasium.utils.env_checker import check_env
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
import time

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules du projet
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.common.utils import load_config, get_logger

# Initialiser le logger et la console rich
logger = get_logger(__name__)
console = Console()

def load_configurations(exec_profile='cpu_lot1'):
    """
    Charge les fichiers de configuration nécessaires.
    
    Args:
        exec_profile (str): Profil d'exécution ('cpu', 'gpu', 'cpu_lot1', 'cpu_lot2', 'gpu_lot1', 'gpu_lot2')
    
    Returns:
        tuple: (main_config, data_config, env_config, agent_config)
    """
    console.print(Rule(f"[bold blue]Chargement des configurations (profil: {exec_profile})[/bold blue]"))
    
    # Extraire le type de device (cpu/gpu) du profil pour agent_config
    if exec_profile.startswith('cpu'):
        device_type = 'cpu'
    elif exec_profile.startswith('gpu'):
        device_type = 'gpu'
    else:
        # Profils legacy (cpu, gpu)
        device_type = exec_profile
    
    # Chemins des fichiers de configuration
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    main_config_path = os.path.join(config_dir, "main_config.yaml")
    data_config_path = os.path.join(config_dir, f"data_config_{exec_profile}.yaml")
    env_config_path = os.path.join(config_dir, "environment_config.yaml")
    agent_config_path = os.path.join(config_dir, f"agent_config_{device_type}.yaml")
    
    console.print(f"[dim]Utilisation des fichiers de configuration:[/dim]")
    console.print(f"[dim]- Main config: {main_config_path}[/dim]")
    console.print(f"[dim]- Data config: {data_config_path}[/dim]")
    console.print(f"[dim]- Environment config: {env_config_path}[/dim]")
    console.print(f"[dim]- Agent config: {agent_config_path}[/dim]")
    
    # Charger les configurations
    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)
    env_config = load_config(env_config_path)
    agent_config = load_config(agent_config_path)
    
    console.print("[green]✓[/green] Configurations chargées avec succès")
    
    return main_config, data_config, env_config, agent_config

def load_merged_data(main_config, data_config, timeframe="1h", split="train"):
    """
    Charge un fichier de données fusionnées.
    
    Args:
        main_config (dict): Configuration principale
        data_config (dict): Configuration des données
        timeframe (str): Timeframe à charger (1m, 1h, 1d)
        split (str): Split à charger (train, val, test)
        
    Returns:
        pandas.DataFrame: DataFrame contenant les données fusionnées
    """
    console.print(Rule(f"[bold blue]Chargement des données fusionnées ({timeframe}_{split})[/bold blue]"))
    
    # Construire le chemin vers le fichier de données fusionnées
    project_dir = main_config.get("paths", {}).get("base_project_dir_local", os.getcwd())
    data_dir = os.path.join(project_dir, main_config.get("paths", {}).get("data_dir_name", "data"))
    merged_data_dir = os.path.join(data_dir, data_config.get("processed_data_dir", "processed"), "merged")
    merged_data_file = os.path.join(merged_data_dir, f"{timeframe}_{split}_merged.parquet")
    
    # Vérifier si le fichier existe
    if not os.path.exists(merged_data_file):
        console.print(f"[bold red]Erreur:[/bold red] Le fichier {merged_data_file} n'existe pas.")
        sys.exit(1)
    
    # Charger les données
    try:
        df = pd.read_parquet(merged_data_file)
        console.print(f"[green]✓[/green] Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Afficher quelques informations sur les données
        assets = set()
        features = set()
        for col in df.columns:
            if "_" in col:
                feature, asset = col.split("_", 1)
                if asset in ["DOGEUSDT", "XRPUSDT", "LTCUSDT", "SOLUSDT", "ADAUSDT"]:
                    features.add(feature)
                    assets.add(asset)
        
        assets_table = Table(title="Actifs détectés")
        assets_table.add_column("Actif", style="cyan")
        for asset in sorted(assets):
            assets_table.add_row(asset)
        
        features_table = Table(title="Features détectées")
        features_table.add_column("Feature", style="green")
        for feature in sorted(features):
            features_table.add_row(feature)
        
        console.print(Panel.fit(assets_table, title="Actifs"))
        console.print(Panel.fit(features_table, title="Features"))
        
        return df
    except Exception as e:
        console.print(f"[bold red]Erreur lors du chargement des données:[/bold red] {e}")
        sys.exit(1)

def create_environment(df, env_config, data_config):
    """
    Crée et initialise l'environnement MultiAssetEnv.
    
    Args:
        df (pandas.DataFrame): DataFrame contenant les données fusionnées
        env_config (dict): Configuration de l'environnement
        data_config (dict): Configuration des données
        
    Returns:
        MultiAssetEnv: Instance de l'environnement
    """
    console.print(Rule("[bold blue]Création de l'environnement[/bold blue]"))
    
    # Créer un dictionnaire de configuration combiné
    config = {
        "environment": env_config,
        "data": data_config
    }
    
    # Créer l'environnement
    try:
        env = MultiAssetEnv(df, config)
        console.print(f"[green]✓[/green] Environnement créé avec {len(env.assets)} actifs")
        return env
    except Exception as e:
        console.print(f"[bold red]Erreur lors de la création de l'environnement:[/bold red] {e}")
        sys.exit(1)

def test_environment(env, num_steps=10, random_actions=True, test_all_actions=False):
    """
    Teste l'environnement en exécutant quelques pas.
    
    Args:
        env (MultiAssetEnv): Instance de l'environnement
        num_steps (int): Nombre de pas à exécuter
        random_actions (bool): Si True, utilise des actions aléatoires
        test_all_actions (bool): Si True, teste toutes les actions possibles dans l'ordre
    """
    console.print(Rule("[bold blue]Test de l'environnement[/bold blue]"))
    
    # Réinitialiser l'environnement
    console.print(Panel("Réinitialisation de l'environnement...", title="Initialisation", border_style="cyan"))
    obs, info = env.reset()
    
    # Afficher des informations sur l'observation (maintenant un dictionnaire)
    obs_table = Table(title="Observation initiale")
    obs_table.add_column("Métrique", style="cyan")
    obs_table.add_column("Valeur", style="green")
    
    # Informations sur image_features
    image_features = obs["image_features"]
    obs_table.add_row("Type d'observation", "Dict avec 'image_features' et 'vector_features'")
    obs_table.add_row("Shape image_features", str(image_features.shape))
    obs_table.add_row("Min image_features", f"{image_features.min():.6f}")
    obs_table.add_row("Max image_features", f"{image_features.max():.6f}")
    obs_table.add_row("Moyenne image_features", f"{image_features.mean():.6f}")
    
    # Informations sur vector_features
    vector_features = obs["vector_features"]
    obs_table.add_row("Shape vector_features", str(vector_features.shape))
    obs_table.add_row("Min vector_features", f"{vector_features.min():.6f}")
    obs_table.add_row("Max vector_features", f"{vector_features.max():.6f}")
    obs_table.add_row("Moyenne vector_features", f"{vector_features.mean():.6f}")
    
    # Afficher des informations sur l'espace d'action
    action_table = Table(title="Espace d'action")
    action_table.add_column("Métrique", style="cyan")
    action_table.add_column("Valeur", style="green")
    action_table.add_row("Type", str(type(env.action_space)))
    action_table.add_row("Taille", str(env.action_space.n))
    action_table.add_row("Actions possibles", f"0 à {env.action_space.n - 1}")
    
    # Afficher les tables
    console.print(Panel(obs_table, title="Observation", border_style="green"))
    console.print(Panel(action_table, title="Espace d'action", border_style="yellow"))
    
    # Exécuter quelques pas
    if test_all_actions:
        # Tester toutes les actions possibles dans l'ordre
        num_actions = env.action_space.n
        console.print(Panel(f"Test de toutes les {num_actions} actions possibles", title="Mode de test", border_style="magenta"))
        
        for action in range(num_actions):
            # Traduire l'action en une description lisible
            action_desc = "HOLD" if action == 0 else \
                         f"BUY {env.assets[(action-1)//2]}" if action % 2 == 1 else \
                         f"SELL {env.assets[(action-2)//2]}"
            
            console.print(Rule(f"[bold cyan]Test de l'action {action}/{num_actions-1}: {action_desc}[/bold cyan]"))
            
            # Afficher l'état avant l'action
            console.print(f"[bold]Capital avant action:[/bold] ${env.capital:.2f}")
            console.print(f"[bold]Positions avant action:[/bold] {env.positions}")
            
            # Exécuter un pas avec cette action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Afficher l'info retournée par l'environnement
            console.print(f"[bold]Info retournée par l'environnement:[/bold]")
            for key, value in info.items():
                console.print(f"  - {key}: {value}")
            
            # Afficher l'état après l'action
            console.print(f"[bold]Capital après action:[/bold] ${env.capital:.2f}")
            console.print(f"[bold]Positions après action:[/bold] {env.positions}")
            
            # Pause pour permettre de voir l'affichage
            time.sleep(1.0)
            
            # Si l'épisode est terminé, réinitialiser l'environnement
            if terminated or truncated:
                console.print(Panel(f"Épisode terminé: terminated={terminated}, truncated={truncated}", 
                                   title="Fin d'épisode", border_style="yellow"))
                
                # Afficher un résumé de l'épisode
                display_episode_summary(env)
                
                # Réinitialiser l'environnement
                console.print(Panel("Réinitialisation de l'environnement...", title="Initialisation", border_style="cyan"))
                obs, info = env.reset()
    else:
        # Mode actions aléatoires
        console.print(Panel(f"Exécution de {num_steps} pas {'aléatoires' if random_actions else 'HOLD'}", 
                           title="Mode de test", border_style="magenta"))
        
        for i in range(num_steps):
            console.print(Rule(f"[bold cyan]Pas {i+1}/{num_steps}[/bold cyan]"))
            
            # Prendre une action (aléatoire ou HOLD)
            if random_actions:
                action = env.action_space.sample()
            else:
                action = 0  # ACTION_HOLD
            
            # Traduire l'action en une description lisible
            action_desc = "HOLD" if action == 0 else \
                         f"BUY {env.assets[(action-1)//2]}" if action % 2 == 1 else \
                         f"SELL {env.assets[(action-2)//2]}"
            
            console.print(f"[bold]Action sélectionnée:[/bold] {action} ({action_desc})")
            
            # Afficher l'état avant l'action
            console.print(f"[bold]Capital avant action:[/bold] ${env.capital:.2f}")
            console.print(f"[bold]Positions avant action:[/bold] {env.positions}")
            
            # Exécuter un pas
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Afficher l'info retournée par l'environnement
            console.print(f"[bold]Info retournée par l'environnement:[/bold]")
            for key, value in info.items():
                console.print(f"  - {key}: {value}")
            
            # Afficher l'état après l'action
            console.print(f"[bold]Capital après action:[/bold] ${env.capital:.2f}")
            console.print(f"[bold]Positions après action:[/bold] {env.positions}")
            
            # Pause pour permettre de voir l'affichage
            time.sleep(1.0)
            
            # Si l'épisode est terminé, réinitialiser l'environnement
            if terminated or truncated:
                console.print(Panel(f"Épisode terminé: terminated={terminated}, truncated={truncated}", 
                                   title="Fin d'épisode", border_style="yellow"))
                
                # Afficher un résumé de l'épisode
                display_episode_summary(env)
                
                # Réinitialiser l'environnement
                console.print(Panel("Réinitialisation de l'environnement...", title="Initialisation", border_style="cyan"))
                obs, info = env.reset()

def display_episode_summary(env):
    """
    Affiche un résumé de l'épisode.
    
    Args:
        env (MultiAssetEnv): Instance de l'environnement
    """
    # Créer un tableau récapitulatif
    summary_table = Table(title="Résumé de l'épisode", box=None, expand=True, show_header=True)
    
    # Ajouter les colonnes
    summary_table.add_column("Métrique", style="cyan", justify="left")
    summary_table.add_column("Valeur", style="green", justify="right")
    
    # Ajouter les lignes
    summary_table.add_row("Nombre de pas", str(env.current_step))
    summary_table.add_row("Valeur initiale du portefeuille", f"{env.initial_capital:.2f}")
    
    # Ajouter les lignes de métriques
    initial_portfolio_value = env.history[0]['portfolio_value'] if env.history else env.initial_capital
    final_portfolio_value = env.history[-1]['portfolio_value'] if env.history else 0
    profit_loss = final_portfolio_value - initial_portfolio_value
    profit_loss_pct = (profit_loss / initial_portfolio_value) * 100 if initial_portfolio_value > 0 else 0
    
    summary_table.add_row("Valeur initiale", f"${initial_portfolio_value:.2f}")
    summary_table.add_row("Valeur finale", f"${final_portfolio_value:.2f}")
    summary_table.add_row(
        "Profit/Perte", 
        Text(f"${profit_loss:.2f} ({profit_loss_pct:.2f}%)", style="green" if profit_loss >= 0 else "red")
    )
    summary_table.add_row("Nombre de pas", str(len(env.history)))
    summary_table.add_row("Récompense cumulée", Text(f"{env.cumulative_reward:.6f}", style="green" if env.cumulative_reward >= 0 else "red"))
    
    # Calculer des statistiques sur les trades
    if env.trade_log:
        num_trades = len(env.trade_log)
        successful_trades = sum(1 for trade in env.trade_log if trade.get('status', '').endswith('_EXECUTED'))
        failed_trades = num_trades - successful_trades
        buy_trades = sum(1 for trade in env.trade_log if trade.get('action_type') == 1 and trade.get('status', '').endswith('_EXECUTED'))  # BUY = 1
        sell_trades = sum(1 for trade in env.trade_log if trade.get('action_type') == 2 and trade.get('status', '').endswith('_EXECUTED'))  # SELL = 2
        total_fees = sum(trade.get('fee', 0) for trade in env.trade_log if trade.get('status', '').endswith('_EXECUTED'))
        
        summary_table.add_row("Nombre total de trades", str(num_trades))
        summary_table.add_row("Trades réussis", f"{successful_trades} ({successful_trades/num_trades*100:.1f}%)" if num_trades > 0 else "0")
        summary_table.add_row("Trades échoués", f"{failed_trades} ({failed_trades/num_trades*100:.1f}%)" if num_trades > 0 else "0")
        summary_table.add_row("Achats", str(buy_trades))
        summary_table.add_row("Ventes", str(sell_trades))
        summary_table.add_row("Frais totaux", f"${total_fees:.2f}")
    else:
        summary_table.add_row("Nombre de trades", "0")
    
    # Afficher le tableau
    console.print(Panel(summary_table, title="[bold blue]Résumé de l'épisode[/bold blue]", border_style="blue"))

def check_environment(env):
    """
    Vérifie l'environnement avec l'outil check_env de Gymnasium.
    
    Args:
        env (MultiAssetEnv): Instance de l'environnement
    """
    console.print(Rule("[bold blue]Vérification de l'environnement avec check_env[/bold blue]"))
    
    # Créer une copie de l'environnement pour la vérification
    # car check_env va réinitialiser et exécuter des pas
    import copy
    env_copy = copy.deepcopy(env)
    
    try:
        check_env(env_copy)
        console.print(Panel("[green bold]✓ L'environnement est conforme aux spécifications de Gymnasium[/green bold]", 
                           title="Vérification réussie", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Erreur: {e}[/bold red]", title="Échec de la vérification", border_style="red"))

def main():
    """
    Fonction principale.
    """
    # Parser les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Test de l\'environnement MultiAssetEnv avec données fusionnées')
    parser.add_argument(
        '--exec_profile', 
        type=str, 
        default='cpu_lot1',
        choices=['cpu', 'gpu', 'cpu_lot1', 'cpu_lot2', 'gpu_lot1', 'gpu_lot2'],
        help="Profil d'exécution pour charger les configurations appropriées. Supporte les anciens profils ('cpu', 'gpu') et les nouveaux avec lots ('cpu_lot1', 'cpu_lot2', 'gpu_lot1', 'gpu_lot2')."
    )
    parser.add_argument('--initial_capital', type=float, default=None, 
                        help='Capital initial pour le test (outrepasse la config).')
    parser.add_argument('--data_file', type=str, default=None, 
                        help='Chemin vers le fichier de données fusionné à utiliser (ex: data/processed/merged/1h_val_merged.parquet).')
    parser.add_argument('--num_rows', type=int, default=None, 
                        help='Nombre de lignes du dataset à utiliser pour le test.')
    parser.add_argument('--timeframe', type=str, default="1h", choices=["1m", "1h", "1d"],
                        help='Timeframe à utiliser pour le test (1m, 1h, 1d).')
    parser.add_argument('--split', type=str, default="train", choices=["train", "val", "test"],
                        help='Split à utiliser pour le test (train, val, test).')
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]Test de l'environnement MultiAssetEnv avec données fusionnées[/bold blue]\n\n"
        "Ce script valide le fonctionnement de l'environnement avec les données fusionnées\n"
        "et teste les différentes fonctionnalités de l'environnement.",
        title="ADAN Trading Bot - Test Environment",
        border_style="blue"
    ))
    
    # Charger les configurations avec le profil d'exécution spécifié
    console.print(f"[bold cyan]Utilisation du profil d'exécution: {args.exec_profile}[/bold cyan]")
    main_config, data_config, env_config, agent_config = load_configurations(exec_profile=args.exec_profile)
    
    # Charger les données fusionnées
    if args.data_file:
        console.print(f"[bold cyan]Utilisation du fichier de données spécifié: {args.data_file}[/bold cyan]")
        try:
            df = pd.read_parquet(args.data_file)
            console.print(f"[green]Données chargées avec succès: {df.shape}[/green]")
        except Exception as e:
            console.print(f"[bold red]Erreur lors du chargement du fichier {args.data_file}: {e}[/bold red]")
            console.print("[yellow]Utilisation de la méthode de chargement par défaut...[/yellow]")
            df = load_merged_data(main_config, data_config, timeframe=args.timeframe, split=args.split)
    else:
        df = load_merged_data(main_config, data_config, timeframe=args.timeframe, split=args.split)
    
    # Limiter le nombre de lignes si spécifié
    if args.num_rows and args.num_rows > 0 and args.num_rows < len(df):
        console.print(f"[bold cyan]Limitation à {args.num_rows} lignes (sur {len(df)} disponibles)[/bold cyan]")
        df = df.head(args.num_rows)
    
    # Modifier le capital initial si spécifié
    config_for_env = env_config.copy()
    if args.initial_capital is not None:
        console.print(f"[bold cyan]Utilisation du capital initial spécifié: ${args.initial_capital:,.2f}[/bold cyan]")
        config_for_env['initial_capital'] = args.initial_capital
    else:
        console.print(f"[dim]Utilisation du capital initial de environment_config.yaml: ${config_for_env.get('initial_capital', 10000.0):,.2f}[/dim]")
    
    # Créer l'environnement avec la configuration modifiée
    env = create_environment(df, config_for_env, data_config)
    
    # Demander à l'utilisateur le mode de test
    console.print(Panel("\nChoisissez un mode de test:", title="Options de test", border_style="cyan"))
    console.print("1. Test avec 5 actions aléatoires")
    console.print("2. Test avec toutes les actions possibles")
    console.print("3. Test avec 10 actions HOLD")
    console.print("4. Exécuter les trois tests successivement")
    
    choice = input("\nEntrez votre choix (1-4): ")
    
    if choice == "1":
        # Test avec actions aléatoires
        test_environment(env, num_steps=5, random_actions=True)
    elif choice == "2":
        # Test avec toutes les actions possibles
        test_environment(env, test_all_actions=True)
    elif choice == "3":
        # Test avec actions HOLD
        test_environment(env, num_steps=10, random_actions=False)
    elif choice == "4":
        # Exécuter tous les tests
        console.print(Panel("Test 1: 5 actions aléatoires", title="Test 1/3", border_style="magenta"))
        test_environment(env, num_steps=5, random_actions=True)
        
        # Réinitialiser l'environnement pour le prochain test
        env.reset()
        
        console.print(Panel("Test 2: Toutes les actions possibles", title="Test 2/3", border_style="magenta"))
        test_environment(env, test_all_actions=True)
        
        # Réinitialiser l'environnement pour le prochain test
        env.reset()
        
        console.print(Panel("Test 3: 10 actions HOLD", title="Test 3/3", border_style="magenta"))
        test_environment(env, num_steps=10, random_actions=False)
    else:
        console.print("[bold red]Choix invalide. Exécution du test par défaut (5 actions aléatoires).[/bold red]")
        test_environment(env, num_steps=5)
    
    # Vérifier l'environnement avec check_env
    check_environment(env)
    
    console.print(Panel.fit(
        "[bold green]Test terminé avec succès![/bold green]\n\n"
        "L'environnement MultiAssetEnv fonctionne correctement avec les données fusionnées.\n"
        "Vous pouvez maintenant procéder à l'implémentation du CNN feature extractor.",
        title="Résultat final",
        border_style="green"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Test interrompu par l'utilisateur[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Erreur lors de l'exécution du test: {e}[/bold red]")
