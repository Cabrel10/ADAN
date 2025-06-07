#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test ciblé pour vérifier les actions de trading dans l'environnement MultiAssetEnv.
Ce script teste spécifiquement les actions BUY pour vérifier si l'agent peut prendre des positions.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules du projet
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.common.utils import load_config, get_logger

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("trading_test")
console = Console()

def load_configurations():
    """
    Charge les fichiers de configuration nécessaires.
    
    Returns:
        tuple: (main_config, data_config, env_config, agent_config)
    """
    console.print(Rule("[bold blue]Chargement des configurations[/bold blue]"))
    
    # Chemins des fichiers de configuration
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    main_config_path = os.path.join(config_dir, "main_config.yaml")
    data_config_path = os.path.join(config_dir, "data_config.yaml")
    env_config_path = os.path.join(config_dir, "environment_config.yaml")
    agent_config_path = os.path.join(config_dir, "agent_config.yaml")
    
    # Charger les configurations
    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)
    env_config = load_config(env_config_path)
    agent_config = load_config(agent_config_path)
    
    console.print("✓ Configurations chargées avec succès")
    
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
    
    # Chemin du fichier de données fusionnées
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "merged")
    data_file = os.path.join(data_dir, f"{timeframe}_{split}_merged.parquet")
    
    # Vérifier si le fichier existe
    if not os.path.exists(data_file):
        console.print(f"[bold red]Erreur: Le fichier {data_file} n'existe pas.[/bold red]")
        sys.exit(1)
    
    # Charger les données
    df = pd.read_parquet(data_file)
    
    console.print(f"✓ Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Détecter les actifs à partir des colonnes (format 'close_ASSETNAME')
    assets = [col.split('_', 1)[1] for col in df.columns if col.startswith('close_')]
    
    # Afficher les actifs détectés
    assets_table = Table(title="Actifs détectés")
    assets_table.add_column("Actif")
    for asset in assets:
        assets_table.add_row(asset)
    console.print(Panel(assets_table, title="Actifs", border_style="cyan"))
    
    # Détecter les features à partir des colonnes (format 'FEATURE_ASSETNAME')
    features = list(set([col.split('_')[0] for col in df.columns if '_' in col]))
    
    # Afficher les features détectées
    features_table = Table(title="Features détectées")
    features_table.add_column("Feature")
    for feature in features:
        features_table.add_row(feature)
    console.print(Panel(features_table, title="Features", border_style="cyan"))
    
    return df

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
    
    # Créer une copie de la configuration pour la modifier
    config = {
        'environment': env_config,
        'data': data_config
    }
    
    # Créer l'environnement
    env = MultiAssetEnv(df, config)
    
    console.print(f"✓ Environnement créé avec {len(env.assets)} actifs")
    
    return env

def test_buy_actions(env):
    """
    Teste spécifiquement les actions BUY pour vérifier si l'agent peut prendre des positions.
    
    Args:
        env (MultiAssetEnv): Instance de l'environnement
    """
    console.print(Rule("[bold blue]Test des actions BUY[/bold blue]"))
    
    # Réinitialiser l'environnement
    console.print(Panel("Réinitialisation de l'environnement...", title="Initialisation", border_style="cyan"))
    obs, info = env.reset()
    
    # Afficher le capital initial et les positions
    console.print(f"[bold]Capital initial:[/bold] ${env.capital:.2f}")
    console.print(f"[bold]Positions initiales:[/bold] {env.positions}")
    console.print(f"[bold]Prix actuels:[/bold] {env._get_current_prices()}")
    
    # Tester chaque action BUY
    for i, asset in enumerate(env.assets):
        action = 1 + i*2  # Les actions BUY sont impaires (1, 3, 5, ...)
        
        console.print(Rule(f"[bold cyan]Test de l'action BUY pour {asset} (action={action})[/bold cyan]"))
        
        # Afficher l'état avant l'action
        console.print(f"[bold]Capital avant action:[/bold] ${env.capital:.2f}")
        console.print(f"[bold]Positions avant action:[/bold] {env.positions}")
        
        # Obtenir le palier actuel
        current_tier = env.reward_calculator.get_current_tier(env.capital)
        console.print(f"[bold]Palier actuel:[/bold] {current_tier}")
        
        # Obtenir le prix actuel de l'actif
        current_prices = env._get_current_prices()
        current_price = current_prices.get(asset, 0)
        console.print(f"[bold]Prix actuel de {asset}:[/bold] ${current_price:.4f}")
        
        # Calculer la taille de position attendue
        expected_quantity = env._get_position_size(asset, current_price, current_tier)
        expected_value = expected_quantity * current_price
        console.print(f"[bold]Taille de position attendue:[/bold] {expected_quantity:.6f} {asset} (valeur: ${expected_value:.2f})")
        
        # Exécuter un pas avec cette action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Afficher l'info retournée par l'environnement
        console.print(f"[bold]Info retournée par l'environnement:[/bold]")
        for key, value in info.items():
            console.print(f"  - {key}: {value}")
        
        # Afficher l'état après l'action
        console.print(f"[bold]Capital après action:[/bold] ${env.capital:.2f}")
        console.print(f"[bold]Positions après action:[/bold] {env.positions}")
        
        # Vérifier si la position a été prise
        position_taken = asset in env.positions
        if position_taken:
            console.print(f"[bold green]✓ Position prise pour {asset}[/bold green]")
            console.print(f"  - Quantité: {env.positions[asset]['qty']:.6f}")
            console.print(f"  - Prix d'entrée: ${env.positions[asset]['price']:.4f}")
            console.print(f"  - Valeur: ${env.positions[asset]['qty'] * current_price:.2f}")
        else:
            console.print(f"[bold red]✗ Aucune position prise pour {asset}[/bold red]")
            console.print(f"  - Raison possible: voir les logs pour plus de détails")
        
        # Pause pour permettre de voir l'affichage
        time.sleep(1.0)
    
    # Afficher un résumé final
    console.print(Rule("[bold blue]Résumé final[/bold blue]"))
    console.print(f"[bold]Capital final:[/bold] ${env.capital:.2f}")
    console.print(f"[bold]Positions finales:[/bold] {env.positions}")
    
    # Vérifier si des positions ont été prises
    if env.positions:
        console.print("[bold green]✓ L'agent a réussi à prendre des positions[/bold green]")
    else:
        console.print("[bold red]✗ L'agent n'a pas réussi à prendre de positions[/bold red]")
        console.print("  - Vérifiez les logs pour plus de détails")

def main():
    """
    Fonction principale.
    """
    console.print(Panel.fit(
        "[bold blue]Test des actions de trading dans l'environnement MultiAssetEnv[/bold blue]\n\n"
        "Ce script teste spécifiquement les actions BUY pour vérifier si l'agent peut prendre des positions.",
        title="ADAN Trading Bot - Test Trading Actions",
        border_style="blue"
    ))
    
    # Charger les configurations
    main_config, data_config, env_config, agent_config = load_configurations()
    
    # Modifier le capital initial si nécessaire
    env_config['initial_capital'] = 10000.0
    console.print(f"[bold cyan]Capital initial configuré à: ${env_config['initial_capital']:,.2f}[/bold cyan]")
    
    # Charger les données fusionnées
    df = load_merged_data(main_config, data_config, timeframe="1h", split="train")
    
    # Limiter le nombre de lignes pour le test
    num_rows = 100
    if num_rows > 0 and num_rows < len(df):
        console.print(f"[bold cyan]Limitation à {num_rows} lignes (sur {len(df)} disponibles)[/bold cyan]")
        df = df.head(num_rows)
    
    # Créer l'environnement
    env = create_environment(df, env_config, data_config)
    
    # Tester les actions BUY
    test_buy_actions(env)
    
    console.print(Panel.fit(
        "[bold green]Test terminé![/bold green]\n\n"
        "Consultez les logs pour plus de détails sur l'exécution des actions.",
        title="Résultat final",
        border_style="green"
    ))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Test interrompu par l'utilisateur[/bold yellow]")
    except Exception as e:
        logger.exception("Erreur lors de l'exécution du test")
        console.print(f"\n[bold red]Erreur lors de l'exécution du test: {e}[/bold red]")
