#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour tester le StateBuilder et vérifier que les features sont correctement trouvées.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import load_config, get_logger
from src.adan_trading_bot.data_processing.data_loader import load_merged_data
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from src.adan_trading_bot.environment.state_builder import StateBuilder

logger = get_logger(__name__)
console = Console()

def load_configurations(exec_profile='cpu'):
    """
    Charge les fichiers de configuration nécessaires.
    
    Args:
        exec_profile (str): Profil d'exécution ('cpu' ou 'gpu')
    
    Returns:
        tuple: (main_config, data_config, env_config, agent_config)
    """
    console.print(f"[bold blue]Chargement des configurations (profil: {exec_profile})[/bold blue]")
    
    # Chemins des fichiers de configuration
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    main_config_path = os.path.join(config_dir, "main_config.yaml")
    data_config_path = os.path.join(config_dir, f"data_config_{exec_profile}.yaml")
    env_config_path = os.path.join(config_dir, "environment_config.yaml")
    
    console.print(f"[dim]Utilisation des fichiers de configuration:[/dim]")
    console.print(f"[dim]- Main config: {main_config_path}[/dim]")
    console.print(f"[dim]- Data config: {data_config_path}[/dim]")
    console.print(f"[dim]- Environment config: {env_config_path}[/dim]")
    
    # Charger les configurations
    main_config = load_config(main_config_path)
    data_config = load_config(data_config_path)
    env_config = load_config(env_config_path)
    
    return main_config, data_config, env_config

def test_state_builder(data_file, num_rows, exec_profile):
    """
    Teste le StateBuilder avec un fichier de données fusionnées.
    
    Args:
        data_file (str): Chemin vers le fichier de données fusionnées
        num_rows (int): Nombre de lignes à utiliser
        exec_profile (str): Profil d'exécution ('cpu' ou 'gpu')
    """
    # Charger les configurations
    main_config, data_config, env_config = load_configurations(exec_profile)
    
    # Charger les données fusionnées
    if data_file:
        console.print(f"[bold cyan]Chargement des données depuis {data_file}[/bold cyan]")
        try:
            df = pd.read_parquet(data_file)
            console.print(f"[bold green]Données chargées directement depuis le fichier spécifié[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Erreur lors du chargement du fichier {data_file}: {e}[/bold red]")
            console.print("[yellow]Tentative de chargement avec la méthode par défaut...[/yellow]")
            df = load_merged_data(data_config)
    else:
        # Utiliser la fonction load_merged_data avec la configuration
        df = load_merged_data(data_config)
    
    if df is None or df.empty:
        console.print(f"[bold red]Erreur: Impossible de charger les données[/bold red]")
        return
    
    # Limiter le nombre de lignes si spécifié
    if num_rows and num_rows > 0 and num_rows < len(df):
        df = df.iloc[:num_rows]
    
    console.print(f"[bold green]Données chargées: {len(df)} lignes, {len(df.columns)} colonnes[/bold green]")
    
    # Créer une table pour afficher les premières colonnes
    table = Table(title="Aperçu des colonnes")
    table.add_column("Index", style="cyan")
    table.add_column("Nom de colonne", style="green")
    
    for i, col in enumerate(df.columns[:30]):
        table.add_row(str(i), col)
    
    console.print(table)
    
    # Créer un environnement MultiAssetEnv
    console.print(f"[bold cyan]Création de l'environnement MultiAssetEnv[/bold cyan]")
    # Créer un dictionnaire de configuration combiné
    combined_config = {
        'main': main_config,
        'data': data_config,
        'environment': env_config
    }
    # Instancier l'environnement avec les arguments positionnels corrects
    env = MultiAssetEnv(df, combined_config)
    
    # Obtenir le StateBuilder
    state_builder = env.state_builder
    
    # Afficher les informations sur le StateBuilder
    console.print(f"[bold cyan]Informations sur le StateBuilder[/bold cyan]")
    console.print(f"Assets: {state_builder.assets}")
    console.print(f"Base features: {state_builder.base_feature_names}")
    console.print(f"CNN input window size: {state_builder.cnn_input_window_size}")
    
    # Tester la construction de l'observation
    console.print(f"[bold cyan]Test de la construction de l'observation[/bold cyan]")
    
    # Reset de l'environnement
    obs, _ = env.reset()
    
    # Vérifier que l'observation a la bonne forme
    console.print(f"Observation image_features shape: {obs['image_features'].shape}")
    console.print(f"Observation vector_features shape: {obs['vector_features'].shape}")
    
    # Vérifier les features trouvées et manquantes
    found_features = []
    missing_features = []
    
    for asset in state_builder.assets:
        for base_feature in state_builder.base_feature_names:
            column_to_find = f"{base_feature}_{asset}"
            if column_to_find in df.columns:
                found_features.append(column_to_find)
            else:
                missing_features.append(column_to_find)
    
    # Afficher les résultats
    console.print(f"[bold green]Features trouvées: {len(found_features)}/{len(state_builder.assets) * len(state_builder.base_feature_names)}[/bold green]")
    
    if missing_features:
        console.print(f"[bold red]Features manquantes: {len(missing_features)}/{len(state_builder.assets) * len(state_builder.base_feature_names)}[/bold red]")
        console.print(f"Exemples de features manquantes: {missing_features[:10]}")
    else:
        console.print(f"[bold green]Toutes les features ont été trouvées![/bold green]")
    
    # Tester l'exécution d'un ordre
    console.print(f"[bold cyan]Test de l'exécution d'un ordre[/bold cyan]")
    
    # Obtenir les prix actuels
    prices = env._get_current_prices()
    console.print(f"Prix actuels: {prices}")
    
    # Exécuter un ordre d'achat pour le premier actif
    if len(prices) > 0:
        asset = list(prices.keys())[0]
        price = prices[asset]
        console.print(f"Test d'achat de {asset} au prix ${price:.4f}")
        
        # Exécuter l'ordre
        result = env.order_manager.execute_order(
            action_type=1,  # BUY
            asset_id=asset,
            quantity=None,
            current_price=price,
            order_type="MARKET",
            allocated_value_usdt=100.0,  # Allouer 100 USDT
            capital=env.capital,
            positions=env.positions  # Ajouter l'argument positions manquant
        )
        
        console.print(f"Résultat de l'ordre: {result}")
    else:
        console.print(f"[bold red]Aucun prix disponible pour tester l'exécution d'un ordre[/bold red]")
    
    console.print(f"[bold green]Test terminé![/bold green]")

def main():
    """
    Fonction principale.
    """
    parser = argparse.ArgumentParser(description='Test du StateBuilder et de la recherche des features')
    parser.add_argument(
        '--exec_profile', 
        type=str, 
        default='cpu',
        choices=['cpu', 'gpu'],
        help="Profil d'exécution ('cpu' ou 'gpu')"
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help="Chemin vers le fichier de données fusionnées (ex: data/processed/merged/1h_train_merged.parquet)"
    )
    parser.add_argument(
        '--num_rows',
        type=int,
        default=100,
        help="Nombre de lignes à utiliser pour le test"
    )
    args = parser.parse_args()
    
    # Déterminer le fichier de données par défaut si non spécifié
    if args.data_file is None:
        # Construire le chemin par défaut
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_data_file = os.path.join(
            project_root,
            "data",
            "processed",
            "merged",
            "1h_train_merged.parquet"
        )
        args.data_file = default_data_file
        console.print(f"[yellow]Aucun fichier de données spécifié. Utilisation du fichier par défaut: {args.data_file}[/yellow]")
    
    # Tester le StateBuilder
    test_state_builder(args.data_file, args.num_rows, args.exec_profile)

if __name__ == "__main__":
    main()
