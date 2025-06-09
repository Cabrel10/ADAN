#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'évaluation rapide simplifié pour ADAN.
Utilise directement les données fusionnées existantes.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import load_config
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

# load_test_data is now modified to accept config and construct path dynamically
def load_test_data(config):
    """Charge les données de test fusionnées en fonction de la configuration."""
    try:
        project_root = config.get('paths', {}).get('base_project_dir_local', '.')
        data_dir_name = config.get('paths', {}).get('data_dir_name', 'data')
        processed_dir_name = config.get('data', {}).get('processed_data_dir', 'processed')
        
        timeframe_to_load = config.get('data', {}).get('training_timeframe', '1m') # Default if somehow not set
        lot_id = config.get('data', {}).get('lot_id', None)
        
        console.print(f"[cyan]Attempting to load test data for timeframe: {timeframe_to_load}[/cyan]")

        base_merged_path = os.path.join(project_root, data_dir_name, processed_dir_name, 'merged')
        unified_segment = 'unified'

        if lot_id:
            merged_dir = os.path.join(base_merged_path, lot_id, unified_segment)
        else:
            merged_dir = os.path.join(base_merged_path, unified_segment)
        
        file_name = f"{timeframe_to_load}_test_merged.parquet"
        test_file_path = os.path.join(merged_dir, file_name)
        
        console.print(f"Constructed test data path: {test_file_path}")

        if os.path.exists(test_file_path):
            df = pd.read_parquet(test_file_path)
            console.print(f"[green]✅ Données de test chargées depuis {test_file_path}: {df.shape}[/green]")
            return df
        else:
            console.print(f"[red]❌ Fichier de test introuvable: {test_file_path}[/red]")
            if not os.path.exists(merged_dir):
                console.print(f"  Le répertoire merged/unified ({merged_dir}) n'existe pas.")
                parent_merged_dir = os.path.dirname(merged_dir)
                if os.path.exists(parent_merged_dir):
                    console.print(f"  Contenu de {parent_merged_dir}: {os.listdir(parent_merged_dir)}")
            else:
                console.print(f"  Contenu de {merged_dir}: {os.listdir(merged_dir)}")
            return None
    except Exception as e:
        console.print(f"[red]❌ Erreur lors du chargement des données de test: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return None

def calculate_quick_metrics(env, initial_capital=15.0):
    """Calcule rapidement les métriques essentielles."""
    history = env.history
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    final_capital = df['capital'].iloc[-1] if 'capital' in df.columns else initial_capital
    final_portfolio = df['portfolio_value'].iloc[-1] if 'portfolio_value' in df.columns else initial_capital
    
    # ROI
    roi_capital_pct = ((final_capital - initial_capital) / initial_capital) * 100
    roi_portfolio_pct = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Nombre de trades
    trades = [h for h in history if h.get('action_type') in [1, 2]]  # BUY=1, SELL=2
    n_trades = len(trades)
    
    # Win rate approximatif
    positive_rewards = [h['reward'] for h in history if h.get('reward', 0) > 0]
    win_rate = len(positive_rewards) / len(history) * 100 if history else 0
    
    # Récompense moyenne
    avg_reward = np.mean([h.get('reward', 0) for h in history])
    final_reward = df['cumulative_reward'].iloc[-1] if 'cumulative_reward' in df.columns else 0
    
    return {
        'final_capital': final_capital,
        'final_portfolio': final_portfolio,
        'roi_capital_pct': roi_capital_pct,
        'roi_portfolio_pct': roi_portfolio_pct,
        'n_trades': n_trades,
        'win_rate_pct': win_rate,
        'avg_reward': avg_reward,
        'final_reward': final_reward,
        'total_steps': len(history)
    }

def classify_performance(metrics):
    """Classification rapide de performance."""
    roi = metrics.get('roi_portfolio_pct', 0)
    win_rate = metrics.get('win_rate_pct', 0)
    
    if roi > 20 and win_rate > 60:
        return "🏆 EXCELLENT"
    elif roi > 10 and win_rate > 50:
        return "🥈 BON"
    elif roi > 0 and win_rate > 40:
        return "🥉 ACCEPTABLE"
    elif roi > -10:
        return "⚠️ RISQUÉ"
    else:
        return "❌ CRITIQUE"

def main():
    parser = argparse.ArgumentParser(description='ADAN - Test Modèle Rapide')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle à tester (.zip)')
    parser.add_argument('--capital', type=float, default=15.0,
                        help='Capital initial (défaut: 15$)')
    parser.add_argument('--steps', type=int, default=500,
                        help='Nombre de steps par épisode (défaut: 500)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Nombre d\'épisodes (défaut: 3)')
    parser.add_argument(
        '--exec_profile',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu', 'smoke_cpu'],
        help="Profil d'exécution pour charger data_config_{profile}.yaml (défaut: cpu)"
    )
    parser.add_argument(
        '--training_timeframe',
        type=str,
        default='1m', # Defaulting to '1m' as this is a quick test script
        choices=['1m', '1h', '1d'],
        help="Timeframe d'évaluation (e.g., '1m', '1h', '1d', défaut: 1m)"
    )
    
    args = parser.parse_args()
    
    # Affichage de démarrage
    console.print(Panel.fit(
        f"[bold cyan]⚡ ADAN - Test Modèle Rapide[/bold cyan]\n"
        f"[yellow]Modèle: {os.path.basename(args.model_path)}[/yellow]\n"
        f"[green]Capital: ${args.capital:.2f}[/green]\n"
        f"[blue]Episodes: {args.episodes} × {args.steps} steps[/blue]",
        title="Test Express"
    ))
    
    # Vérifications
    if not os.path.exists(args.model_path):
        console.print(f"[bold red]❌ Modèle non trouvé: {args.model_path}[/bold red]")
        return 1
    
    try:
        # Charger modèle
        console.print("[cyan]🔄 Chargement du modèle...[/cyan]")
        model = PPO.load(args.model_path)
        
        # Charger données de test
        console.print("[cyan]🔄 Chargement des données de test...[/cyan]")
        # df_test = load_test_data() # Will be called after config is loaded

        # Charger les configurations
        console.print(f"[cyan]🔄 Chargement des configurations pour le profil: {args.exec_profile}...[/cyan]")
        main_config = load_config('config/main_config.yaml')
        data_config_path = f'config/data_config_{args.exec_profile}.yaml'
        data_config = load_config(data_config_path)
        env_config_path = 'config/environment_config.yaml' # Assuming this is standard
        env_config = load_config(env_config_path)

        if not main_config or not data_config or not env_config:
            console.print("[red]❌ Erreur: Un ou plusieurs fichiers de configuration n'ont pu être chargés.[/red]")
            return 1

        config = {
            # 'main': main_config, # Keep full main_config if other parts of it are needed later
            'paths': main_config.get('paths', {}),
            'data': data_config,  # Keep full data_config
            'environment': env_config.get('environment', {}) # Get the 'environment' sub-dictionary
        }
        
        # Override du capital initial depuis les arguments
        if 'environment' not in config: config['environment'] = {}
        config['environment']['initial_capital'] = args.capital

        # Override du training_timeframe depuis les arguments
        if 'data' not in config: config['data'] = {}
        config['data']['training_timeframe'] = args.training_timeframe # Already defaults to '1m' in args

        console.print(f"[green]✅ Configurations chargées. Timeframe pour évaluation: {config['data']['training_timeframe']}[/green]")

        # Charger les données de test en utilisant la config (qui contient maintenant le bon timeframe)
        df_test = load_test_data(config) # Pass config to load_test_data
        if df_test is None:
            return 1

        # Créer environnement
        console.print("[cyan]🔄 Création de l'environnement...[/cyan]")
        # console.print(f"[yellow]⚠️ Utilisation de toutes les features disponibles: {df_test.shape[1]} colonnes[/yellow]") # To be removed
        env = MultiAssetEnv(
            df_received=df_test,
            config=config, # Pass the correct config
            # scaler=None, # Scaler not typically used for quick test
            # encoder=None,
            max_episode_steps_override=args.steps
        )
        # env.initial_capital = args.capital # This is now set via config['environment']['initial_capital']
        
        # Évaluation
        all_metrics = []
        
        with Progress(
            TextColumn("[bold blue]Test en cours"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Episodes", total=args.episodes)
            
            for episode in range(args.episodes):
                obs, _ = env.reset()
                episode_steps = 0
                done = False
                
                while not done and episode_steps < args.steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_steps += 1
                    
                    if done or truncated:
                        break
                
                # Calculer métriques de l'épisode
                metrics = calculate_quick_metrics(env, args.capital)
                all_metrics.append(metrics)
                
                progress.update(task, advance=1)
        
        # Moyenner les métriques
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Classification
        classification = classify_performance(avg_metrics)
        
        # Affichage résultats
        console.print(f"\n[bold]{classification}[/bold]")
        
        results_table = Table(title="[bold green]📊 Résultats Test Rapide[/bold green]")
        results_table.add_column("Métrique", style="dim cyan")
        results_table.add_column("Valeur", style="bright_white")
        
        results_table.add_row("💰 Capital Final", f"${avg_metrics['final_capital']:.2f}")
        results_table.add_row("📊 Portefeuille Final", f"${avg_metrics['final_portfolio']:.2f}")
        results_table.add_row("📈 ROI Capital", f"{avg_metrics['roi_capital_pct']:+.2f}%")
        results_table.add_row("🎯 ROI Portefeuille", f"{avg_metrics['roi_portfolio_pct']:+.2f}%")
        results_table.add_row("🔄 Nombre de Trades", f"{avg_metrics['n_trades']:.0f}")
        results_table.add_row("🏆 Taux de Victoire", f"{avg_metrics['win_rate_pct']:.1f}%")
        results_table.add_row("⭐ Récompense Moyenne", f"{avg_metrics['avg_reward']:.4f}")
        results_table.add_row("📏 Steps Moyens", f"{avg_metrics['total_steps']:.0f}")
        
        console.print(results_table)
        
        # Recommandations
        roi = avg_metrics['roi_portfolio_pct']
        if roi > 10:
            recommendation = "[green]✅ Modèle excellent! Prêt pour trading réel.[/green]"
        elif roi > 0:
            recommendation = "[yellow]⚠️ Performance acceptable, surveillance recommandée.[/yellow]"
        else:
            recommendation = "[red]❌ Performance insuffisante, re-entraînement nécessaire.[/red]"
        
        console.print(Panel(
            f"{recommendation}\n\n"
            f"[cyan]Gain/Perte estimé sur ${args.capital:.2f}:[/cyan] {roi/100 * args.capital:+.2f}$\n"
            f"[cyan]Performance par trade:[/cyan] {avg_metrics['roi_portfolio_pct']/max(1, avg_metrics['n_trades']):.2f}% par trade",
            title="Recommandation Express"
        ))
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]❌ Erreur: {str(e)}[/bold red]")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())