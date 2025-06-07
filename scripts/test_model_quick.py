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

def load_training_configs():
    """Charge les configurations utilisées pendant l'entraînement."""
    try:
        # Charger les configurations avec les mêmes profils que l'entraînement
        main_config = load_config('config/main_config.yaml')
        data_config = load_config('config/data_config_cpu.yaml')
        env_config = load_config('config/environment_config.yaml')
        
        # Fusionner les configurations comme pendant l'entraînement
        config = {
            'data': data_config.get('data', {}),
            'environment': env_config.get('environment', {}),
            'main': main_config
        }
        
        console.print(f"[green]✅ Configurations chargées - Assets: {config['data'].get('assets', [])}[/green]")
        return config
    except Exception as e:
        console.print(f"[red]❌ Erreur lors du chargement des configs: {str(e)}[/red]")
        return None
</text>

<old_text>
        # Charger les configurations utilisées pendant l'entraînement
        console.print("[cyan]🔄 Chargement des configurations...[/cyan]")
        config = load_training_configs()
        if config is None:
            return 1
        
        # Override du capital initial
        config['environment']['initial_capital'] = args.capital

def load_test_data():
    """Charge directement les données de test fusionnées."""
    test_file = "data/processed/merged/unified/1m_test_merged.parquet"
    
    if not os.path.exists(test_file):
        console.print(f"[red]❌ Fichier de test non trouvé: {test_file}[/red]")
        return None
    
    try:
        df = pd.read_parquet(test_file)
        console.print(f"[green]✅ Données de test chargées: {df.shape}[/green]")
        return df
    except Exception as e:
        console.print(f"[red]❌ Erreur lors du chargement: {str(e)}[/red]")
        return None

def load_training_configs():
    """Charge les configurations utilisées pendant l'entraînement."""
    try:
        # Charger les configurations avec les mêmes profils que l'entraînement
        main_config = load_config('config/main_config.yaml')
        data_config = load_config('config/data_config_cpu.yaml')
        env_config = load_config('config/environment_config.yaml')
        
        # Fusionner les configurations comme pendant l'entraînement
        config = {
            'data': data_config.get('data', {}),
            'environment': env_config.get('environment', {}),
            'main': main_config
        }
        
        console.print(f"[green]✅ Configurations chargées - Assets: {config['data'].get('assets', [])}[/green]")
        return config
    except Exception as e:
        console.print(f"[red]❌ Erreur lors du chargement des configs: {str(e)}[/red]")
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
        df_test = load_test_data()
        if df_test is None:
            return 1
        
        # Charger configurations basiques
        config = {
            'data': {
                'assets': ["ADAUSDT", "BNBUSDT", "BTCUSDT", "ETHUSDT", "XRPUSDT"],
                'training_timeframe': '1m',
                'base_market_features': [
                    "open", "high", "low", "close", "volume"
                ]
            },
            'environment': {
                'initial_capital': args.capital,
                'transaction': {'fee_percent': 0.001, 'fixed_fee': 0.0},
                'order_rules': {'min_value_tolerable': 1.0, 'min_value_absolute': 0.5},
                'penalties': {
                    'time_step': -0.001,
                    'invalid_order_base': -0.3,
                    'out_of_funds': -0.5,
                    'max_positions_reached': -0.2
                },
                'reward_tiers': [
                    {'threshold': 0, 'max_positions': 3, 'allocation_frac_per_pos': 0.2, 'reward_pos_mult': 1.0, 'reward_neg_mult': 1.0}
                ]
            }
        }
        
        # Créer environnement
        console.print("[cyan]🔄 Création de l'environnement...[/cyan]")
        console.print(f"[yellow]⚠️ Utilisation de toutes les features disponibles: {df_test.shape[1]} colonnes[/yellow]")
        env = MultiAssetEnv(
            df_received=df_test,
            config=config,
            scaler=None,
            encoder=None,
            max_episode_steps_override=args.steps
        )
        env.initial_capital = args.capital
        
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