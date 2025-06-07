#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'évaluation rapide pour ADAN avec capital de 15$.
Teste rapidement les performances d'un modèle avec gestion des flux monétaires.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adan_trading_bot.common.utils import get_path, load_config
from src.adan_trading_bot.common.custom_logger import setup_logging
from src.adan_trading_bot.data_processing.feature_engineer import prepare_data_pipeline
from src.adan_trading_bot.environment.multi_asset_env import MultiAssetEnv
from stable_baselines3 import PPO
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

console = Console()

def calculate_quick_metrics(env, initial_capital=15.0):
    """Calcule rapidement les métriques essentielles."""
    history = env.history
    if not history:
        return {}
    
    df = pd.DataFrame(history)
    final_capital = df['capital'].iloc[-1] if 'capital' in df.columns else initial_capital
    final_portfolio = df['portfolio_value'].iloc[-1] if 'portfolio_value' in df.columns else initial_capital
    
    # ROI
    roi_pct = ((final_capital - initial_capital) / initial_capital) * 100
    portfolio_roi_pct = ((final_portfolio - initial_capital) / initial_capital) * 100
    
    # Nombre de trades
    trades = [h for h in history if h.get('action_type') in [1, 2]]  # BUY=1, SELL=2
    n_trades = len(trades)
    
    # Win rate approximatif (épisodes avec reward positif)
    positive_rewards = [h['reward'] for h in history if h.get('reward', 0) > 0]
    win_rate = len(positive_rewards) / len(history) * 100 if history else 0
    
    # Récompense moyenne et finale
    avg_reward = np.mean([h.get('reward', 0) for h in history])
    final_reward = df['cumulative_reward'].iloc[-1] if 'cumulative_reward' in df.columns else 0
    
    return {
        'final_capital': final_capital,
        'final_portfolio': final_portfolio,
        'roi_capital_pct': roi_pct,
        'roi_portfolio_pct': portfolio_roi_pct,
        'n_trades': n_trades,
        'win_rate_pct': win_rate,
        'avg_reward': avg_reward,
        'final_reward': final_reward,
        'total_steps': len(history)
    }

def classify_quick_performance(metrics):
    """Classification rapide de performance."""
    roi = metrics.get('roi_portfolio_pct', 0)
    win_rate = metrics.get('win_rate_pct', 0)
    trades = metrics.get('n_trades', 0)
    
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
    parser = argparse.ArgumentParser(description='ADAN - Évaluation Rapide avec Capital 15$')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle à évaluer (.zip)')
    parser.add_argument('--profile', type=str, default='cpu',
                        choices=['cpu', 'gpu'],
                        help='Profil de configuration')
    parser.add_argument('--steps', type=int, default=500,
                        help='Nombre de steps d\'évaluation (défaut: 500)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Nombre d\'épisodes (défaut: 3)')
    parser.add_argument('--capital', type=float, default=15.0,
                        help='Capital initial (défaut: 15$)')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging('config/logging_config.yaml')
    
    console.print(Panel.fit(
        f"[bold cyan]⚡ ADAN - Évaluation Rapide[/bold cyan]\n"
        f"[yellow]Modèle: {os.path.basename(args.model_path)}[/yellow]\n"
        f"[green]Capital: ${args.capital:.2f}[/green]\n"
        f"[blue]Episodes: {args.episodes} × {args.steps} steps[/blue]",
        title="Évaluation Express"
    ))
    
    # Vérifications
    if not os.path.exists(args.model_path):
        console.print(f"[bold red]❌ Modèle non trouvé: {args.model_path}[/bold red]")
        return 1
    
    try:
        # Charger modèle
        console.print("[cyan]🔄 Chargement du modèle...[/cyan]")
        model = PPO.load(args.model_path)
        
        # Charger configs
        console.print("[cyan]🔄 Chargement des configurations...[/cyan]")
        main_config = load_config('config/main_config.yaml')
        data_config = load_config(f'config/data_config_{args.profile}.yaml')
        env_config = load_config('config/environment_config.yaml')
        agent_config = load_config(f'config/agent_config_{args.profile}.yaml')
        
        # Reconstruire la structure de config attendue par prepare_data_pipeline
        combined_config = {
            'paths': main_config.get('paths', {}),
            'data': data_config,
            'environment': env_config,
            'agent': agent_config,
            'general': main_config.get('general', {})
        }
        
        # Préparer données test
        console.print("[cyan]🔄 Préparation des données de test...[/cyan]")
        train_df, val_df, df_test = prepare_data_pipeline(
            combined_config, 
            is_training=True
        )
        scaler = None
        encoder = None
        
        # Créer environnement
        env = MultiAssetEnv(
            df_received=df_test,
            config=combined_config,
            scaler=scaler,
            encoder=encoder,
            max_episode_steps_override=args.steps
        )
        env.initial_capital = args.capital
        
        # Évaluation rapide
        all_metrics = []
        
        with Progress(
            TextColumn("[bold blue]Évaluation"),
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
        classification = classify_quick_performance(avg_metrics)
        
        # Affichage résultats
        console.print(f"\n[bold]{classification}[/bold]")
        
        results_table = Table(title="[bold green]📊 Résultats Évaluation Rapide[/bold green]")
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
        
        # Recommandations rapides
        roi = avg_metrics['roi_portfolio_pct']
        if roi > 10:
            recommendation = "[green]✅ Modèle prêt pour trading réel avec ce capital![/green]"
        elif roi > 0:
            recommendation = "[yellow]⚠️ Performance acceptable, surveillance recommandée[/yellow]"
        else:
            recommendation = "[red]❌ Performance insuffisante, re-entraînement nécessaire[/red]"
        
        console.print(Panel(
            f"{recommendation}\n\n"
            f"[cyan]Capital optimal estimé:[/cyan] ${args.capital * (1 + roi/100):.2f}\n"
            f"[cyan]Gain/Perte sur 15$:[/cyan] {roi/100 * args.capital:+.2f}$",
            title="Recommandation Express"
        ))
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]❌ Erreur: {str(e)}[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())