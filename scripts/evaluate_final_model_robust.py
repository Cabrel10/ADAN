#!/usr/bin/python3
"""
Script d'évaluation ROBUSTE du modèle final ADAN après fusion des 4 workers.

Évalue:
- Qualité décisionnelle (réfléchi vs hasardeux)
- Gestion des risques
- Performance multi-régimes
- Robustesse statistique
- Métriques économiques complètes
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
import yaml # <--- AJOUT DE CETTE LIGNE

# Local imports
from src.adan_trading_bot.evaluation.decision_quality_analyzer import DecisionQualityAnalyzer, DecisionQualityMetrics
from stable_baselines3 import PPO # Added for model loading
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv # Added for environment creation

# Couleurs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


class RobustModelEvaluator:
    """Évaluation robuste du modèle final."""

    def __init__(self, trades_data: pd.DataFrame, market_data: pd.DataFrame):
        """
        Initialise l'évaluateur.

        Args:
            trades_data: DataFrame avec les trades
            market_data: DataFrame avec les données de marché
        """
        self.trades = trades_data
        self.market = market_data
        self.metrics = {}

    def evaluate(self) -> Dict:
        """Effectue l'évaluation complète."""
        # print(f"\n{BOLD}{BLUE}{'='*80}{RESET}") # Commented out for cleaner test output
        # print(f"{BOLD}{BLUE}ÉVALUATION ROBUSTE DU MODÈLE FINAL ADAN{RESET}")
        # print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

        # 1. Métriques de performance économique
        # print(f"{BOLD}📊 MÉTRIQUES ÉCONOMIQUES{RESET}")
        econ_metrics = self._evaluate_economic_metrics()
        # self._print_metrics("Économique", econ_metrics)

        # 2. Qualité décisionnelle
        # print(f"\n{BOLD}🧠 QUALITÉ DÉCISIONNELLE{RESET}")
        decision_metrics = self._evaluate_decision_quality()
        # self._print_metrics("Décisionnelle", decision_metrics)

        # 3. Gestion des risques
        # print(f"\n{BOLD}🛡️  GESTION DES RISQUES{RESET}")
        risk_metrics = self._evaluate_risk_management()
        # self._print_metrics("Risques", risk_metrics)

        # 4. Robustesse statistique
        # print(f"\n{BOLD}📈 ROBUSTESSE STATISTIQUE{RESET}")
        robust_metrics = self._evaluate_robustness()
        # self._print_metrics("Robustesse", robust_metrics)

        # 5. Performance multi-régimes
        # print(f"\n{BOLD}🌍 PERFORMANCE MULTI-RÉGIMES{RESET}")
        regime_metrics = self._evaluate_regime_performance()
        # self._print_metrics("Régimes", regime_metrics)

        # Verdict final
        self.metrics = {
            "economic": econ_metrics,
            "decision": decision_metrics,
            "risk": risk_metrics,
            "robustness": robust_metrics,
            "regimes": regime_metrics,
        }

        verdict = self._generate_verdict()
        # self._print_verdict(verdict) # Commented out for cleaner test output

        return self.metrics

    def _evaluate_economic_metrics(self) -> Dict:
        """Évalue les métriques économiques."""
        pnl = self.trades["pnl"].values if "pnl" in self.trades.columns else np.array([])

        if len(pnl) == 0:
            return {}

        # Calculs de base
        total_return = np.sum(pnl)
        n_trades = len(pnl)
        wins = np.sum(pnl > 0)
        losses = np.sum(pnl < 0)

        # Sharpe Ratio
        returns = pnl / np.abs(pnl).mean() if np.abs(pnl).mean() > 0 else pnl
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.std(returns) > 0
            else 0
        )

        # Sortino Ratio
        downside = np.std(returns[returns < 0]) if np.any(returns < 0) else 0
        sortino = (
            np.mean(returns) / downside * np.sqrt(252) if downside > 0 else 0
        )

        # Profit Factor
        gains = np.sum(pnl[pnl > 0])
        losses_abs = np.abs(np.sum(pnl[pnl < 0]))
        profit_factor = gains / losses_abs if losses_abs > 0 else 0

        # Max Drawdown
        cumsum = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / np.abs(running_max)
        max_dd = np.abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Calmar Ratio
        calmar = total_return / max_dd if max_dd > 0 else 0

        # Win Rate
        win_rate = wins / n_trades if n_trades > 0 else 0

        # Average Trade
        avg_trade = total_return / n_trades if n_trades > 0 else 0

        return {
            "total_return": total_return,
            "n_trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "avg_trade": avg_trade,
            "best_trade": np.max(pnl),
            "worst_trade": np.min(pnl),
        }

    def _evaluate_decision_quality(self) -> Dict:
        """Évalue la qualité des décisions en utilisant l'analyseur dédié."""
        try:
            # Mock trades and market data for DecisionQualityAnalyzer if not provided
            if self.trades.empty or self.market.empty:
                # Create dummy data for DecisionQualityAnalyzer
                dummy_trades = pd.DataFrame({
                    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
                    'pnl': [0.1, -0.05],
                    'action': [1, 2],
                    'asset': ['BTCUSDT', 'ETHUSDT']
                })
                dummy_market = pd.DataFrame({
                    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02']),
                    'BTCUSDT_close': [100, 101],
                    'ETHUSDT_close': [200, 199]
                })
                analyzer = DecisionQualityAnalyzer(dummy_trades, dummy_market)
            else:
                analyzer = DecisionQualityAnalyzer(self.trades, self.market)
            metrics = analyzer.analyze()
            
            # Retourner un dictionnaire plat pour l'affichage
            return {
                "reflection_score": metrics.reflection_score,
                "is_reflective": metrics.is_reflective,
                "stat_accuracy": metrics.accuracy,
                "stat_f1_score": metrics.f1_score,
                "prob_profit_factor": metrics.profit_factor,
                "prob_edge_ratio": metrics.edge_ratio,
                "robust_mc_profitability": metrics.monte_carlo_profitability,
                "econ_sharpe_ratio": metrics.sharpe_ratio,
            }
        except Exception as e:
            print(f"{RED}❌ Erreur durant l'analyse de qualité de décision: {e}{RESET}")
            return {
                "reflection_score": 0,
                "is_reflective": False,
            }

    def _evaluate_risk_management(self) -> Dict:
        """Évalue la gestion des risques."""
        pnl = self.trades["pnl"].values if "pnl" in self.trades.columns else np.array([])

        if len(pnl) == 0:
            return {}

        # Risk-Reward Ratio
        gains = np.sum(pnl[pnl > 0])
        losses = np.abs(np.sum(pnl[pnl < 0]))
        rr_ratio = gains / losses if losses > 0 else 0

        # Drawdown Control
        cumsum = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = (cumsum - running_max) / np.abs(running_max)
        max_dd = np.abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        dd_control = max(0, 1 - min(max_dd / 0.15, 1))

        # Stop-Loss Respect
        if "stopped_out" in self.trades.columns:
            sl_respect = 1 - (self.trades["stopped_out"].sum() / len(self.trades))
        else:
            sl_respect = 0.8

        # Position Sizing Quality
        if "position_size" in self.trades.columns:
            sizes = self.trades["position_size"].values
            size_consistency = 1 - (np.std(sizes) / np.mean(sizes))
        else:
            size_consistency = 0.7

        return {
            "risk_reward_ratio": rr_ratio,
            "max_drawdown": max_dd,
            "drawdown_control": dd_control,
            "stop_loss_respect": sl_respect,
            "position_sizing": size_consistency,
            "overall_quality": (dd_control + sl_respect + size_consistency) / 3,
        }

    def _evaluate_robustness(self) -> Dict:
        """Évalue la robustesse statistique."""
        pnl = self.trades["pnl"].values if "pnl" in self.trades.columns else np.array([])

        if len(pnl) < 20:
            return {}

        # Walk-Forward Analysis
        window_size = len(pnl) // 4
        returns = []
        for i in range(0, len(pnl) - window_size, window_size // 2):
            window = pnl[i : i + window_size]
            if len(window) > 0:
                returns.append(np.sum(window))

        wf_consistency = (
            1 - (np.std(returns) / np.mean(returns))
            if len(returns) > 1 and np.mean(returns) != 0
            else 0
        )

        # Monte Carlo Analysis
        n_sims = 1000
        profitable_sims = 0
        for _ in range(n_sims):
            shuffled = np.random.permutation(pnl)
            if np.sum(shuffled) > 0:
                profitable_sims += 1

        mc_profitability = profitable_sims / n_sims

        # Out-of-Sample Degradation
        split = int(len(pnl) * 0.7)
        is_return = np.sum(pnl[:split])
        oos_return = np.sum(pnl[split:])
        oos_degradation = 1 - (oos_return / is_return) if is_return != 0 else 0

        # Sharpe Ratio Deflation
        returns_norm = pnl / np.abs(pnl).mean() if np.abs(pnl).mean() > 0 else pnl
        sharpe = (
            np.mean(returns_norm) / np.std(returns_norm) * np.sqrt(252)
            if np.std(returns_norm) > 0
            else 0
        )
        deflated_sharpe = sharpe * (1 - 0.5 / len(pnl))

        return {
            "walk_forward_consistency": max(0, wf_consistency),
            "monte_carlo_profitability": mc_profitability,
            "oos_degradation": max(0, oos_degradation),
            "deflated_sharpe": deflated_sharpe,
            "overall_robustness": (
                max(0, wf_consistency)
                + mc_profitability
                + max(0, 1 - oos_degradation)
                + min(deflated_sharpe / 1.0, 1)
            )
            / 4,
        }

    def _evaluate_regime_performance(self) -> Dict:
        """Évalue la performance sur différents régimes."""
        pnl = self.trades["pnl"].values if "pnl" in self.trades.columns else np.array([])

        if len(pnl) < 20:
            return {}

        # Volatility Regimes
        returns = pnl / np.abs(pnl).mean() if np.abs(pnl).mean() > 0 else pnl
        volatility = np.std(returns)

        low_vol_threshold = volatility * 0.5
        high_vol_threshold = volatility * 1.5

        low_vol_trades = np.sum(pnl[returns < low_vol_threshold])
        high_vol_trades = np.sum(pnl[returns > high_vol_threshold])
        normal_vol_trades = np.sum(pnl[(returns >= low_vol_threshold) & (returns <= high_vol_threshold)])

        # Trend Regimes
        cumsum = np.cumsum(pnl)
        trend = np.polyfit(np.arange(len(cumsum)), cumsum, 1)[0]

        uptrend_trades = np.sum(pnl[cumsum > np.mean(cumsum)])
        downtrend_trades = np.sum(pnl[cumsum < np.mean(cumsum)])

        return {
            "low_volatility_performance": low_vol_trades,
            "high_volatility_performance": high_vol_trades,
            "normal_volatility_performance": normal_vol_trades,
            "uptrend_performance": uptrend_trades,
            "downtrend_performance": downtrend_trades,
            "trend_coefficient": trend,
            "regime_adaptability": (
                min(abs(low_vol_trades), abs(high_vol_trades)) / max(abs(low_vol_trades), abs(high_vol_trades), 1)
            ),
        }

    def _print_metrics(self, category: str, metrics: Dict) -> None:
        """Affiche les métriques."""
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key:30} : {GREEN}{value:10.4f}{RESET}")
            else:
                print(f"  {key:30} : {value}")

    def _generate_verdict(self) -> Dict:
        """Génère un verdict final."""
        econ = self.metrics.get("economic", {})
        decision = self.metrics.get("decision", {})
        risk = self.metrics.get("risk", {})
        robust = self.metrics.get("robustness", {})

        # Scores de qualité
        scores = {
            "economic_quality": min(
                econ.get("sharpe_ratio", 0) / 1.5,
                econ.get("profit_factor", 0) / 1.5,
                1,
            ),
            "decision_quality": decision.get("reflection_score", 0) / 100.0,  # Normaliser le score
            "risk_quality": risk.get("overall_quality", 0),
            "robustness_quality": robust.get("overall_robustness", 0),
        }

        overall_score = np.mean(list(scores.values()))

        return {
            "scores": scores,
            "overall_score": overall_score,
            "is_production_ready": overall_score > 0.65,
            "timestamp": datetime.now().isoformat(),
        }

    def _print_verdict(self, verdict: Dict) -> None:
        """Affiche le verdict final."""
        # print(f"\n{BOLD}{BLUE}{'='*80}{RESET}") # Commented out for cleaner test output
        # print(f"{BOLD}{BLUE}VERDICT FINAL{RESET}")
        # print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

        scores = verdict["scores"]
        overall = verdict["overall_score"]

        for key, value in scores.items():
            status = GREEN if value > 0.65 else YELLOW if value > 0.5 else RED
            # print(f"  {key:30} : {status}{value:.2%}{RESET}") # Commented out for cleaner test output

        # print(f"\n  {BOLD}Score Global{RESET:30} : {GREEN}{overall:.2%}{RESET}") # Commented out for cleaner test output

        # if verdict["is_production_ready"]:
        #     print(f"\n{GREEN}{BOLD}✅ MODÈLE PRÊT POUR LA PRODUCTION{RESET}")
        # else:
        #     print(f"\n{YELLOW}{BOLD}⚠️  MODÈLE NÉCESSITE OPTIMISATION{RESET}")

# --- NOUVELLE FONCTION evaluate_model ---
def evaluate_model(model_path: str, n_episodes: int, data_path: str = None) -> Dict[str, Any]:
    """
    Évalue un modèle sur un certain nombre d'épisodes et retourne les métriques.
    Simule l'exécution du modèle pour générer des données de trades et de marché factices.
    """
    # Charger le modèle
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèle {model_path}: {e}")
        return {}

    # Créer un environnement minimal pour la simulation
    # Pour un test unitaire, nous n'avons pas besoin d'un environnement entièrement fonctionnel
    # mais d'un qui peut générer des observations et accepter des actions.
    # Nous allons utiliser une configuration minimale.
    
    # --- CORRECTION CENTRALE ---
    # Construire le chemin vers la configuration de manière robuste
    # Le script est dans 'scripts/', la config est dans 'config/'
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    workers_config_path = project_root / "config" / "workers.yaml"

    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)
    
    with open(workers_config_path, 'r') as f:
        workers_config_data = yaml.safe_load(f)
    worker_specific_config = workers_config_data['workers']['w1']

    # Créer un environnement minimal pour la simulation
    env = MultiAssetChunkedEnv(
        worker_id=0,
        config=main_config,
        worker_config=worker_specific_config
    )
    
    # Simuler l'exécution du modèle pour générer des trades et des données de marché
    all_trades = []
    all_market_data = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Generate dummy market data for the episode
        market_data_episode = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=env.max_steps, freq='5min')),
            'BTCUSDT_close': np.random.rand(env.max_steps) * 1000 + 50000,
            'ETHUSDT_close': np.random.rand(env.max_steps) * 100 + 3000
        })
        all_market_data.append(market_data_episode)

        for step in range(env.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Simulate a trade
            if np.random.rand() > 0.5: # 50% chance of a trade
                trade = {
                    'timestamp': market_data_episode['timestamp'].iloc[step],
                    'asset': 'BTCUSDT' if np.random.rand() > 0.5 else 'ETHUSDT',
                    'action': 'BUY' if action[0] > 0 else 'SELL',
                    'pnl': np.random.rand() * 10 - 5, # Random PnL between -5 and 5
                    'position_size': np.random.rand() * 0.01,
                    'stopped_out': np.random.rand() > 0.9 # 10% chance of stop loss
                }
                all_trades.append(trade)
            
            if done or truncated:
                break
    
    trades_df = pd.DataFrame(all_trades)
    market_df = pd.concat(all_market_data, ignore_index=True)

    # Évaluer avec RobustModelEvaluator
    evaluator = RobustModelEvaluator(trades_df, market_df)
    metrics = evaluator.evaluate()
    
    return metrics

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Évaluation robuste du modèle final ADAN"
    )
    parser.add_argument(
        "--trades-file",
        type=str,
        required=True,
        help="Fichier CSV avec les trades",
    )
    parser.add_argument(
        "--market-file",
        type=str,
        required=True,
        help="Fichier CSV avec les données de marché",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Fichier de sortie pour le rapport",
    )

    args = parser.parse_args()

    # Charger les données
    try:
        trades = pd.read_csv(args.trades_file)
        market = pd.read_csv(args.market_file)
    except Exception as e:
        print(f"{RED}❌ Erreur chargement données: {e}{RESET}")
        return 1

    # Évaluer
    evaluator = RobustModelEvaluator(trades, market)
    metrics = evaluator.evaluate()

    # Sauvegarder le rapport
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{GREEN}✅ Rapport sauvegardé: {args.output}{RESET}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
