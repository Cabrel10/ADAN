#!/usr/bin/env python3
"""
BACKTEST FINAL RIGOUREUX - ÉVALUATION CRITIQUE AVANT LIVE/SUPPRESSION
Date: 2025-11-23
Objectif: Tester le modèle sur données 2021-2022 (BTC) avec validation exhaustive
Risque: Data leakage, overfitting, erreurs cachées
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader


class RigorousBacktester:
    """Backtest rigoureux avec validation exhaustive"""
    
    def __init__(self, config_path="config/config.yaml", 
                 checkpoint_path="checkpoints_final/adan_model_checkpoint_640000_steps.zip"):
        logger.info("=" * 80)
        logger.info("INITIALISATION BACKTEST RIGOUREUX")
        logger.info("=" * 80)
        
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Charger config
        logger.info(f"Chargement config: {config_path}")
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        
        # Vérifier checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
        logger.info(f"✅ Checkpoint trouvé: {checkpoint_path}")
        
        # Données
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []
        self.equity_curve = []
        
        # Métriques
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_pnl': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'initial_capital': 0.0,
            'final_equity': 0.0,
            'total_return_pct': 0.0,
            'steps': 0
        }
    
    def verify_data_integrity(self):
        """Vérifier l'intégrité des données - CRITIQUE"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: VÉRIFICATION INTÉGRITÉ DES DONNÉES")
        logger.info("=" * 80)
        
        # Charger données BTC 5m
        btc_5m_path = "data/processed/indicators/train/BTCUSDT/5m.parquet"
        logger.info(f"Chargement données: {btc_5m_path}")
        
        df = pd.read_parquet(btc_5m_path)
        logger.info(f"✅ Données chargées: {len(df)} rows")
        
        # Vérifier colonnes
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        logger.info(f"✅ Toutes les colonnes présentes: {required_cols}")
        
        # Vérifier index (timestamp)
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index n'est pas DatetimeIndex, conversion...")
            df.index = pd.to_datetime(df.index)
        
        logger.info(f"Période complète: {df.index.min()} → {df.index.max()}")
        
        # Filtrer 2021-2022
        start_date = pd.Timestamp('2021-01-01')
        end_date = pd.Timestamp('2022-12-31 23:59:59')
        
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        logger.info(f"Période 2021-2022: {len(df_filtered)} rows")
        
        if len(df_filtered) == 0:
            logger.warning("⚠️ Aucune donnée pour 2021-2022, utilisant 2022 complet")
            start_date = pd.Timestamp('2022-01-01')
            end_date = pd.Timestamp('2022-12-31 23:59:59')
            df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
            logger.info(f"Période 2022: {len(df_filtered)} rows")
        
        # Vérifier NaN
        nan_count = df_filtered.isnull().sum().sum()
        logger.info(f"NaN values: {nan_count}")
        if nan_count > 0:
            logger.warning(f"⚠️ {nan_count} NaN détectés, remplissage forward-fill")
            df_filtered = df_filtered.fillna(method='ffill').fillna(method='bfill')
        
        # Vérifier prix valides
        if (df_filtered['close'] <= 0).any():
            raise ValueError("Prix négatifs ou zéro détectés!")
        logger.info(f"✅ Prix valides: min={df_filtered['close'].min():.2f}, max={df_filtered['close'].max():.2f}")
        
        # Vérifier volumes
        if (df_filtered['volume'] < 0).any():
            raise ValueError("Volumes négatifs détectés!")
        logger.info(f"✅ Volumes valides: min={df_filtered['volume'].min():.0f}, max={df_filtered['volume'].max():.0f}")
        
        # Vérifier continuité temporelle
        time_diffs = df_filtered.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs != expected_diff]
        if len(gaps) > 0:
            logger.warning(f"⚠️ {len(gaps)} gaps détectés dans les données")
            logger.info(f"   Gaps: {gaps.head()}")
        
        logger.info("✅ Vérification intégrité: RÉUSSIE")
        return df_filtered
    
    def run_backtest(self, df_data):
        """Exécuter le backtest"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: EXÉCUTION BACKTEST")
        logger.info("=" * 80)
        
        # Configuration
        initial_capital = 20.5
        self.config['initial_capital'] = initial_capital
        self.config['environment']['assets'] = ['BTCUSDT']
        
        logger.info(f"Capital initial: ${initial_capital}")
        logger.info(f"Asset: BTCUSDT")
        logger.info(f"Période: {df_data.index.min()} → {df_data.index.max()}")
        
        # Créer environnement
        logger.info("Création environnement...")
        try:
            env = MultiAssetChunkedEnv(
                config=self.config, 
                worker_id=0, 
                log_level="WARNING"
            )
            logger.info("✅ Environnement créé")
        except Exception as e:
            logger.error(f"❌ Erreur création environnement: {e}")
            raise
        
        # Charger modèle
        logger.info(f"Chargement modèle: {self.checkpoint_path}")
        try:
            model = PPO.load(self.checkpoint_path, env=env)
            logger.info("✅ Modèle chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
        
        # Exécuter évaluation
        logger.info("Lancement évaluation...")
        obs, _ = env.reset()
        done = False
        step = 0
        
        try:
            while not done and step < 100000:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Enregistrer métriques
                pm = env.portfolio_manager
                self.portfolio_values.append(pm.equity)
                self.timestamps.append(env.current_timestamp if hasattr(env, 'current_timestamp') else step)
                
                if step % 5000 == 0:
                    logger.info(f"  Step {step}: Equity=${pm.equity:.2f}, Portfolio=${pm.portfolio_value:.2f}")
                
                step += 1
        
        except Exception as e:
            logger.error(f"❌ Erreur pendant évaluation: {e}")
            logger.error(f"   Step: {step}")
            raise
        
        logger.info(f"✅ Évaluation terminée: {step} steps")
        
        # Extraire trades
        self.extract_trades(env)
        
        # Calculer métriques
        self.calculate_metrics(initial_capital, step)
        
        logger.info("✅ Backtest complété")
    
    def extract_trades(self, env):
        """Extraire les trades du portfolio manager"""
        logger.info("\nExtraction des trades...")
        
        pm = env.portfolio_manager
        if not hasattr(pm, 'trade_log'):
            logger.warning("⚠️ Pas de trade_log")
            return
        
        trade_log = list(pm.trade_log)
        logger.info(f"Trade log: {len(trade_log)} entrées")
        
        # Parser events
        for event in trade_log:
            if event.get('asset') != 'BTCUSDT':
                continue
            
            event_type = (event.get('event') or event.get('action', '')).lower()
            ts = event.get('timestamp')
            
            if not ts:
                continue
            
            if event_type == 'open':
                price = event.get('price')
                size = event.get('size')
                if price and size:
                    self.trades.append({
                        'time': ts,
                        'type': 'OPEN',
                        'price': float(price),
                        'size': float(size),
                        'pnl': 0.0,
                        'sl': float(event.get('sl')) if event.get('sl') else None,
                        'tp': float(event.get('tp')) if event.get('tp') else None,
                    })
            
            elif event_type == 'close':
                price = event.get('exit_price')
                size = event.get('size')
                pnl = event.get('pnl', 0)
                if price and size:
                    self.trades.append({
                        'time': ts,
                        'type': 'CLOSE',
                        'price': float(price),
                        'size': float(size),
                        'pnl': float(pnl),
                    })
        
        logger.info(f"✅ {len(self.trades)} trades extraits")
        
        open_trades = [t for t in self.trades if t['type'] == 'OPEN']
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        logger.info(f"   OPEN: {len(open_trades)}, CLOSE: {len(close_trades)}")
    
    def calculate_metrics(self, initial_capital, steps):
        """Calculer les métriques"""
        logger.info("\nCalcul des métriques...")
        
        self.metrics['initial_capital'] = initial_capital
        self.metrics['steps'] = steps
        
        if self.portfolio_values:
            self.metrics['final_equity'] = self.portfolio_values[-1]
        else:
            self.metrics['final_equity'] = initial_capital
        
        # Return
        if initial_capital > 0:
            self.metrics['total_return_pct'] = (
                (self.metrics['final_equity'] - initial_capital) / initial_capital * 100
            )
        
        # Drawdown
        if self.portfolio_values:
            equity_array = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - peak) / peak
            self.metrics['max_drawdown'] = np.min(drawdown) * 100
        
        # Trades
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        self.metrics['total_trades'] = len(close_trades)
        
        if close_trades:
            pnls = [t['pnl'] for t in close_trades]
            self.metrics['total_pnl'] = sum(pnls)
            
            winning = [p for p in pnls if p > 0]
            losing = [p for p in pnls if p < 0]
            
            self.metrics['winning_trades'] = len(winning)
            self.metrics['losing_trades'] = len(losing)
            
            if len(close_trades) > 0:
                self.metrics['win_rate'] = len(winning) / len(close_trades) * 100
            
            if len(losing) > 0:
                self.metrics['gross_profit'] = sum(winning) if winning else 0
                self.metrics['gross_loss'] = abs(sum(losing))
                self.metrics['profit_factor'] = self.metrics['gross_profit'] / self.metrics['gross_loss'] if self.metrics['gross_loss'] > 0 else float('inf')
        
        # Sharpe ratio
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
            if len(returns) > 0 and np.std(returns) > 0:
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12)
    
    def validate_results(self):
        """Valider les résultats - CRITIQUE"""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: VALIDATION DES RÉSULTATS")
        logger.info("=" * 80)
        
        issues = []
        
        # Check 1: Capital final > initial
        if self.metrics['final_equity'] < self.metrics['initial_capital'] * 0.5:
            issues.append(f"⚠️ Capital final très bas: ${self.metrics['final_equity']:.2f} (initial: ${self.metrics['initial_capital']:.2f})")
        
        # Check 2: Trades > 0
        if self.metrics['total_trades'] == 0:
            issues.append("⚠️ ZÉRO TRADES - Modèle n'a pas tradé!")
        
        # Check 3: Win rate valide
        if self.metrics['win_rate'] < 0 or self.metrics['win_rate'] > 100:
            issues.append(f"⚠️ Win rate invalide: {self.metrics['win_rate']:.2f}%")
        
        # Check 4: Drawdown valide
        if self.metrics['max_drawdown'] > -5 and self.metrics['max_drawdown'] < 0:
            issues.append(f"⚠️ Drawdown très faible: {self.metrics['max_drawdown']:.2f}% (possible overfitting)")
        
        # Check 5: Return cohérent
        if self.metrics['total_return_pct'] < -50:
            issues.append(f"⚠️ Return très négatif: {self.metrics['total_return_pct']:.2f}%")
        
        # Check 6: Profit factor
        if self.metrics['profit_factor'] < 0.8 and self.metrics['profit_factor'] != float('inf'):
            issues.append(f"⚠️ Profit factor faible: {self.metrics['profit_factor']:.2f}")
        
        # Check 7: Sharpe ratio
        if self.metrics['sharpe_ratio'] < -2:
            issues.append(f"⚠️ Sharpe ratio très négatif: {self.metrics['sharpe_ratio']:.2f}")
        
        if issues:
            logger.warning("⚠️ PROBLÈMES DÉTECTÉS:")
            for issue in issues:
                logger.warning(f"   {issue}")
        else:
            logger.info("✅ Tous les checks de validation: RÉUSSIS")
        
        return len(issues) == 0
    
    def generate_report(self):
        """Générer rapport final"""
        logger.info("\n" + "=" * 80)
        logger.info("RAPPORT FINAL - BACKTEST RIGOUREUX")
        logger.info("=" * 80)
        
        logger.info("\n📊 MÉTRIQUES DE PERFORMANCE")
        logger.info("-" * 80)
        logger.info(f"Capital initial:           ${self.metrics['initial_capital']:.2f}")
        logger.info(f"Capital final:             ${self.metrics['final_equity']:.2f}")
        logger.info(f"Total return:              {self.metrics['total_return_pct']:.2f}%")
        logger.info(f"Max drawdown:              {self.metrics['max_drawdown']:.2f}%")
        logger.info(f"Total steps:               {self.metrics['steps']}")
        
        logger.info("\n📈 STATISTIQUES DE TRADING")
        logger.info("-" * 80)
        logger.info(f"Total trades (CLOSE):      {self.metrics['total_trades']}")
        logger.info(f"Winning trades:            {self.metrics['winning_trades']}")
        logger.info(f"Losing trades:             {self.metrics['losing_trades']}")
        logger.info(f"Win rate:                  {self.metrics['win_rate']:.2f}%")
        logger.info(f"Profit factor:             {self.metrics['profit_factor']:.2f}")
        logger.info(f"Gross profit:              ${self.metrics['gross_profit']:.2f}")
        logger.info(f"Gross loss:                ${self.metrics['gross_loss']:.2f}")
        logger.info(f"Total PnL:                 ${self.metrics['total_pnl']:.2f}")
        logger.info(f"Sharpe ratio:              {self.metrics['sharpe_ratio']:.2f}")
        
        logger.info("\n" + "=" * 80)
        
        # Décision
        logger.info("\n🎯 DÉCISION FINALE")
        logger.info("=" * 80)
        
        if self.metrics['total_trades'] == 0:
            decision = "❌ SUPPRESSION - Modèle n'a pas tradé"
        elif self.metrics['total_return_pct'] < -30:
            decision = "❌ SUPPRESSION - Perte excessive"
        elif self.metrics['win_rate'] < 30:
            decision = "❌ SUPPRESSION - Win rate trop faible"
        elif self.metrics['max_drawdown'] < -40:
            decision = "❌ SUPPRESSION - Drawdown trop élevé"
        elif self.metrics['total_return_pct'] > 0 and self.metrics['win_rate'] > 40:
            decision = "✅ LIVE - Modèle performant"
        else:
            decision = "⚠️ RÉVISION - Résultats mitigés"
        
        logger.info(decision)
        logger.info("=" * 80)
        
        return decision


def main():
    """Exécuter le backtest"""
    try:
        backtest = RigorousBacktester()
        
        # Phase 1: Vérifier intégrité
        df_data = backtest.verify_data_integrity()
        
        # Phase 2: Exécuter backtest
        backtest.run_backtest(df_data)
        
        # Phase 3: Valider résultats
        is_valid = backtest.validate_results()
        
        # Phase 4: Rapport final
        decision = backtest.generate_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST TERMINÉ")
        logger.info("=" * 80)
        
        return 0 if is_valid else 1
    
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}")
        logger.error("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
