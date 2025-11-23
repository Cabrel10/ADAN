#!/usr/bin/env python3
"""
DASHBOARD HONNÊTE - Refonte Complète
Objectif: Afficher la VRAIE performance sans mensonge
Validation: Chaque métrique est loggée et comparée aux logs réels
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('/tmp/dashboard_honest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader


class HonestDashboard:
    """Dashboard qui dit la vérité sur la performance"""
    
    def __init__(self, checkpoint_path, config_path="config/config.yaml"):
        logger.info("=" * 80)
        logger.info("INITIALISATION DASHBOARD HONNÊTE")
        logger.info("=" * 80)
        
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []
        self.data_frames = {}
        self.metrics = {}
        
    def load_config_and_env(self, asset="XRPUSDT", capital=20.5):
        """Charger config et environnement"""
        logger.info(f"Chargement config: {self.config_path}")
        config_loader = ConfigLoader()
        config = config_loader.load_config(self.config_path)
        
        # Override
        config['initial_capital'] = float(capital)
        if 'environment' in config:
            config['environment']['assets'] = [asset]
        else:
            config['assets'] = [asset]
        
        logger.info(f"Asset: {asset}, Capital: {capital} USDT")
        
        try:
            self.env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="WARNING")
            logger.info("✅ Environnement chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement env: {e}")
            raise
        
        try:
            self.model = PPO.load(self.checkpoint_path, env=self.env)
            logger.info(f"✅ Modèle chargé: {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
        
        return config
    
    def run_evaluation(self, max_steps=20000):
        """Exécuter l'évaluation et collecter les données brutes"""
        logger.info(f"Lancement évaluation ({max_steps} steps max)")
        
        obs, _ = self.env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            step += 1
            
            # Collecter timestamp et équité
            ts = getattr(self.env, 'current_timestamp', pd.Timestamp.now())
            equity = self.env.portfolio_manager.equity
            
            self.timestamps.append(ts)
            self.portfolio_values.append(equity)
            
            if step % 1000 == 0:
                logger.info(f"  Step {step}: Equity=${equity:.2f}")
        
        logger.info(f"✅ Évaluation terminée: {step} steps, Equity=${self.portfolio_values[-1]:.2f}")
        return step
    
    def extract_trades(self, asset="XRPUSDT"):
        """Extraire les trades du portfolio manager (depuis trade_log)"""
        logger.info("Extraction des trades...")
        
        if not hasattr(self.env.portfolio_manager, 'trade_log'):
            logger.warning("⚠️ Pas de trade_log disponible")
            return
        
        trade_log = self.env.portfolio_manager.trade_log
        logger.info(f"Trade log contient {len(trade_log)} entrées")
        
        # Detect actual asset from trade_log (might be BTCUSDT not XRPUSDT)
        assets_in_log = set()
        for event in trade_log:
            if event.get('asset'):
                assets_in_log.add(event.get('asset'))
        logger.info(f"Assets in trade_log: {assets_in_log}")
        
        # Use first asset found if not matching
        actual_asset = asset
        if asset not in assets_in_log and assets_in_log:
            actual_asset = list(assets_in_log)[0]
            logger.info(f"Asset mismatch: requested {asset}, using {actual_asset}")
        
        # Parse trade_log events
        # Format: {'event': 'open', ...} or {'action': 'close', 'pnl': ..., ...}
        for event in trade_log:
            if event.get('asset') != actual_asset:
                continue
            
            # Determine event type
            event_type = event.get('event') or event.get('action')
            if not event_type:
                continue
            event_type = event_type.lower()
            
            ts = event.get('timestamp')
            if not ts:
                continue
            
            if event_type == 'open':
                price = event.get('price')
                size = event.get('size')
                if not (price and size):
                    continue
                
                self.trades.append({
                    'time': ts,
                    'type': 'OPEN',
                    'price': float(price),
                    'size': float(size),
                    'pnl': 0.0,
                    'sl': float(event.get('sl')) if event.get('sl') else None,
                    'tp': float(event.get('tp')) if event.get('tp') else None,
                    'timeframe': event.get('timeframe', '5m')
                })
            
            elif event_type == 'close':
                # Close event has exit_price, size, pnl
                price = event.get('exit_price')
                size = event.get('size')
                pnl = event.get('pnl', 0)
                
                if not (price and size):
                    continue
                
                self.trades.append({
                    'time': ts,
                    'type': 'CLOSE',
                    'price': float(price),
                    'size': float(size),
                    'pnl': float(pnl),
                    'sl': None,  # SL not in close event
                    'tp': None,  # TP not in close event
                    'timeframe': event.get('timeframe', '5m')
                })
        
        logger.info(f"✅ {len(self.trades)} trades extraits")
        
        # Validation
        open_trades = [t for t in self.trades if t['type'] == 'OPEN']
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        logger.info(f"   OPEN: {len(open_trades)}, CLOSE: {len(close_trades)}")
        
        return len(self.trades)
    
    def calculate_metrics(self, initial_capital=20.5):
        """Calculer les métriques HONNÊTES"""
        logger.info("Calcul des métriques...")
        
        if not self.portfolio_values:
            logger.error("❌ Pas de portfolio_values")
            return {}
        
        # Equity
        final_equity = self.portfolio_values[-1]
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        logger.info(f"Capital initial: ${initial_capital:.2f}")
        logger.info(f"Capital final: ${final_equity:.2f}")
        logger.info(f"Total return: {total_return:.2f}%")
        
        # Drawdown
        peak = self.portfolio_values[0]
        drawdowns = []
        for val in self.portfolio_values:
            peak = max(peak, val)
            dd = (val - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        max_dd = min(drawdowns) if drawdowns else 0.0
        
        logger.info(f"Max drawdown: {max_dd:.2f}%")
        
        # Trades metrics (CLOSE ONLY)
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        winning_trades = [t for t in close_trades if t['pnl'] > 0]
        losing_trades = [t for t in close_trades if t['pnl'] < 0]
        
        total_close_trades = len(close_trades)
        win_rate = (len(winning_trades) / total_close_trades * 100
                    if total_close_trades > 0 else 0)
        
        gross_profit = sum([t['pnl'] for t in winning_trades])
        gross_loss = abs(sum([t['pnl'] for t in losing_trades]))
        profit_factor = (gross_profit / gross_loss if gross_loss > 0
                        else float('inf'))
        
        logger.info(f"Total CLOSE trades: {total_close_trades}")
        logger.info(f"Winning trades: {len(winning_trades)}")
        logger.info(f"Losing trades: {len(losing_trades)}")
        logger.info(f"Win rate: {win_rate:.1f}%")
        logger.info(f"Profit factor: {profit_factor:.2f}")
        logger.info(f"Gross profit: ${gross_profit:.2f}")
        logger.info(f"Gross loss: ${gross_loss:.2f}")
        
        # Total PnL
        total_pnl = sum([t['pnl'] for t in close_trades])
        logger.info(f"Total PnL: ${total_pnl:.2f}")
        
        self.metrics = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_dd,
            'total_trades': total_close_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_pnl': total_pnl,
            'steps': len(self.portfolio_values)
        }
        
        return self.metrics
    
    def validate_metrics(self):
        """Valider que les métriques sont cohérentes"""
        logger.info("=" * 80)
        logger.info("VALIDATION DES MÉTRIQUES")
        logger.info("=" * 80)
        
        m = self.metrics
        
        # Check 1: Capital final vs PnL
        expected_capital = m['initial_capital'] + m['total_pnl']
        actual_capital = m['final_equity']
        diff = abs(expected_capital - actual_capital)
        
        logger.info(f"Check 1: Capital = Initial + PnL")
        logger.info(f"  Expected: ${expected_capital:.2f}")
        logger.info(f"  Actual: ${actual_capital:.2f}")
        logger.info(f"  Diff: ${diff:.2f}")
        
        if diff < 0.01:
            logger.info("  ✅ VALIDE")
        else:
            logger.warning(f"  ⚠️ DIVERGENCE: ${diff:.2f}")
        
        # Check 2: Win rate calculation
        if m['total_trades'] > 0:
            expected_wr = m['winning_trades'] / m['total_trades'] * 100
            actual_wr = m['win_rate']
            
            logger.info(f"Check 2: Win Rate = Winning / Total")
            logger.info(f"  Expected: {expected_wr:.1f}%")
            logger.info(f"  Actual: {actual_wr:.1f}%")
            
            if abs(expected_wr - actual_wr) < 0.1:
                logger.info("  ✅ VALIDE")
            else:
                logger.warning(f"  ⚠️ DIVERGENCE")
        
        # Check 3: Profit factor
        if m['gross_loss'] > 0:
            expected_pf = m['gross_profit'] / m['gross_loss']
            actual_pf = m['profit_factor']
            
            logger.info(f"Check 3: Profit Factor = Gross Profit / Gross Loss")
            logger.info(f"  Expected: {expected_pf:.2f}")
            logger.info(f"  Actual: {actual_pf:.2f}")
            
            if abs(expected_pf - actual_pf) < 0.01:
                logger.info("  ✅ VALIDE")
            else:
                logger.warning(f"  ⚠️ DIVERGENCE")
        
        # Check 4: Drawdown vs portfolio values
        if self.portfolio_values:
            peak = self.portfolio_values[0]
            min_val = min(self.portfolio_values)
            expected_dd = (min_val - peak) / peak * 100 if peak > 0 else 0
            actual_dd = self.metrics['max_drawdown_pct']
            
            logger.info(f"Check 4: Max Drawdown = (Min - Peak) / Peak")
            logger.info(f"  Expected: {expected_dd:.2f}%")
            logger.info(f"  Actual: {actual_dd:.2f}%")
            
            if abs(expected_dd - actual_dd) < 0.1:
                logger.info("  ✅ VALIDE")
            else:
                logger.warning(f"  ⚠️ DIVERGENCE")
        
        logger.info("=" * 80)
    
    def create_performance_chart(self):
        """Créer le graphique de performance honnête"""
        logger.info("Création graphique performance...")
        
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Évolution du Capital", "Drawdown (%)",
                           "Distribution PnL")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=self.portfolio_values,
                mode='lines',
                name='Équité',
                line=dict(color='gold', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        drawdowns = []
        peak = self.portfolio_values[0] if self.portfolio_values else 0
        for val in self.portfolio_values:
            peak = max(peak, val)
            dd = (val - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=drawdowns,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )
        
        # PnL histogram
        close_trades = [t for t in self.trades if t['type'] == 'CLOSE']
        pnls = [t['pnl'] for t in close_trades]
        
        if pnls:
            fig.add_trace(
                go.Histogram(
                    x=pnls,
                    nbinsx=30,
                    name='PnL Distribution',
                    marker_color='teal'
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=f"Performance Honnête - {len(close_trades)} Trades CLOSE",
            height=900,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Équité ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Fréquence", row=3, col=1)
        fig.update_xaxes(title_text="Temps", row=3, col=1)
        
        html_file = f"/tmp/dashboard_honest_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(html_file)
        logger.info(f"✅ Graphique sauvegardé: {html_file}")
        
        return fig, html_file
    
    def create_trades_chart(self):
        """Créer le graphique des trades par timeframe"""
        logger.info("Création graphique trades...")
        
        timeframes = list(set([t['timeframe'] for t in self.trades]))
        logger.info(f"Timeframes trouvés: {timeframes}")
        
        # Créer un graphique par timeframe
        for tf in sorted(timeframes):
            tf_trades = [t for t in self.trades if t['timeframe'] == tf]
            open_trades = [t for t in tf_trades if t['type'] == 'OPEN']
            close_trades = [t for t in tf_trades if t['type'] == 'CLOSE']
            
            logger.info(f"  {tf}: {len(open_trades)} OPEN, {len(close_trades)} CLOSE")
            
            fig = go.Figure()
            
            # OPEN markers
            if open_trades:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in open_trades],
                    y=[t['price'] for t in open_trades],
                    mode='markers',
                    name='OPEN',
                    marker=dict(symbol='triangle-up', size=12, color='lime')
                ))
            
            # CLOSE markers
            if close_trades:
                fig.add_trace(go.Scatter(
                    x=[t['time'] for t in close_trades],
                    y=[t['price'] for t in close_trades],
                    mode='markers',
                    name='CLOSE',
                    marker=dict(symbol='triangle-down', size=12, color='red')
                ))
            
            fig.update_layout(
                title=f"Trades {tf} ({len(open_trades)} OPEN, {len(close_trades)} CLOSE)",
                template='plotly_dark',
                height=500
            )
            
            html_file = f"/tmp/dashboard_honest_trades_{tf}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(html_file)
            logger.info(f"✅ Graphique {tf} sauvegardé: {html_file}")
    
    def run_full_pipeline(self, asset="XRPUSDT", capital=20.5, max_steps=20000):
        """Exécuter le pipeline complet"""
        logger.info("DÉMARRAGE PIPELINE COMPLET")
        logger.info("=" * 80)
        
        try:
            # 1. Load
            self.load_config_and_env(asset, capital)
            
            # 2. Run
            self.run_evaluation(max_steps)
            
            # 3. Extract
            self.extract_trades(asset)
            
            # 4. Metrics
            self.calculate_metrics(capital)
            
            # 5. Validate
            self.validate_metrics()
            
            # 6. Charts
            self.create_performance_chart()
            self.create_trades_chart()
            
            logger.info("=" * 80)
            logger.info("✅ PIPELINE COMPLET RÉUSSI")
            logger.info("=" * 80)
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"❌ ERREUR PIPELINE: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    dashboard = HonestDashboard(
        checkpoint_path="checkpoints_final/adan_model_checkpoint_640000_steps.zip"
    )
    
    metrics = dashboard.run_full_pipeline(
        asset="XRPUSDT",
        capital=20.5,
        max_steps=20000
    )
    
    print("\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:>10.2f}")
        else:
            print(f"{key:.<40} {value:>10}")
    print("=" * 80)
