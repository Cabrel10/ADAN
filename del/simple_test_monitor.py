#!/usr/bin/env python3
"""
Test simple du monitor avec données préchargées
Version minimale pour vérifier que les données sont correctement chargées
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDataTester:
    """Test simple des données préchargées"""
    
    def __init__(self):
        self.data_dir = Path("historical_data")
        self.data = {}
    
    def load_and_test_data(self):
        """Charge et teste les données"""
        logger.info("🔍 TEST DES DONNÉES PRÉCHARGÉES")
        logger.info("="*40)
        
        timeframes = ['5m', '1h', '4h']
        all_good = True
        
        for tf in timeframes:
            file_path = self.data_dir / f"BTC_USDT_{tf}_data.csv"
            
            if not file_path.exists():
                logger.error(f"❌ {tf}: Fichier manquant")
                all_good = False
                continue
            
            try:
                # Charger les données
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                self.data[tf] = df
                
                # Tests de base
                logger.info(f"\n📊 {tf.upper()}:")
                logger.info(f"   Périodes: {len(df)}")
                logger.info(f"   Colonnes: {list(df.columns)}")
                logger.info(f"   Période: {df.index[0]} → {df.index[-1]}")
                
                # Vérifier les indicateurs
                required_indicators = ['rsi', 'adx', 'atr', 'volatility']
                missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
                
                if missing_indicators:
                    logger.warning(f"   ⚠️  Indicateurs manquants: {missing_indicators}")
                else:
                    logger.info(f"   ✅ Tous les indicateurs présents")
                
                # Vérifier les valeurs récentes
                latest = df.iloc[-1]
                logger.info(f"   💰 Prix actuel: ${latest['close']:.2f}")
                
                if 'rsi' in df.columns:
                    rsi_val = latest['rsi']
                    if pd.isna(rsi_val):
                        logger.warning(f"   ⚠️  RSI: NaN")
                    else:
                        logger.info(f"   📈 RSI: {rsi_val:.1f}")
                
                if 'adx' in df.columns:
                    adx_val = latest['adx']
                    if pd.isna(adx_val):
                        logger.warning(f"   ⚠️  ADX: NaN")
                    else:
                        logger.info(f"   📊 ADX: {adx_val:.1f}")
                
                if 'volatility' in df.columns:
                    vol_val = latest['volatility']
                    if pd.isna(vol_val):
                        logger.warning(f"   ⚠️  Volatilité: NaN")
                    else:
                        logger.info(f"   📉 Volatilité: {vol_val*100:.2f}%")
                
                # Test de suffisance des données
                if len(df) < 28:
                    logger.error(f"   ❌ Données insuffisantes: {len(df)} < 28")
                    all_good = False
                else:
                    logger.info(f"   ✅ Données suffisantes: {len(df)} ≥ 28")
                
            except Exception as e:
                logger.error(f"❌ {tf}: Erreur chargement - {e}")
                all_good = False
        
        return all_good
    
    def test_indicator_calculation(self):
        """Test le calcul des indicateurs sur une fenêtre"""
        logger.info("\n🧮 TEST CALCUL INDICATEURS")
        logger.info("="*30)
        
        for tf, df in self.data.items():
            if len(df) < 28:
                continue
                
            logger.info(f"\n{tf.upper()}:")
            
            # Test fenêtre glissante
            window_size = {'5m': 20, '1h': 10, '4h': 5}[tf]
            window_data = df.tail(window_size)
            
            logger.info(f"   Fenêtre: {len(window_data)} périodes")
            logger.info(f"   Prix: ${window_data['close'].iloc[0]:.2f} → ${window_data['close'].iloc[-1]:.2f}")
            
            # Calculer variation
            price_change = (window_data['close'].iloc[-1] / window_data['close'].iloc[0] - 1) * 100
            logger.info(f"   Variation: {price_change:+.2f}%")
            
            # Vérifier RSI
            if 'rsi' in window_data.columns:
                rsi_values = window_data['rsi'].dropna()
                if len(rsi_values) > 0:
                    logger.info(f"   RSI: {rsi_values.iloc[0]:.1f} → {rsi_values.iloc[-1]:.1f}")
                    
                    # Interpréter RSI
                    latest_rsi = rsi_values.iloc[-1]
                    if latest_rsi > 70:
                        logger.info(f"   📊 RSI: Surachat ({latest_rsi:.1f})")
                    elif latest_rsi < 30:
                        logger.info(f"   📊 RSI: Survente ({latest_rsi:.1f})")
                    else:
                        logger.info(f"   📊 RSI: Neutre ({latest_rsi:.1f})")
    
    def simulate_trading_decision(self):
        """Simule une décision de trading basée sur les données"""
        logger.info("\n🎯 SIMULATION DÉCISION TRADING")
        logger.info("="*35)
        
        if not self.data:
            logger.error("❌ Aucune donnée disponible")
            return
        
        # Analyser chaque timeframe
        signals = {}
        
        for tf, df in self.data.items():
            if len(df) < 5:
                continue
            
            latest = df.iloc[-1]
            
            # Signal basé sur RSI
            rsi_signal = 0
            if 'rsi' in df.columns and not pd.isna(latest['rsi']):
                rsi = latest['rsi']
                if rsi > 70:
                    rsi_signal = -1  # SELL
                elif rsi < 30:
                    rsi_signal = 1   # BUY
            
            # Signal basé sur tendance (prix)
            price_signal = 0
            if len(df) >= 5:
                recent_prices = df['close'].tail(5)
                if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                    price_signal = 1  # Tendance haussière
                else:
                    price_signal = -1  # Tendance baissière
            
            signals[tf] = {
                'rsi_signal': rsi_signal,
                'price_signal': price_signal,
                'rsi_value': latest.get('rsi', None),
                'price': latest['close']
            }
            
            logger.info(f"{tf}: RSI={latest.get('rsi', 'N/A'):.1f}, Prix=${latest['close']:.2f}")
        
        # Décision d'ensemble
        total_signals = sum(s['rsi_signal'] + s['price_signal'] for s in signals.values())
        
        if total_signals > 2:
            decision = "🟢 BUY"
        elif total_signals < -2:
            decision = "🔴 SELL"
        else:
            decision = "⚪ HOLD"
        
        logger.info(f"\n🎯 DÉCISION ENSEMBLE: {decision}")
        logger.info(f"   Score total: {total_signals}")
        
        return decision

def main():
    """Test principal"""
    logger.info("🚀 DÉMARRAGE TEST DONNÉES PRÉCHARGÉES")
    
    tester = SimpleDataTester()
    
    # 1. Charger et tester les données
    if not tester.load_and_test_data():
        logger.error("❌ Problème avec les données")
        logger.info("💡 Exécutez: python scripts/quick_data_fix.py")
        return
    
    # 2. Tester le calcul des indicateurs
    tester.test_indicator_calculation()
    
    # 3. Simuler une décision de trading
    decision = tester.simulate_trading_decision()
    
    logger.info("\n" + "="*50)
    logger.info("✅ TEST TERMINÉ AVEC SUCCÈS")
    logger.info("="*50)
    logger.info("💡 Les données sont prêtes pour le trading!")
    logger.info("🚀 Vous pouvez maintenant lancer le monitor complet")

if __name__ == "__main__":
    main()