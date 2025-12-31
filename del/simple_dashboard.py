#!/usr/bin/env python3
"""
Dashboard simple pour ADAN avec données préchargées
Affiche les métriques en temps réel
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import os

class SimpleADANDashboard:
    """Dashboard simple pour ADAN"""
    
    def __init__(self):
        self.data_dir = Path("historical_data")
        self.log_file = Path("paper_trading.log")
        
    def clear_screen(self):
        """Efface l'écran"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def load_current_data(self):
        """Charge les données actuelles"""
        data = {}
        
        timeframes = ['5m', '1h', '4h']
        for tf in timeframes:
            file_path = self.data_dir / f"BTC_USDT_{tf}_data.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                data[tf] = df.iloc[-1]  # Dernière ligne
        
        return data
    
    def get_trading_status(self):
        """Récupère le statut du trading depuis les logs"""
        if not self.log_file.exists():
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Chercher les dernières informations
            last_balance = 29.0
            last_trades = 0
            last_pnl = 0.0
            last_confidence = 0.75
            last_signal = "HOLD"
            
            for line in reversed(lines[-100:]):  # 100 dernières lignes
                if "Balance: $" in line:
                    try:
                        last_balance = float(line.split("Balance: $")[1].split()[0])
                    except:
                        pass
                elif "PnL total:" in line:
                    try:
                        pnl_str = line.split("PnL total: ")[1].split("%")[0]
                        last_pnl = float(pnl_str)
                    except:
                        pass
                elif "Trades aujourd'hui:" in line:
                    try:
                        last_trades = int(line.split("Trades aujourd'hui: ")[1].split()[0])
                    except:
                        pass
                elif "Confiance globale:" in line:
                    try:
                        last_confidence = float(line.split("Confiance globale: ")[1].split()[0])
                    except:
                        pass
                elif "SIGNAL BUY" in line:
                    last_signal = "BUY"
                elif "SIGNAL SELL" in line:
                    last_signal = "SELL"
                elif "SIGNAL HOLD" in line:
                    last_signal = "HOLD"
            
            return {
                'balance': last_balance,
                'trades_today': last_trades,
                'pnl': last_pnl,
                'confidence': last_confidence,
                'last_signal': last_signal
            }
            
        except Exception as e:
            return None
    
    def display_header(self):
        """Affiche l'en-tête"""
        print("="*80)
        print("🚀 ADAN TRADING BOT - DASHBOARD TEMPS RÉEL")
        print("="*80)
        print(f"⏰ Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def display_market_data(self, data):
        """Affiche les données de marché"""
        print("📊 DONNÉES DE MARCHÉ")
        print("-" * 40)
        
        if not data:
            print("❌ Aucune donnée disponible")
            return
        
        for tf, row in data.items():
            price = row['close']
            rsi = row.get('rsi', 0)
            adx = row.get('adx', 0)
            vol = row.get('volatility', 0) * 100
            
            # Couleurs pour RSI
            if rsi > 70:
                rsi_status = "🔴 SURACHAT"
            elif rsi < 30:
                rsi_status = "🟢 SURVENTE"
            else:
                rsi_status = "⚪ NEUTRE"
            
            # Tendance ADX
            if adx > 50:
                trend_strength = "💪 FORTE"
            elif adx > 25:
                trend_strength = "📈 MODÉRÉE"
            else:
                trend_strength = "😴 FAIBLE"
            
            print(f"{tf.upper():>3}: ${price:>8.2f} | RSI: {rsi:>5.1f} {rsi_status} | ADX: {adx:>5.1f} {trend_strength} | Vol: {vol:>5.1f}%")
        
        print()
    
    def display_trading_status(self, status):
        """Affiche le statut du trading"""
        print("💰 STATUT TRADING")
        print("-" * 40)
        
        if not status:
            print("❌ Statut non disponible")
            return
        
        # Balance avec couleur
        balance = status['balance']
        pnl = status['pnl']
        
        if pnl > 0:
            pnl_color = "🟢"
        elif pnl < 0:
            pnl_color = "🔴"
        else:
            pnl_color = "⚪"
        
        print(f"Balance:     ${balance:>8.2f}")
        print(f"PnL Total:   {pnl_color} {pnl:>+6.2f}%")
        print(f"Trades/jour: {status['trades_today']:>8}")
        print(f"Confiance:   {status['confidence']:>8.2f}")
        
        # Signal actuel
        signal = status['last_signal']
        if signal == "BUY":
            signal_display = "🟢 BUY"
        elif signal == "SELL":
            signal_display = "🔴 SELL"
        else:
            signal_display = "⚪ HOLD"
        
        print(f"Signal:      {signal_display}")
        print()
    
    def display_system_health(self):
        """Affiche la santé du système"""
        print("🔧 SANTÉ DU SYSTÈME")
        print("-" * 40)
        
        # Vérifier les fichiers de données
        data_status = "✅ OK"
        for tf in ['5m', '1h', '4h']:
            file_path = self.data_dir / f"BTC_USDT_{tf}_data.csv"
            if not file_path.exists():
                data_status = "❌ MANQUANT"
                break
        
        print(f"Données:     {data_status}")
        
        # Vérifier les logs
        log_status = "✅ OK" if self.log_file.exists() else "❌ MANQUANT"
        print(f"Logs:        {log_status}")
        
        # Vérifier l'âge des données
        try:
            status_file = self.data_dir / "quick_load_status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                created_at = datetime.fromisoformat(status_data['created_at'])
                age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                
                if age_hours < 1:
                    age_status = f"✅ {age_hours:.1f}h"
                elif age_hours < 6:
                    age_status = f"⚠️  {age_hours:.1f}h"
                else:
                    age_status = f"❌ {age_hours:.1f}h"
                
                print(f"Âge données: {age_status}")
        except:
            print(f"Âge données: ❓ INCONNU")
        
        print()
    
    def display_instructions(self):
        """Affiche les instructions"""
        print("📋 COMMANDES")
        print("-" * 40)
        print("Ctrl+C : Arrêter le dashboard")
        print("Monitor: python scripts/working_monitor.py")
        print("Logs:    tail -f paper_trading.log")
        print()
    
    def run(self, refresh_interval=10):
        """Lance le dashboard"""
        print("🚀 Démarrage du dashboard ADAN...")
        
        try:
            while True:
                # Effacer l'écran
                self.clear_screen()
                
                # Afficher l'en-tête
                self.display_header()
                
                # Charger et afficher les données
                market_data = self.load_current_data()
                self.display_market_data(market_data)
                
                # Afficher le statut du trading
                trading_status = self.get_trading_status()
                self.display_trading_status(trading_status)
                
                # Afficher la santé du système
                self.display_system_health()
                
                # Afficher les instructions
                self.display_instructions()
                
                # Attendre avant la prochaine mise à jour
                print(f"🔄 Mise à jour dans {refresh_interval}s...")
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard arrêté")

def main():
    """Fonction principale"""
    dashboard = SimpleADANDashboard()
    dashboard.run(refresh_interval=10)

if __name__ == "__main__":
    main()