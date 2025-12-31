#!/usr/bin/env python3
"""
Script de Monitoring Détaillé - ADAN Trading Bot
Capture l'état complet du système toutes les 5 minutes
Pour confrontation avec le dashboard
"""

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional

class DetailedMonitor:
    def __init__(self, log_file: str = "monitor_validated.log"):
        self.log_file = Path(log_file)
        self.report_file = Path("monitoring_report_5min.log")
        self.history = deque(maxlen=288)  # 24h à 5min = 288 entrées
        self.start_time = datetime.now()
        
    def get_process_info(self) -> Optional[Dict]:
        """Récupère infos du processus monitor"""
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if 'paper_trading_monitor.py' in line and 'grep' not in line:
                    parts = line.split()
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'time': parts[9],
                        'status': 'RUNNING ✅'
                    }
            return {'status': 'STOPPED ❌'}
        except Exception as e:
            return {'status': f'ERROR: {e}'}
    
    def parse_last_iteration(self) -> Dict:
        """Parse la dernière itération complète des logs"""
        if not self.log_file.exists():
            return {}
        
        data = {
            'timestamp': None,
            'price': None,
            'rsi': None,
            'adx': None,
            'volatility': None,
            'regime': None,
            'workers': {},
            'consensus': None,
            'confidence': None,
            'positions': [],
            'dbe': {},
            'warmup': False,
            'errors': []
        }
        
        # Lire les 200 dernières lignes (suffisant pour 1 itération complète)
        try:
            with open(self.log_file, 'r') as f:
                lines = deque(f, maxlen=200)
        except Exception as e:
            data['errors'].append(f"Erreur lecture log: {e}")
            return data
        
        # Parser ligne par ligne
        for line in lines:
            # Timestamp
            if not data['timestamp']:
                match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if match:
                    data['timestamp'] = match.group(1)
            
            # Market Data
            if '📊 Market Data:' in line:
                price_match = re.search(r'Price=\$?([\d,]+\.?\d*)', line)
                rsi_match = re.search(r'RSI=([\d.]+)', line)
                adx_match = re.search(r'ADX=([\d.]+)', line)
                vol_match = re.search(r'Vol=([\d.]+)%', line)
                regime_match = re.search(r'Regime=(\w+\s?\w*)', line)
                
                if price_match:
                    data['price'] = float(price_match.group(1).replace(',', ''))
                if rsi_match:
                    data['rsi'] = float(rsi_match.group(1))
                if adx_match:
                    data['adx'] = float(adx_match.group(1))
                if vol_match:
                    data['volatility'] = float(vol_match.group(1))
                if regime_match:
                    data['regime'] = regime_match.group(1)
            
            # Workers votes
            if 'raw=' in line and 'w1:' not in line:  # Vote individuel
                worker_match = re.search(r'(w\d):\s*raw=([\d.]+).*→\s*(\w+),\s*conf=([\d.]+)', line)
                if worker_match:
                    worker_id = worker_match.group(1)
                    data['workers'][worker_id] = {
                        'raw': float(worker_match.group(2)),
                        'action': worker_match.group(3),
                        'confidence': float(worker_match.group(4))
                    }
            
            # Consensus
            if 'Consensus Final' in line or 'consensus=' in line:
                consensus_match = re.search(r'consensus[=:]\s*(\d+)', line)
                if consensus_match:
                    consensus_val = int(consensus_match.group(1))
                    data['consensus'] = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}.get(consensus_val, 'UNKNOWN')
            
            # DBE
            if 'DBE activé' in line or 'Régime détecté' in line:
                regime_match = re.search(r'Régime:\s*(\w+)', line)
                mult_match = re.search(r'multiplicateurs.*TP[:\s]*([\d.]+).*SL[:\s]*([\d.]+)', line, re.IGNORECASE)
                if regime_match:
                    data['dbe']['regime'] = regime_match.group(1)
                if mult_match:
                    data['dbe']['tp_mult'] = float(mult_match.group(1))
                    data['dbe']['sl_mult'] = float(mult_match.group(2))
            
            # Warmup
            if '🔄 Warmup' in line:
                data['warmup'] = True
            
            # Position info
            if 'Position ouverte' in line or 'BUY @' in line:
                pos_match = re.search(r'BUY @ ([\d.]+)', line)
                if pos_match:
                    data['positions'].append({
                        'side': 'BUY',
                        'entry': float(pos_match.group(1)),
                        'status': 'OPEN'
                    })
            
            # Erreurs
            if 'Shape mismatch' in line:
                data['errors'].append('Shape mismatch détecté')
            if 'SATURATION DETECTED' in line:
                data['errors'].append('Saturation worker détectée')
        
        return data
    
    def get_positions_from_dashboard(self) -> List[Dict]:
        """Extrait positions actives depuis les logs récents"""
        positions = []
        
        # Chercher les dernières infos de positions dans les logs
        if not self.log_file.exists():
            return positions
        
        try:
            with open(self.log_file, 'r') as f:
                lines = deque(f, maxlen=500)
            
            for line in lines:
                # Position ouverte avec TP/SL
                if 'TP=' in line and 'SL=' in line:
                    tp_match = re.search(r'TP:\s*([\d.]+)', line)
                    sl_match = re.search(r'SL:\s*([\d.]+)', line)
                    entry_match = re.search(r'Entry:\s*([\d.]+)', line)
                    
                    if tp_match and sl_match:
                        positions.append({
                            'tp': float(tp_match.group(1)),
                            'sl': float(sl_match.group(1)),
                            'entry': float(entry_match.group(1)) if entry_match else None
                        })
        except Exception:
            pass
        
        return positions
    
    def calculate_metrics(self, current_data: Dict, prev_data: Optional[Dict]) -> Dict:
        """Calcule métriques dérivées"""
        metrics = {}
        
        # Delta prix
        if prev_data and current_data.get('price') and prev_data.get('price'):
            delta = current_data['price'] - prev_data['price']
            delta_pct = (delta / prev_data['price']) * 100
            metrics['price_delta'] = delta
            metrics['price_delta_pct'] = delta_pct
        
        # Distance au TP/SL
        if current_data.get('price'):
            positions = self.get_positions_from_dashboard()
            if positions:
                pos = positions[0]
                price = current_data['price']
                
                if pos.get('tp'):
                    dist_tp = pos['tp'] - price
                    dist_tp_pct = (dist_tp / price) * 100
                    metrics['distance_tp'] = dist_tp
                    metrics['distance_tp_pct'] = dist_tp_pct
                
                if pos.get('sl'):
                    dist_sl = price - pos['sl']
                    dist_sl_pct = (dist_sl / price) * 100
                    metrics['distance_sl'] = dist_sl
                    metrics['distance_sl_pct'] = dist_sl_pct
        
        return metrics
    
    def format_report(self, data: Dict, metrics: Dict, process_info: Dict) -> str:
        """Formate le rapport complet"""
        timestamp = data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        report = f"""
{'='*100}
🕐 RAPPORT MONITORING - {timestamp}
{'='*100}

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🖥️  ÉTAT SYSTÈME                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ PID          : {process_info.get('pid', 'N/A'):<20} Status: {process_info.get('status', 'UNKNOWN')}
│ CPU          : {process_info.get('cpu', 'N/A'):<20} Mem   : {process_info.get('mem', 'N/A')}%
│ Uptime       : {process_info.get('time', 'N/A'):<20} Depuis: {self.start_time.strftime('%H:%M:%S')}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 📈 MARCHÉ & INDICATEURS                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Prix         : ${data.get('price', 0):,.2f}
│ Delta 5min   : {metrics.get('price_delta', 0):+.2f} USD ({metrics.get('price_delta_pct', 0):+.3f}%)
│ RSI          : {data.get('rsi', 0):.2f}
│ ADX          : {data.get('adx', 0):.2f}
│ Volatilité   : {data.get('volatility', 0):.2f}%
│ Régime       : {data.get('regime', 'UNKNOWN')}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🤖 WORKERS & DÉCISIONS                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
"""
        
        # Workers
        for wid in ['w1', 'w2', 'w3', 'w4']:
            if wid in data.get('workers', {}):
                w = data['workers'][wid]
                report += f"│ {wid.upper():4} : raw={w['raw']:.4f} → {w['action']:4} | conf={w['confidence']:.3f}\n"
            else:
                report += f"│ {wid.upper():4} : N/A\n"
        
        report += f"""│
│ Consensus    : {data.get('consensus', 'N/A')}
│ Saturation   : {'❌ Détectée' if 'Saturation' in str(data.get('errors', [])) else '✅ Aucune'}
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🎯 DBE (DYNAMIC BEHAVIOR ENGINE)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Régime DBE   : {data.get('dbe', {}).get('regime', 'SIDEWAYS')}
│ Mult. TP     : {data.get('dbe', {}).get('tp_mult', 1.0):.2f}x
│ Mult. SL     : {data.get('dbe', {}).get('sl_mult', 1.0):.2f}x
│ Impact       : TP base 3% × {data.get('dbe', {}).get('tp_mult', 1.0):.1f} = {3 * data.get('dbe', {}).get('tp_mult', 1.0):.1f}%
│              : SL base 2% × {data.get('dbe', {}).get('sl_mult', 1.0):.1f} = {2 * data.get('dbe', {}).get('sl_mult', 1.0):.1f}%
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 💼 POSITIONS ACTIVES                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
"""
        
        positions = self.get_positions_from_dashboard()
        if positions:
            pos = positions[0]
            current_price = data.get('price', 0)
            
            report += f"""│ Nombre       : {len(positions)}
│ Entry        : ${pos.get('entry', 0):,.2f}
│ Current      : ${current_price:,.2f}
│ TP Target    : ${pos.get('tp', 0):,.2f} (dist: {metrics.get('distance_tp', 0):+.2f} USD / {metrics.get('distance_tp_pct', 0):+.2f}%)
│ SL Target    : ${pos.get('sl', 0):,.2f} (dist: {metrics.get('distance_sl', 0):+.2f} USD / {metrics.get('distance_sl_pct', 0):+.2f}%)
│ P&L Non-réal : {((current_price - pos.get('entry', current_price)) / pos.get('entry', 1)) * 100:+.3f}%
"""
        else:
            report += "│ Aucune position ouverte\n"
        
        report += """└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 🔧 DONNÉES TECHNIQUES                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
"""
        
        report += f"""│ Warmup 4h    : {'🔄 Actif' if data.get('warmup') else '✅ Désactivé'}
│ Shape Match  : {'❌ Erreur' if 'Shape mismatch' in str(data.get('errors', [])) else '✅ OK'}
│ Erreurs      : {len(data.get('errors', []))} détectées
"""
        
        if data.get('errors'):
            for err in data['errors']:
                report += f"│              - {err}\n"
        
        report += """└─────────────────────────────────────────────────────────────────────────────┘

"""
        return report
    
    def run_monitoring_cycle(self):
        """Exécute un cycle de monitoring"""
        print(f"🔍 Démarrage monitoring détaillé - {datetime.now().strftime('%H:%M:%S')}")
        print(f"📝 Rapport: {self.report_file}")
        print(f"🔁 Fréquence: 5 minutes")
        print(f"⏱️  Durée: 24h (288 captures)")
        print("\n" + "="*100)
        
        cycle = 0
        prev_data = None
        
        try:
            while True:
                cycle += 1
                
                # Récupérer données
                process_info = self.get_process_info()
                current_data = self.parse_last_iteration()
                metrics = self.calculate_metrics(current_data, prev_data)
                
                # Générer rapport
                report = self.format_report(current_data, metrics, process_info)
                
                # Afficher à l'écran
                print(report)
                
                # Sauvegarder dans fichier
                with open(self.report_file, 'a') as f:
                    f.write(report)
                
                # Historique
                self.history.append({
                    'timestamp': current_data.get('timestamp'),
                    'price': current_data.get('price'),
                    'workers': current_data.get('workers'),
                    'consensus': current_data.get('consensus')
                })
                
                # Stats rapides
                print(f"📊 Cycle #{cycle} | Captures totales: {len(self.history)}/288")
                
                if process_info.get('status') == 'STOPPED ❌':
                    print("\n⚠️  ALERTE: Monitor arrêté détecté!")
                    print("Attente de redémarrage ou arrêt manuel...")
                
                prev_data = current_data
                
                # Attendre 5 minutes
                next_time = datetime.fromtimestamp((datetime.now().timestamp() + 300))
                print(f"\n⏳ Prochain rapport dans 5 minutes ({datetime.now().strftime('%H:%M:%S')} → {next_time.strftime('%H:%M:%S')})...")
                print("="*100 + "\n")
                
                time.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring arrêté manuellement")
            print(f"📊 Total captures: {len(self.history)}")
            print(f"📝 Rapport complet: {self.report_file}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                   ADAN MONITORING DÉTAILLÉ 5 MINUTES                     ║
║                  Surveillance Complète 24h du Bot de Trading             ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    monitor = DetailedMonitor()
    monitor.run_monitoring_cycle()

if __name__ == "__main__":
    main()
