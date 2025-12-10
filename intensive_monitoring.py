#!/usr/bin/env python3
"""
🔍 MONITORING INTENSIF - DÉTECTION D'INSTABILITÉ EN TEMPS RÉEL
Surveille les signaux d'alerte critiques pour valider l'hypothèse
"""

import re
import time
from pathlib import Path
from datetime import datetime
from collections import deque

log_file = Path("/mnt/new_data/adan_logs/training_20251207_232821.log")

# Historique pour détection de tendances
metrics = {
    'nan_count': 0,
    'error_count': 0,
    'step_count': 0,
    'portfolio_values': deque(maxlen=50),
    'last_check_time': time.time(),
    'start_time': time.time()
}

ALERT_THRESHOLDS = {
    'gradient_clip_frequency': 0.3,  # Si > 30% des gradients sont clippés
    'loss_spike': 2.0,  # Si loss augmente de 2x
    'nan_detection': 1,  # Dès le premier NaN
}

print("=" * 80)
print("🔍 MONITORING INTENSIF - VALIDATION DE L'HYPOTHÈSE")
print("=" * 80)
print(f"Log: {log_file}")
print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

check_count = 0

while True:
    try:
        check_count += 1
        current_time = time.time()
        elapsed = current_time - metrics['start_time']
        
        # Lire les dernières lignes
        with open(log_file, 'r') as f:
            lines = f.readlines()[-1000:]
        
        # Détection des NaN
        nan_lines = [l for l in lines if 'nan' in l.lower() or 'invalid values' in l.lower()]
        if len(nan_lines) > metrics['nan_count']:
            print(f"⚠️  ALERTE NaN DÉTECTÉ!")
            print(f"   Nombre: {len(nan_lines)} (nouveau: {len(nan_lines) - metrics['nan_count']})")
            print(f"   Temps écoulé: {elapsed:.1f}s")
            metrics['nan_count'] = len(nan_lines)
            print()
        
        # Détection des erreurs
        error_lines = [l for l in lines if 'ERROR' in l or 'Exception' in l]
        if len(error_lines) > metrics['error_count']:
            print(f"❌ ERREUR DÉTECTÉE!")
            for line in error_lines[-2:]:
                print(f"   {line.strip()[:100]}")
            metrics['error_count'] = len(error_lines)
            print()
        
        # Extraction des steps
        step_lines = [l for l in lines if '[STEP' in l and 'Portfolio value:' in l]
        if step_lines:
            last_step_line = step_lines[-1]
            match = re.search(r'\[STEP (\d+)', last_step_line)
            if match:
                step = int(match.group(1))
                metrics['step_count'] = step
                
                # Extraction du portfolio
                port_match = re.search(r'Portfolio value: ([\d.]+)', last_step_line)
                if port_match:
                    portfolio = float(port_match.group(1))
                    metrics['portfolio_values'].append(portfolio)
        
        # Affichage du statut
        print(f"[{check_count}] ⏱️  Temps: {elapsed:.1f}s | 📈 Step: {metrics['step_count']} | 💰 Portfolio: {metrics['portfolio_values'][-1]:.2f if metrics['portfolio_values'] else 'N/A'}")
        
        # Vérification de la stabilité
        if metrics['portfolio_values']:
            recent_portfolios = list(metrics['portfolio_values'])[-10:]
            if len(recent_portfolios) > 5:
                avg = sum(recent_portfolios) / len(recent_portfolios)
                volatility = max(recent_portfolios) - min(recent_portfolios)
                print(f"   📊 Moyenne: ${avg:.2f} | Volatilité: ${volatility:.2f}")
        
        # Vérification des processus
        import subprocess
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
            process_count = len([l for l in result.stdout.split('\n') if 'train_parallel_agents.py' in l and 'grep' not in l])
            print(f"   🤖 Processus: {process_count}")
        except:
            pass
        
        print()
        
        # Pause avant la prochaine vérification
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring arrêté par l'utilisateur")
        break
    except Exception as e:
        print(f"❌ Erreur monitoring: {e}")
        time.sleep(10)

print("\n" + "=" * 80)
print("📊 RÉSUMÉ FINAL")
print("=" * 80)
print(f"Durée totale: {(time.time() - metrics['start_time']):.1f}s")
print(f"Steps atteints: {metrics['step_count']}")
print(f"NaN détectés: {metrics['nan_count']}")
print(f"Erreurs détectées: {metrics['error_count']}")
print(f"Vérifications: {check_count}")
print("=" * 80)
