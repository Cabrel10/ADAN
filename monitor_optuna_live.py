#!/usr/bin/env python3
"""
Monitoring en temps réel de l'optimisation Optuna
Affiche les trials complétés et les métriques
"""
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def parse_trial_line(line: str) -> dict:
    """Parse une ligne de trial Optuna"""
    try:
        # Format: Trial 5: Score=45.23, Sharpe=2.34, DD=18.0%, WR=58.0%, Trades=145
        if 'Trial' not in line or 'Score=' not in line:
            return None
        
        trial_match = re.search(r'Trial (\d+):', line)
        score_match = re.search(r'Score=([\d.-]+)', line)
        sharpe_match = re.search(r'Sharpe=([\d.-]+)', line)
        dd_match = re.search(r'DD=([\d.]+)%', line)
        wr_match = re.search(r'WR=([\d.]+)%', line)
        trades_match = re.search(r'Trades=(\d+)', line)
        
        if not all([trial_match, score_match, sharpe_match, dd_match, wr_match, trades_match]):
            return None
        
        return {
            'trial': int(trial_match.group(1)),
            'score': float(score_match.group(1)),
            'sharpe': float(sharpe_match.group(1)),
            'dd': float(dd_match.group(1)),
            'wr': float(wr_match.group(1)),
            'trades': int(trades_match.group(1)),
        }
    except:
        return None

def monitor_live(worker: str, check_interval: int = 5):
    """Monitore en temps réel"""
    
    log_file = Path(f"optuna_results/{worker}_optimization.log")
    
    print(f"\n{'='*100}")
    print(f"🔍 MONITORING EN TEMPS RÉEL - {worker}")
    print(f"{'='*100}")
    print(f"Fichier log : {log_file}")
    print(f"Intervalle de vérification : {check_interval}s")
    print(f"Démarrage : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    trials_seen = set()
    best_score = -999
    best_trial = None
    trial_history = []
    
    while True:
        if not log_file.exists():
            print(f"⏳ En attente du fichier log...")
            time.sleep(check_interval)
            continue
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Parser les trials
            for line in lines:
                trial_data = parse_trial_line(line)
                if trial_data and trial_data['trial'] not in trials_seen:
                    trials_seen.add(trial_data['trial'])
                    trial_history.append(trial_data)
                    
                    # Afficher le trial
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Trial {trial_data['trial']:2d} | "
                          f"Score: {trial_data['score']:7.2f} | "
                          f"Sharpe: {trial_data['sharpe']:6.2f} | "
                          f"DD: {trial_data['dd']:5.1f}% | "
                          f"WR: {trial_data['wr']:5.1f}% | "
                          f"Trades: {trial_data['trades']:3d}")
                    
                    # Mettre à jour le meilleur
                    if trial_data['score'] > best_score:
                        best_score = trial_data['score']
                        best_trial = trial_data
                        print(f"  ⭐ NOUVEAU MEILLEUR SCORE: {best_score:.2f}")
            
            # Afficher le résumé
            if trial_history:
                print(f"\n{'─'*100}")
                print(f"📊 RÉSUMÉ ACTUEL ({len(trial_history)} trials)")
                print(f"{'─'*100}")
                
                scores = [t['score'] for t in trial_history]
                sharpes = [t['sharpe'] for t in trial_history]
                dds = [t['dd'] for t in trial_history]
                wrs = [t['wr'] for t in trial_history]
                
                print(f"Score      : Min={min(scores):7.2f} | Max={max(scores):7.2f} | Avg={sum(scores)/len(scores):7.2f}")
                print(f"Sharpe     : Min={min(sharpes):6.2f} | Max={max(sharpes):6.2f} | Avg={sum(sharpes)/len(sharpes):6.2f}")
                print(f"Drawdown   : Min={min(dds):5.1f}% | Max={max(dds):5.1f}% | Avg={sum(dds)/len(dds):5.1f}%")
                print(f"Win Rate   : Min={min(wrs):5.1f}% | Max={max(wrs):5.1f}% | Avg={sum(wrs)/len(wrs):5.1f}%")
                
                if best_trial:
                    print(f"\n🏆 MEILLEUR TRIAL")
                    print(f"   Trial {best_trial['trial']}: Score={best_trial['score']:.2f}, "
                          f"Sharpe={best_trial['sharpe']:.2f}, DD={best_trial['dd']:.1f}%, "
                          f"WR={best_trial['wr']:.1f}%, Trades={best_trial['trades']}")
                
                # Vérifier si terminé
                if 'PHASE 2 COMPLETE' in ''.join(lines):
                    print(f"\n✅ OPTIMISATION TERMINÉE")
                    break
                
                print(f"{'─'*100}\n")
        
        except Exception as e:
            print(f"⚠️  Erreur : {e}")
        
        time.sleep(check_interval)

if __name__ == "__main__":
    worker = sys.argv[1] if len(sys.argv) > 1 else "W1"
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    try:
        monitor_live(worker, interval)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring arrêté par l'utilisateur")
