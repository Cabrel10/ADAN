#!/usr/bin/env python3
"""Analyse les erreurs et anomalies dans les logs d'entraînement ADAN"""

import re
import sys
from pathlib import Path
from collections import defaultdict

LOG_FILE = "/mnt/new_data/adan_logs/training_1765061104.log"

def analyze_logs():
    print("🔍 ANALYSE DES LOGS D'ENTRAÎNEMENT ADAN")
    print("=" * 60)
    
    if not Path(LOG_FILE).exists():
        print(f"❌ Fichier log non trouvé: {LOG_FILE}")
        return
    
    # Statistiques
    stats = {
        'total_lines': 0,
        'errors': [],
        'warnings': [],
        'force_trade_caps': 0,
        'workers': defaultdict(int),
        'steps': [],
        'pnl_values': [],
    }
    
    try:
        with open(LOG_FILE, 'r', errors='ignore') as f:
            for line in f:
                stats['total_lines'] += 1
                
                # Erreurs
                if 'ERROR' in line or 'Exception' in line:
                    stats['errors'].append(line.strip()[:100])
                
                # Warnings
                if 'WARNING' in line:
                    stats['warnings'].append(line.strip()[:100])
                
                # Force trade cap
                if 'FORCE_TRADE_CAP' in line:
                    stats['force_trade_caps'] += 1
                
                # Workers
                for w in ['w0', 'w1', 'w2', 'w3']:
                    if f'Worker {w[-1]}' in line or f'Worker={w}' in line:
                        stats['workers'][w] += 1
                
                # Steps
                if '[STEP' in line:
                    match = re.search(r'\[STEP (\d+)\]', line)
                    if match:
                        stats['steps'].append(int(match.group(1)))
                
                # PnL
                if 'pnl' in line.lower():
                    match = re.search(r'pnl[:\s]+([0-9.-]+)', line, re.IGNORECASE)
                    if match:
                        try:
                            stats['pnl_values'].append(float(match.group(1)))
                        except:
                            pass
    
    except Exception as e:
        print(f"❌ Erreur lors de la lecture: {e}")
        return
    
    # Affichage des résultats
    print(f"\n📊 STATISTIQUES GÉNÉRALES")
    print(f"  Total lignes: {stats['total_lines']:,}")
    print(f"  Erreurs: {len(stats['errors'])}")
    print(f"  Warnings: {len(stats['warnings'])}")
    print(f"  Force Trade Caps: {stats['force_trade_caps']}")
    
    print(f"\n👷 ACTIVITÉ PAR WORKER")
    for w in ['w0', 'w1', 'w2', 'w3']:
        print(f"  {w}: {stats['workers'][w]:,} lignes")
    
    if stats['steps']:
        print(f"\n📈 PROGRESSION DES STEPS")
        print(f"  Min: {min(stats['steps'])}")
        print(f"  Max: {max(stats['steps'])}")
        print(f"  Moyenne: {sum(stats['steps'])/len(stats['steps']):.0f}")
    
    if stats['pnl_values']:
        print(f"\n💰 PnL")
        print(f"  Min: {min(stats['pnl_values']):.4f}")
        print(f"  Max: {max(stats['pnl_values']):.4f}")
        print(f"  Moyenne: {sum(stats['pnl_values'])/len(stats['pnl_values']):.4f}")
    
    if stats['errors']:
        print(f"\n❌ ERREURS DÉTECTÉES")
        for err in stats['errors'][:5]:
            print(f"  - {err}")
    
    if stats['warnings']:
        print(f"\n⚠️  WARNINGS DÉTECTÉS")
        for warn in stats['warnings'][:5]:
            print(f"  - {warn}")
    
    print(f"\n✅ Entraînement en cours - {stats['total_lines']:,} lignes collectées")

if __name__ == '__main__':
    analyze_logs()
