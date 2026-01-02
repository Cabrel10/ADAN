#!/usr/bin/env python3
"""
Script de diagnostic de santé des workers ADAN
Analyse les logs pour détecter les anomalies de comportement
"""

import re
from collections import Counter, defaultdict
from pathlib import Path

LOG_FILE = "config/logs/adan_trading_bot.log"

def analyze_logs():
    if not Path(LOG_FILE).exists():
        print(f"❌ Fichier de log introuvable: {LOG_FILE}")
        return

    print(f"🔍 Analyse du comportement des workers dans {LOG_FILE}...\n")

    worker_votes = defaultdict(list)
    market_data = []

    # Regex pour capturer les votes
    vote_pattern = re.compile(r"\s+(w[1-4]):\s+(HOLD|BUY|SELL)\s+\(conf=([0-9.]+)\)")
    # Regex pour capturer les données de marché
    market_pattern = re.compile(r"RSI=([0-9.]+).*ADX=([0-9.]+)")

    with open(LOG_FILE, 'r') as f:
        for line in f:
            # Capturer les votes
            vote_match = vote_pattern.search(line)
            if vote_match:
                worker, action, conf = vote_match.groups()
                worker_votes[worker].append({
                    'action': action,
                    'conf': float(conf)
                })

            # Capturer les données de marché
            market_match = market_pattern.search(line)
            if market_match:
                rsi, adx = market_match.groups()
                market_data.append({'rsi': float(rsi), 'adx': float(adx)})

    # --- ANALYSE DES DONNÉES DE MARCHÉ ---
    print("1️⃣  SANTÉ DES YEUX (Données de marché)")
    if not market_data:
        print("   ⚠️  Aucune donnée de marché trouvée.")
    else:
        last_rsi = market_data[-1]['rsi']
        unique_rsi = len(set(d['rsi'] for d in market_data))
        if unique_rsi == 1 and last_rsi == 50.0:
            print("   ❌ AVEUGLE : RSI bloqué à 50.00 (Valeur par défaut). Les indicateurs ne sont pas calculés.")
        elif unique_rsi < 3:
            print(f"   ⚠️  STAGNANT : Seulement {unique_rsi} valeurs de RSI différentes. Marché figé ?")
        else:
            print(f"   ✅ VIVANT : {unique_rsi} variations de RSI détectées. (Dernier: {last_rsi})")

    print("\n2️⃣  SANTÉ DES CERVEAUX (Comportement des Workers)")
    if not worker_votes:
        print("   ⚠️  Aucun vote détecté pour le moment.")
        return

    for worker in sorted(worker_votes.keys()):
        votes = worker_votes[worker]
        total = len(votes)
        counts = Counter(v['action'] for v in votes)
        avg_conf = sum(v['conf'] for v in votes) / total

        # Détection de frénésie
        switches = 0
        for i in range(1, len(votes)):
            if votes[i]['action'] != votes[i-1]['action']:
                switches += 1
        switch_rate = switches / total if total > 0 else 0

        print(f"   🤖 {worker.upper()} ({total} décisions):")
        print(f"      • Distribution : BUY={counts['BUY']} | SELL={counts['SELL']} | HOLD={counts['HOLD']}")
        print(f"      • Confiance Moy: {avg_conf:.1%}")

        # Diagnostic
        if counts['HOLD'] == total:
            print("      • Statut : 💤 DORT (100% HOLD)")
        elif switch_rate > 0.5:
            print(f"      • Statut : ⚡ FRÉNÉTIQUE (Change d'avis {switch_rate:.0%} du temps)")
        elif counts['BUY'] > 0 and counts['SELL'] > 0:
            print("      • Statut : ✅ ACTIF & ÉQUILIBRÉ")
        else:
            print("      • Statut : ⚠️  BIAISÉ (Un seul type d'action)")

    # --- ANALYSE DE GROUPE ---
    print("\n3️⃣  DIVERSITÉ DE L'ENSEMBLE")
    last_actions = [worker_votes[w][-1]['action'] for w in sorted(worker_votes) if worker_votes[w]]
    if len(set(last_actions)) == 1 and len(last_actions) == 4:
        print(f"   ⚠️  TROUPEAU : Tous les workers disent {last_actions[0]} en ce moment.")
    elif len(set(last_actions)) > 1:
        print(f"   ✅ DIVERSITÉ : Les avis divergent ({last_actions}). C'est bon pour l'ensemble.")

if __name__ == "__main__":
    analyze_logs()
