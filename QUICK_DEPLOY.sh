#!/bin/bash

# 🚀 QUICK DEPLOY - COVARIATE SHIFT FIX
# Déploie la correction du covariate shift en 3 étapes

set -e

echo "🚀 DÉPLOIEMENT RAPIDE - COVARIATE SHIFT FIX"
echo "=========================================="

# Étape 1: Arrêter l'ancien monitor
echo ""
echo "📍 Étape 1: Arrêter l'ancien monitor..."
pkill -f paper_trading_monitor.py || true
sleep 2
echo "✅ Ancien monitor arrêté"

# Étape 2: Vérifier la syntaxe
echo ""
echo "📍 Étape 2: Vérifier la syntaxe..."
python -m py_compile scripts/paper_trading_monitor.py
echo "✅ Syntaxe OK"

# Étape 3: Tester les imports
echo ""
echo "📍 Étape 3: Tester les imports..."
python << 'EOF'
try:
    from adan_trading_bot.normalization import ObservationNormalizer, DriftDetector
    normalizer = ObservationNormalizer()
    detector = DriftDetector()
    print("✅ Imports OK")
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)
EOF

# Étape 4: Redémarrer le monitor
echo ""
echo "📍 Étape 4: Redémarrer le monitor..."
python scripts/paper_trading_monitor.py \
  --api_key "JZELi7qLcOcp5gr7AAYpnlJnW9wxbHHeX99uqFWNFxJIKKb6pVhrmYu2mboWMFeA" \
  --api_secret "dFem0rr6ItWQ65sUxMRHseAUI8dtYDMI7WB69SrWYT4td5VKdjqFmilwb89cw4zY" &

MONITOR_PID=$!
echo "✅ Monitor lancé (PID: $MONITOR_PID)"

# Étape 5: Attendre et vérifier
echo ""
echo "📍 Étape 5: Vérifier les logs..."
sleep 5

echo ""
echo "📊 LOGS RÉCENTS:"
echo "==============="
tail -20 paper_trading.log | grep -E "(Normalisation|Dérive|signal|✅|❌)" || echo "Pas de logs de normalisation encore"

echo ""
echo "✅ DÉPLOIEMENT COMPLET!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Vérifier que le normaliseur est initialisé: grep 'Normaliseur initialisé' paper_trading.log"
echo "2. Vérifier que les signaux varient: grep 'Ensemble Decision' paper_trading.log"
echo "3. Monitorer les logs: tail -f paper_trading.log"
echo ""
echo "🎯 Attendez 2-3 cycles (2-3 minutes) pour voir les premiers signaux variés"
