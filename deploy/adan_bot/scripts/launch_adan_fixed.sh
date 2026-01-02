#!/bin/bash
# Script de lancement ADAN avec données préchargées
# Résout définitivement le problème de données insuffisantes

echo "🚀 LANCEMENT ADAN AVEC DONNÉES PRÉCHARGÉES"
echo "=========================================="

# 1. Vérifier que les données sont présentes
echo "🔍 Vérification des données..."
if [ ! -d "historical_data" ] || [ ! -f "historical_data/BTC_USDT_5m_data.csv" ]; then
    echo "❌ Données manquantes, téléchargement..."
    python scripts/quick_data_fix.py
    if [ $? -ne 0 ]; then
        echo "❌ Échec du téléchargement des données"
        exit 1
    fi
fi

echo "✅ Données présentes"

# 2. Arrêter les anciens processus
echo "🛑 Arrêt des anciens processus..."
pkill -f "paper_trading_monitor.py" 2>/dev/null
sleep 2

# 3. Nettoyer les anciens logs
echo "🧹 Nettoyage des logs..."
> paper_trading.log

# 4. Lancer le monitor en arrière-plan
echo "🤖 Lancement du monitor ADAN..."
nohup python scripts/paper_trading_monitor.py > monitor_output.log 2>&1 &
MONITOR_PID=$!
echo "   Monitor PID: $MONITOR_PID"

# 5. Attendre que le monitor s'initialise
echo "⏳ Initialisation du monitor..."
sleep 5

# 6. Vérifier que le monitor fonctionne
if ps -p $MONITOR_PID > /dev/null; then
    echo "✅ Monitor actif"
else
    echo "❌ Monitor arrêté, vérifiez monitor_output.log"
    exit 1
fi

# 7. Lancer le dashboard
echo "📊 Lancement du dashboard..."
echo ""
echo "🎯 SYSTÈME ADAN OPÉRATIONNEL"
echo "=============================="
echo "Monitor PID: $MONITOR_PID"
echo "Logs monitor: tail -f monitor_output.log"
echo "Logs trading: tail -f paper_trading.log"
echo ""
echo "🔄 Dashboard en cours de lancement..."
echo "Appuyez sur Ctrl+C pour arrêter le dashboard"
echo ""

# Lancer le dashboard (bloquant)
python scripts/simple_dashboard.py

# 8. Nettoyage à l'arrêt
echo ""
echo "🛑 Arrêt du système..."
kill $MONITOR_PID 2>/dev/null
echo "✅ Système arrêté"