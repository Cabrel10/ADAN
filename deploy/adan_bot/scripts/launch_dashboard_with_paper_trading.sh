#!/bin/bash

# 🚀 LAUNCH DASHBOARD WITH PAPER TRADING INTEGRATION
# Démarre le dashboard avec intégration du paper trading en temps réel

echo "🎯 ADAN DASHBOARD + PAPER TRADING LAUNCHER"
echo "=========================================="
echo ""

# Vérifier que le paper trading est actif
echo "1️⃣  Vérification du paper trading..."
if pgrep -f "paper_trading_monitor.py" > /dev/null; then
    echo "✅ Paper trading actif"
else
    echo "❌ Paper trading inactif - Lancement..."
    nohup python scripts/paper_trading_monitor.py > monitor_hierarchy_fixed.log 2>&1 &
    sleep 5
fi

echo ""
echo "2️⃣  Lancement du dashboard..."
echo "   - Données réelles de Binance Testnet"
echo "   - Intégration paper trading"
echo "   - Refresh rate: 30s"
echo ""

# Lancer le dashboard avec intégration paper trading
python scripts/adan_btc_dashboard.py --paper-trading --refresh 30.0

echo ""
echo "✅ Dashboard fermé"
