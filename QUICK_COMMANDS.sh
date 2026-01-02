#!/bin/bash

# 🚀 COMMANDES RAPIDES - ADAN TRADING BOT

echo "📋 COMMANDES RAPIDES"
echo "===================="
echo ""

echo "1️⃣  VÉRIFIER LE TEST D'ENDURANCE"
echo "   ps aux | grep paper_trading_monitor | grep -v grep"
echo ""

echo "2️⃣  VOIR LES LOGS EN TEMPS RÉEL"
echo "   tail -f deploy/adan_bot/logs/endurance_test.log"
echo ""

echo "3️⃣  VÉRIFIER LE STATUT"
echo "   ./check_test_status.sh"
echo ""

echo "4️⃣  ARRÊTER LE TEST"
echo "   kill \$(cat deploy/adan_bot/logs/endurance_test.pid)"
echo ""

echo "5️⃣  COMPRESSER LE PAQUET"
echo "   ./compress_package.sh"
echo ""

echo "6️⃣  TRANSFÉRER VERS LE SERVEUR"
echo "   scp deploy/packages/adan_bot_*.tar.gz user@server:/path/"
echo ""

echo "7️⃣  DÉPLOYER SUR LE SERVEUR"
echo "   tar -xzf adan_bot_*.tar.gz"
echo "   cd adan_bot"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -r requirements.txt"
echo "   ./start.sh"
echo ""

echo "8️⃣  MONITORER LE BOT"
echo "   tail -f logs/adan_trading_bot.log"
echo ""

echo "===================="
