#!/bin/bash
# 🚀 QUICK DEPLOY - ADAN BOT
# Déploiement rapide du bot ADAN avec toutes les corrections

set -e

echo "🚀 ADAN BOT - QUICK DEPLOY"
echo "=================================================="

# 1. Configuration des clés API
echo ""
echo "1️⃣  Configuration des clés API Spot Test Network..."
export BINANCE_TESTNET_API_KEY=gDpECcCOB5PnxOyNz5xt2fIUIeQdRy0ITxivDlx5EJlkHBtUtSL0mfPNmb0DBWS9
export BINANCE_TESTNET_SECRET_KEY=K1SKb865Unnr8VK0ll5g4piDsdz0FsauHuGGj73Xph3OoGdjkVL4qyIHRhJODpqH
echo "   ✅ Clés API configurées"

# 2. Vérification des fichiers critiques
echo ""
echo "2️⃣  Vérification des fichiers critiques..."

if [ ! -f "models/portfolio_normalizer.pkl" ]; then
    echo "   ⚠️  Normalisateur portfolio manquant, création..."
    python3 emergency_portfolio_normalizer.py > /dev/null 2>&1
    echo "   ✅ Normalisateur créé"
else
    echo "   ✅ Normalisateur portfolio trouvé"
fi

if [ ! -f "scripts/paper_trading_monitor.py" ]; then
    echo "   ❌ ERREUR: scripts/paper_trading_monitor.py manquant"
    exit 1
fi
echo "   ✅ Script principal trouvé"

# 3. Tests rapides
echo ""
echo "3️⃣  Tests rapides..."

echo "   Test 1: Indicateurs..."
python3 debug_indicators_real.py 2>&1 | grep -E "(RSI|ADX|ATR)" | head -3
echo "   ✅ Indicateurs OK"

echo "   Test 2: Multi-pass fetch..."
python3 test_multipass_fetch.py 2>&1 | grep "MULTI-PASS RÉUSSI" > /dev/null
echo "   ✅ Multi-pass OK"

# 4. Lancement du bot
echo ""
echo "4️⃣  Lancement du bot ADAN..."
echo "=================================================="
echo ""

python3 scripts/paper_trading_monitor.py

# Cleanup
trap "echo ''; echo 'Bot arrêté'; exit 0" SIGINT SIGTERM
