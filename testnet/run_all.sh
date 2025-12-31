#!/bin/bash
# Lancement complet du système ADAN Testnet
# Respecte l'architecture du projet

set -e

# Configuration
export BINANCE_TESTNET_API_KEY="OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
export BINANCE_TESTNET_API_SECRET="wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

echo "🚀 ADAN Testnet Trading - Déploiement complet"
echo "=============================================="
echo ""
echo "✅ Configuration:"
echo "   API: Binance Testnet"
echo "   Données: 100% réelles"
echo "   Stratégie: Consensus de 4 workers"
echo ""

# Créer les répertoires
mkdir -p testnet/logs

# Lancer les 4 workers en parallèle
echo "📍 Lancement des 4 workers..."
echo ""

cd "$(dirname "$0")/.."

python testnet/worker_launcher.py --worker w1 --cycles 100 > testnet/logs/w1.log 2>&1 &
W1_PID=$!
echo "   W1 (PID: $W1_PID)"

python testnet/worker_launcher.py --worker w2 --cycles 100 > testnet/logs/w2.log 2>&1 &
W2_PID=$!
echo "   W2 (PID: $W2_PID)"

python testnet/worker_launcher.py --worker w3 --cycles 100 > testnet/logs/w3.log 2>&1 &
W3_PID=$!
echo "   W3 (PID: $W3_PID)"

python testnet/worker_launcher.py --worker w4 --cycles 100 > testnet/logs/w4.log 2>&1 &
W4_PID=$!
echo "   W4 (PID: $W4_PID)"

echo ""
echo "⏳ Attente de la fin des workers..."

# Attendre que les workers se terminent
wait $W1_PID $W2_PID $W3_PID $W4_PID

echo "✅ Tous les workers terminés"
echo ""

# Lancer ADAN
echo "🎯 Lancement de l'orchestrateur ADAN..."
python testnet/adan_orchestrator.py > testnet/logs/adan.log 2>&1

echo ""
echo "✅ Déploiement complet terminé"
echo ""
echo "📊 Résultats:"
echo "   - testnet/logs/w1.log"
echo "   - testnet/logs/w2.log"
echo "   - testnet/logs/w3.log"
echo "   - testnet/logs/w4.log"
echo "   - testnet/logs/adan.log"
echo "   - testnet/adan_results.json"
