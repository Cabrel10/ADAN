#!/bin/bash
# ADAN Orchestrator - Consensus voting avec les 4 workers
# Utilise paper_trading_monitor.py avec mode ADAN

export BINANCE_TESTNET_API_KEY="OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
export BINANCE_TESTNET_API_SECRET="wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

cd "$(dirname "$0")/.."

python scripts/paper_trading_monitor.py \
  --workers w1,w2,w3,w4 \
  --testnet \
  --capital 50 \
  --cycles 50 \
  --consensus \
  2>&1 | tee testnet/adan.log
