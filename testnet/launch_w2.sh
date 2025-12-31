#!/bin/bash
# W2 Worker - Testnet Trading avec paper_trading_monitor.py

export BINANCE_TESTNET_API_KEY="OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
export BINANCE_TESTNET_API_SECRET="wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

cd "$(dirname "$0")/.."

python scripts/paper_trading_monitor.py \
  --worker w2 \
  --testnet \
  --capital 10 \
  --cycles 100 \
  2>&1 | tee testnet/w2.log
