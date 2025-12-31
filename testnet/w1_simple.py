#!/usr/bin/env python3
"""
W1 Simple Launcher - 1 heure (12 steps de 5 minutes)
Lance directement paper_trading_monitor.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Configuration
os.environ["BINANCE_TESTNET_API_KEY"] = "OBpX76eDVonGa51ycDN6NKUtk1tE3FXRsc3wTrFKq5SfFoWTL2U9ZS005nTvQ3oW"
os.environ["BINANCE_TESTNET_API_SECRET"] = "wEqgNGKE2sf6PrchcNYFAMoNkof7p7Jk33YzdOzLjvstM4eO3PD3tzWbAXoe2LoZ"

# Aller au répertoire racine
os.chdir(Path(__file__).parent.parent)

# Lancer paper_trading_monitor.py
cmd = [
    sys.executable,
    "scripts/paper_trading_monitor.py",
    "--api_key", os.environ["BINANCE_TESTNET_API_KEY"],
    "--api_secret", os.environ["BINANCE_TESTNET_API_SECRET"]
]

print("🚀 Lancement de W1 - 1 heure (12 steps)")
print(f"Commande: {' '.join(cmd)}")
print("")

result = subprocess.run(cmd)
sys.exit(result.returncode)
