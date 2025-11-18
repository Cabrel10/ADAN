#!/bin/bash
################################################################################
# ADAN Trading Bot - Colab Setup Script v2 (Robust Edition)
# Fixes: ModuleNotFoundError, Dependency Conflicts, Pickling Issues
################################################################################

set -e  # Exit on error

echo "=========================================="
echo "ADAN Trading Bot - Colab Setup v2"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Update system packages
echo -e "\n${YELLOW}[1/8] Updating system packages...${NC}"
apt-get update -qq
apt-get install -y -qq build-essential python3-dev > /dev/null 2>&1

# Step 2: Install TA-Lib dependencies
echo -e "\n${YELLOW}[2/8] Installing TA-Lib C library...${NC}"
apt-get install -y -qq wget > /dev/null 2>&1
cd /tmp
wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.28-src.tar.gz
tar -xzf ta-lib-0.4.28-src.tar.gz
cd ta-lib
./configure > /dev/null 2>&1
make > /dev/null 2>&1
make install > /dev/null 2>&1
ldconfig
cd /tmp && rm -rf ta-lib ta-lib-0.4.28-src.tar.gz

# Step 3: Clone repository
echo -e "\n${YELLOW}[3/8] Cloning ADAN repository...${NC}"
if [ ! -d "/content/ADAN0" ]; then
    cd /content
    git clone https://github.com/Cabrel10/ADAN0.git
else
    cd /content/ADAN0
    git pull origin main
fi

# Step 4: Install Python dependencies with correct versions
echo -e "\n${YELLOW}[4/8] Installing Python dependencies...${NC}"
cd /content/ADAN0

# Upgrade pip first
pip install --upgrade pip setuptools wheel -q

# Install with compatible versions (numpy < 2.0 for stability)
pip install --no-cache-dir -q \
    "numpy>=1.24.0,<2.0.0" \
    "pandas>=2.0.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "matplotlib>=3.7.0" \
    "yfinance>=0.2.0" \
    "pandas-ta>=0.3.14b0" \
    "torch>=2.0.0" \
    "gymnasium>=0.29.0" \
    "stable-baselines3>=2.0.0" \
    "optuna>=3.0.0" \
    "pyyaml>=6.0" \
    "tqdm>=4.65.0"

# Install TA-Lib Python wrapper
echo -e "\n${YELLOW}[5/8] Installing TA-Lib Python wrapper...${NC}"
pip install --no-cache-dir -q TA-Lib

# Step 6: Install project in editable mode
echo -e "\n${YELLOW}[6/8] Installing ADAN project (editable mode)...${NC}"
pip install --no-cache-dir -e . -q

# Step 7: Verify imports
echo -e "\n${YELLOW}[7/8] Verifying imports...${NC}"
python3 -c "
import sys
try:
    import adan_trading_bot
    print('✓ adan_trading_bot imported successfully')
except ImportError as e:
    print(f'✗ Failed to import adan_trading_bot: {e}')
    sys.exit(1)

try:
    from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    print('✓ MultiAssetChunkedEnv imported successfully')
except ImportError as e:
    print(f'✗ Failed to import MultiAssetChunkedEnv: {e}')
    sys.exit(1)

try:
    from stable_baselines3 import PPO
    print('✓ stable-baselines3 imported successfully')
except ImportError as e:
    print(f'✗ Failed to import stable-baselines3: {e}')
    sys.exit(1)

try:
    import optuna
    print('✓ optuna imported successfully')
except ImportError as e:
    print(f'✗ Failed to import optuna: {e}')
    sys.exit(1)

print('✓ All critical imports verified')
"

# Step 8: Verify data files
echo -e "\n${YELLOW}[8/8] Verifying data files...${NC}"
if [ -f "/content/ADAN0/data/BTCUSDT_5m_1h_4h.parquet" ]; then
    echo "✓ Data file found: BTCUSDT_5m_1h_4h.parquet"
else
    echo "✗ Data file NOT found: BTCUSDT_5m_1h_4h.parquet"
    echo "  Please ensure data files are in /content/ADAN0/data/"
fi

echo -e "\n${GREEN}=========================================="
echo "✓ Setup completed successfully!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Run training: python scripts/train_parallel_agents.py --config-path config/config.yaml"
echo "2. Or run optimization: python scripts/optimize_hyperparams.py --worker w1"
echo ""
