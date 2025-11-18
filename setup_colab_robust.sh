#!/bin/bash
# ADAN Trading Bot - Robust Colab Setup Script
# Handles TA-Lib installation and dependency isolation
# Usage: bash setup_colab_robust.sh

set -e  # Exit on error

echo "╭────────────────────────────────────────────────────────────────╮"
echo "│  🚀 ADAN Trading Bot - Colab Setup (Robust Edition)           │"
echo "╰────────────────────────────────────────────────────────────────╯"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# ============================================================================
# PHASE 1: System Dependencies for TA-Lib
# ============================================================================
print_status "PHASE 1: Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq \
    build-essential \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    python3-dev \
    2>&1 | grep -v "^Get:" | grep -v "^Reading" | grep -v "^Building" || true

print_success "System dependencies installed"

# ============================================================================
# PHASE 2: Install TA-Lib from source (most reliable method)
# ============================================================================
print_status "PHASE 2: Installing TA-Lib..."

# Check if TA-Lib is already installed
if python3 -c "import talib" 2>/dev/null; then
    TALIB_VERSION=$(python3 -c "import talib; print(talib.__version__)")
    print_success "TA-Lib already installed (version: $TALIB_VERSION)"
else
    print_warning "TA-Lib not found, installing from source..."
    
    # Download TA-Lib source
    cd /tmp
    if [ ! -d "ta-lib" ]; then
        print_status "Downloading TA-Lib source code..."
        wget -q https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.28-src.tar.gz
        tar -xzf ta-lib-0.4.28-src.tar.gz
    fi
    
    # Compile TA-Lib
    cd ta-lib
    print_status "Configuring TA-Lib..."
    ./configure --prefix=/usr --enable-shared 2>&1 | tail -5
    
    print_status "Compiling TA-Lib (this may take 2-3 minutes)..."
    make -j$(nproc) 2>&1 | tail -10
    
    print_status "Installing TA-Lib..."
    make install 2>&1 | tail -5
    
    # Update library cache
    ldconfig
    
    # Install Python bindings
    cd /tmp/ta-lib/src/tools/fix_mac_compilation/
    cd /tmp/ta-lib
    
    print_status "Installing Python bindings for TA-Lib..."
    pip install --no-cache-dir ta-lib 2>&1 | grep -E "(Successfully|Collecting|Installing)" || true
    
    # Verify installation
    if python3 -c "import talib; print(talib.__version__)" 2>/dev/null; then
        TALIB_VERSION=$(python3 -c "import talib; print(talib.__version__)")
        print_success "TA-Lib installed successfully (version: $TALIB_VERSION)"
    else
        print_error "TA-Lib installation failed"
        exit 1
    fi
fi

# ============================================================================
# PHASE 3: Clone Repository
# ============================================================================
print_status "PHASE 3: Cloning ADAN repository..."

if [ ! -d "/content/ADAN0" ]; then
    cd /content
    git clone https://github.com/Cabrel10/ADAN0.git 2>&1 | grep -E "(Cloning|done)" || true
    print_success "Repository cloned"
else
    print_warning "Repository already exists, updating..."
    cd /content/ADAN0
    git pull origin main 2>&1 | grep -E "(Already|Fast-forward|Updating)" || true
fi

cd /content/ADAN0

# ============================================================================
# PHASE 4: Install Python Dependencies
# ============================================================================
print_status "PHASE 4: Installing Python dependencies..."

# Upgrade pip
pip install --upgrade pip setuptools wheel -q 2>&1 | tail -3

# Install requirements
if [ -f "requirements-colab.txt" ]; then
    print_status "Installing from requirements-colab.txt..."
    pip install --no-cache-dir -r requirements-colab.txt 2>&1 | grep -E "(Successfully|Collecting|Installing)" | tail -20 || true
else
    print_warning "requirements-colab.txt not found, installing core dependencies..."
    pip install --no-cache-dir \
        torch==2.8.0 \
        numpy>=1.24.0,<2.0.0 \
        pandas>=2.0.0,<3.0.0 \
        stable-baselines3==2.7.0 \
        gymnasium>=0.28.0 \
        optuna>=4.5.0 \
        pyyaml>=6.0 \
        2>&1 | grep -E "(Successfully|Collecting)" | tail -15 || true
fi

print_success "Python dependencies installed"

# ============================================================================
# PHASE 5: Install Package in Development Mode
# ============================================================================
print_status "PHASE 5: Installing ADAN package..."

if [ -f "setup.py" ]; then
    pip install -e . -q 2>&1 | tail -5 || true
    print_success "ADAN package installed"
else
    print_warning "setup.py not found, skipping package installation"
fi

# ============================================================================
# PHASE 6: Verify Imports
# ============================================================================
print_status "PHASE 6: Verifying critical imports..."

python3 << 'EOF'
import sys
import importlib

critical_modules = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('gymnasium', 'Gymnasium'),
    ('stable_baselines3', 'Stable Baselines3'),
    ('talib', 'TA-Lib'),
    ('optuna', 'Optuna'),
    ('yaml', 'PyYAML'),
]

print("\n📦 Import Verification:")
print("─" * 50)

all_ok = True
for module_name, display_name in critical_modules:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name:<20} {version}")
    except ImportError as e:
        print(f"  ✗ {display_name:<20} FAILED: {e}")
        all_ok = False

print("─" * 50)

if all_ok:
    print("\n✅ All critical imports successful!")
    sys.exit(0)
else:
    print("\n❌ Some imports failed!")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "All imports verified"
else
    print_error "Import verification failed"
    exit 1
fi

# ============================================================================
# PHASE 7: Environment Setup
# ============================================================================
print_status "PHASE 7: Setting up environment..."

# Create necessary directories
mkdir -p /content/ADAN0/logs
mkdir -p /content/ADAN0/checkpoints
mkdir -p /content/ADAN0/data

# Set environment variables
export PYTHONPATH="/content/ADAN0:$PYTHONPATH"
export TA_LIBRARY_PATH="/usr/lib"

print_success "Environment configured"

# ============================================================================
# Final Summary
# ============================================================================
echo ""
echo "╭────────────────────────────────────────────────────────────────╮"
echo "│  ✅ Setup Complete!                                           │"
echo "╰────────────────────────────────────────────────────────────────╯"
echo ""
echo "📊 System Information:"
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  TA-Lib: $(python3 -c 'import talib; print(talib.__version__)')"
echo "  Stable-Baselines3: $(python3 -c 'import stable_baselines3; print(stable_baselines3.__version__)')"
echo ""
echo "📁 Working Directory: /content/ADAN0"
echo ""
echo "🚀 Ready to run training!"
echo ""
