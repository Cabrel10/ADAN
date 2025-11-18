#!/bin/bash

################################################################################
# 🚀 ADAN Trading Bot - Configuration Colab
# 
# Ce script configure automatiquement l'environnement Colab avec toutes
# les dépendances nécessaires pour l'entraînement du bot de trading
#
# Exécution: curl -sSL https://raw.githubusercontent.com/Cabrel10/ADAN0/main/setup_colab.sh | bash
################################################################################

set -e

echo "🚀 ADAN Trading Bot - Configuration Colab"
echo "=========================================="
echo ""

# Configuration de base
REPO_URL="https://github.com/Cabrel10/ADAN0.git"
REPO_DIR="/content/ADAN0"
PYTHON_VERSION="3.11"

# ============================================================================
# ÉTAPE 1: Installation des dépendances système
# ============================================================================

echo "📦 [1/8] Installation des dépendances système..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq \
    build-essential \
    wget \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-opencv \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    > /dev/null 2>&1

echo "✅ Dépendances système installées"
echo ""

# ============================================================================
# ÉTAPE 2: Installation de TA-Lib (requis pour indicateurs techniques)
# ============================================================================

echo "📊 [2/8] Installation de TA-Lib..."

if [ ! -f "/usr/local/lib/libta_lib.so" ]; then
    cd /tmp
    
    # Télécharger et compiler TA-Lib
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz 2>/dev/null || {
        echo "⚠️  Téléchargement TA-Lib échoué, tentative alternative..."
        apt-get install -y -qq ta-lib > /dev/null 2>&1 || true
    }
    
    if [ -f "ta-lib-0.4.0-src.tar.gz" ]; then
        tar -xzf ta-lib-0.4.0-src.tar.gz > /dev/null 2>&1
        cd ta-lib/
        ./configure --prefix=/usr > /dev/null 2>&1
        make > /dev/null 2>&1
        make install > /dev/null 2>&1
        ldconfig > /dev/null 2>&1
        cd /content
    fi
    
    echo "✅ TA-Lib compilé et installé"
else
    echo "✅ TA-Lib déjà installé"
fi

echo ""

# ============================================================================
# ÉTAPE 3: Mise à jour de pip et installation des packages Python
# ============================================================================

echo "🐍 [3/8] Installation des packages Python (cela peut prendre 3-5 min)..."

# Mise à jour de pip
pip install -q --upgrade pip setuptools wheel > /dev/null 2>&1

# Installation des packages de base
pip install -q \
    numpy==1.24.3 \
    pandas==2.1.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.0 \
    > /dev/null 2>&1

echo "✅ Packages de base installés"

# Installation des packages de trading
pip install -q \
    yfinance==0.2.32 \
    pandas-ta==0.3.14b0 \
    > /dev/null 2>&1

echo "✅ Packages de trading installés"

# Installation de PyTorch (CPU optimisé pour Colab)
pip install -q \
    torch==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    > /dev/null 2>&1

echo "✅ PyTorch (CPU) installé"

# Installation de Stable Baselines3 et dépendances RL
pip install -q \
    gymnasium==0.29.1 \
    stable-baselines3==2.1.0 \
    optuna==3.4.0 \
    > /dev/null 2>&1

echo "✅ Packages RL installés"

# Installation de TA-Lib Python
pip install -q TA-Lib==0.4.28 > /dev/null 2>&1 || {
    echo "⚠️  TA-Lib Python installation échouée, tentative alternative..."
    pip install -q --no-cache-dir TA-Lib > /dev/null 2>&1 || true
}

echo "✅ TA-Lib Python installé"

# Installation des packages utilitaires
pip install -q \
    pyyaml==6.0.1 \
    tqdm==4.66.1 \
    > /dev/null 2>&1

echo "✅ Packages utilitaires installés"

echo ""

# ============================================================================
# ÉTAPE 4: Clonage du dépôt GitHub
# ============================================================================

echo "📂 [4/8] Clonage du dépôt GitHub..."

if [ -d "$REPO_DIR" ]; then
    echo "♻️  Nettoyage de l'ancien dépôt..."
    rm -rf "$REPO_DIR"
fi

git clone -q "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

echo "✅ Dépôt cloné avec succès"
echo ""

# ============================================================================
# ÉTAPE 5: Vérification de la structure des données
# ============================================================================

echo "🔍 [5/8] Vérification des données..."

if [ ! -d "data/processed/indicators" ]; then
    echo "❌ ERREUR: Répertoire data/processed/indicators manquant!"
    exit 1
fi

# Compter les fichiers parquet
PARQUET_COUNT=$(find data/processed/indicators -name "*.parquet" 2>/dev/null | wc -l)

if [ "$PARQUET_COUNT" -eq 0 ]; then
    echo "❌ ERREUR: Aucun fichier parquet trouvé!"
    exit 1
fi

# Calculer la taille totale
TOTAL_SIZE=$(du -sh data/processed/indicators 2>/dev/null | cut -f1)

echo "✅ $PARQUET_COUNT fichiers parquet trouvés ($TOTAL_SIZE)"

# Afficher les détails
echo ""
echo "   Fichiers disponibles:"
find data/processed/indicators -name "*.parquet" -type f | sort | while read file; do
    size=$(ls -lh "$file" | awk '{print $5}')
    echo "   ✓ $(basename "$file") ($size)"
done

echo ""

# ============================================================================
# ÉTAPE 6: Création des dossiers nécessaires
# ============================================================================

echo "📁 [6/8] Création des dossiers de travail..."

mkdir -p checkpoints
mkdir -p logs/rewards
mkdir -p results
mkdir -p data/raw

echo "✅ Dossiers créés"
echo ""

# ============================================================================
# ÉTAPE 7: Configuration pour Colab (CPU uniquement)
# ============================================================================

echo "⚙️  [7/8] Configuration pour Colab..."

# Créer un fichier de configuration d'environnement
cat > /content/colab_env.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
EOF

chmod +x /content/colab_env.sh

echo "✅ Configuration Colab appliquée"
echo ""

# ============================================================================
# ÉTAPE 8: Test d'import des modules
# ============================================================================

echo "🧪 [8/8] Test d'import des modules..."

python3 << 'PYTHON_TEST'
import sys
sys.path.insert(0, '/content/ADAN0')

try:
    # Test des imports critiques
    import numpy as np
    import pandas as pd
    import torch
    import gymnasium
    from stable_baselines3 import PPO
    import optuna
    import yaml
    
    # Test des imports du projet
    from src.adan_trading_bot.environment.dynamic_behavior_engine import DynamicBehaviorEngine
    from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    from src.adan_trading_bot.common.config_loader import ConfigLoader
    
    print("✅ Tous les imports réussis")
    
    # Afficher les versions
    print("\n📦 Versions installées:")
    print(f"   - NumPy: {np.__version__}")
    print(f"   - Pandas: {pd.__version__}")
    print(f"   - PyTorch: {torch.__version__}")
    print(f"   - Gymnasium: {gymnasium.__version__}")
    print(f"   - Stable-Baselines3: {sys.modules['stable_baselines3'].__version__}")
    print(f"   - Optuna: {optuna.__version__}")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)
PYTHON_TEST

echo ""

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

echo "=========================================="
echo "✅ Configuration Colab terminée!"
echo "=========================================="
echo ""
echo "📊 Informations de l'environnement:"
echo "   - Répertoire: $REPO_DIR"
echo "   - Python: $(python3 --version)"
echo "   - Pip: $(pip --version | awk '{print $2}')"
echo ""
echo "📦 Données disponibles:"
echo "   - Fichiers parquet: $PARQUET_COUNT"
echo "   - Taille totale: $TOTAL_SIZE"
echo ""
echo "🚀 Pour lancer l'entraînement, exécutez:"
echo "   cd $REPO_DIR && bash launch_training.sh"
echo ""
echo "📋 Options de lancement:"
echo "   - 500k timesteps (défaut): bash launch_training.sh"
echo "   - 1M timesteps: bash launch_training.sh 1000000"
echo "   - 10M timesteps: bash launch_training.sh 10000000"
echo ""
echo "=========================================="
echo ""
