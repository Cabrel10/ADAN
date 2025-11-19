#!/bin/bash
# Script d'installation complet pour Google Colab
# Exécution: bash setup_colab.sh

set -e  # Exit on error

echo "🚀 ADAN Trading Bot - Setup Colab"
echo "=================================="

# 1. Montage de Google Drive
echo "📁 Montage de Google Drive..."
python3 -c "from google.colab import drive; drive.mount('/content/drive')" 2>/dev/null || echo "⚠️  Drive déjà monté"

# 2. Extraction du projet
echo "📦 Extraction du projet..."
cd /content
if [ -f "drive/MyDrive/bot_fixed.tar.gz" ]; then
    tar -xzf drive/MyDrive/bot_fixed.tar.gz
    cd bot
else
    echo "❌ Fichier bot_fixed.tar.gz introuvable dans Drive"
    exit 1
fi

# 3. Création de l'environnement conda (optionnel)
echo "🐍 Configuration Python..."
# Colab utilise Python 3.10 par défaut, pas besoin de conda

# 4. Installation des dépendances système
echo "⚙️  Installation des dépendances système..."
apt-get update -qq
apt-get install -y -qq build-essential wget curl git python3-dev libffi-dev libssl-dev pkg-config

# 5. Installation de TA-Lib (optionnel, peut être lent)
echo "📊 Installation de TA-Lib..."
pip install -q ta-lib 2>/dev/null || echo "⚠️  TA-Lib installation skipped"

# 6. Installation des dépendances Python
echo "📚 Installation des dépendances Python..."
pip install --upgrade pip setuptools wheel -q
pip install -q -r requirements-colab.txt

# 7. Installation du package en mode editable
echo "📦 Installation du package ADAN..."
pip install -e . --no-deps -q

# 8. Test des imports
echo "✅ Test des imports..."
python3 scripts/test_imports.py

echo ""
echo "=================================="
echo "✅ Setup terminé avec succès!"
echo "=================================="
echo ""
echo "Prochaine étape: lancer l'entraînement"
echo "python scripts/train_parallel_agents.py --config-path config/config.yaml --checkpoint-dir checkpoints --resume"
