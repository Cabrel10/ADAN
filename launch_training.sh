#!/bin/bash

################################################################################
# 🚀 ADAN Trading Bot - Script de Lancement d'Entraînement
#
# Usage:
#   bash launch_training.sh              # 500k timesteps (défaut)
#   bash launch_training.sh 1000000      # 1M timesteps
#   bash launch_training.sh 10000000     # 10M timesteps
#
# Environnement: Colab ou Local
# Optimisé pour: CPU (pas GPU requis)
################################################################################

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paramètres par défaut
TIMESTEPS=${1:-500000}
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"
CONFIG_FILE="config/config.yaml"
SCRIPT_FILE="scripts/train_parallel_agents.py"

# Déterminer le répertoire de travail
if [ -f "launch_training.sh" ]; then
    WORK_DIR="."
elif [ -f "ADAN0/launch_training.sh" ]; then
    WORK_DIR="ADAN0"
    cd "$WORK_DIR"
else
    echo "❌ ERREUR: Impossible de trouver le répertoire du projet"
    exit 1
fi

# ============================================================================
# AFFICHAGE DU BANNIÈRE
# ============================================================================

clear
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║         🚀 ADAN TRADING BOT - LANCEMENT DE L'ENTRAÎNEMENT 🚀              ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# VÉRIFICATION DE L'ENVIRONNEMENT
# ============================================================================

echo "🔍 Vérification de l'environnement..."
echo ""

# Vérifier les fichiers essentiels
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERREUR: $CONFIG_FILE manquant!"
    exit 1
fi

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ ERREUR: $SCRIPT_FILE manquant!"
    exit 1
fi

if [ ! -d "data/processed/indicators" ]; then
    echo "❌ ERREUR: Données manquantes (data/processed/indicators)!"
    exit 1
fi

echo "✅ Fichiers essentiels présents"

# Vérifier Python et les packages
python3 << 'PYTHON_CHECK'
import sys
sys.path.insert(0, '.')

try:
    import torch
    import gymnasium
    from stable_baselines3 import PPO
    from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
    print("✅ Tous les packages requis sont installés")
except ImportError as e:
    print(f"❌ Package manquant: {e}")
    sys.exit(1)
PYTHON_CHECK

echo ""

# ============================================================================
# AFFICHAGE DE LA CONFIGURATION
# ============================================================================

echo "⚙️  Configuration de l'entraînement:"
echo ""
echo "   📊 Timesteps: $(printf "%'d" $TIMESTEPS)"
echo "   📁 Répertoire: $(pwd)"
echo "   💾 Config: $CONFIG_FILE"
echo "   📝 Logs: $LOG_DIR/"
echo "   🔄 Checkpoints: $CHECKPOINT_DIR/"
echo ""

# Calculer la durée estimée
if [ "$TIMESTEPS" -le 500000 ]; then
    DURATION="1-2 heures"
elif [ "$TIMESTEPS" -le 1000000 ]; then
    DURATION="2-4 heures"
elif [ "$TIMESTEPS" -le 5000000 ]; then
    DURATION="8-12 heures"
else
    DURATION="24-48 heures"
fi

echo "⏱️  Durée estimée: $DURATION"
echo ""

# ============================================================================
# MISE À JOUR DE LA CONFIGURATION
# ============================================================================

echo "📝 Mise à jour de la configuration..."

# Créer une sauvegarde de la config
cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"

# Mettre à jour les timesteps
python3 << PYTHON_UPDATE
import yaml

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Mettre à jour les timesteps
config['training']['timesteps_per_instance'] = $TIMESTEPS

with open('$CONFIG_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Config mise à jour: timesteps = {$TIMESTEPS:,}")
PYTHON_UPDATE

echo ""

# ============================================================================
# CRÉATION DES DOSSIERS
# ============================================================================

echo "📁 Création des dossiers de travail..."

mkdir -p "$LOG_DIR/rewards"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "results"

echo "✅ Dossiers créés"
echo ""

# ============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# ============================================================================

echo "⚙️  Configuration de l'environnement..."

# Désactiver GPU (forcer CPU)
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# Optimisation CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "✅ Environnement configuré (CPU mode)"
echo ""

# ============================================================================
# AFFICHAGE DES INFORMATIONS SYSTÈME
# ============================================================================

echo "💻 Informations système:"
echo ""

python3 << 'PYTHON_INFO'
import os
import sys
import torch
import numpy as np

print(f"   - Python: {sys.version.split()[0]}")
print(f"   - PyTorch: {torch.__version__}")
print(f"   - NumPy: {np.__version__}")
print(f"   - CPU cores: {os.cpu_count()}")
print(f"   - GPU disponible: {torch.cuda.is_available()}")

# Afficher la mémoire disponible
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   - RAM disponible: {mem.available / (1024**3):.1f} GB")
except:
    pass

print()
PYTHON_INFO

# ============================================================================
# LANCEMENT DE L'ENTRAÎNEMENT
# ============================================================================

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      🚀 LANCEMENT DE L'ENTRAÎNEMENT 🚀                     ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Générer le timestamp pour le log
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "📝 Logs sauvegardés dans: $LOG_FILE"
echo ""
echo "⏳ Entraînement en cours... (Ctrl+C pour arrêter)"
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# Lancer l'entraînement avec logs en temps réel
python3 "$SCRIPT_FILE" \
    --config-path "$CONFIG_FILE" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-level INFO \
    2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                   ✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ✅                   ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📊 Résultats disponibles dans:"
    echo "   - Logs: $LOG_DIR/"
    echo "   - Checkpoints: $CHECKPOINT_DIR/"
    echo "   - Résultats: results/"
    echo ""
    
    # Afficher les statistiques
    echo "📈 Statistiques d'entraînement:"
    echo ""
    
    DECISION_COUNT=$(grep -c "\[DBE_DECISION\]" "$LOG_FILE" 2>/dev/null || echo "0")
    REGIME_COUNT=$(grep -c "\[REGIME_DETECTION\]" "$LOG_FILE" 2>/dev/null || echo "0")
    ERROR_COUNT=$(grep -c "ERROR\|Exception" "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo "   - Décisions DBE: $DECISION_COUNT"
    echo "   - Détections régime: $REGIME_COUNT"
    echo "   - Erreurs: $ERROR_COUNT"
    echo ""
    
    # Afficher les dernières lignes du log
    echo "📋 Derniers logs:"
    echo ""
    tail -10 "$LOG_FILE" | sed 's/^/   /'
    echo ""
    
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "💡 Prochaines étapes:"
    echo "   1. Analyser les résultats dans $LOG_DIR/"
    echo "   2. Vérifier les checkpoints dans $CHECKPOINT_DIR/"
    echo "   3. Lancer l'optimisation Optuna si nécessaire"
    echo ""
    
else
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                      ❌ ERREUR PENDANT L'ENTRAÎNEMENT ❌                   ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📋 Consultez les logs pour plus de détails:"
    echo "   tail -100 $LOG_FILE"
    echo ""
    echo "🔍 Rechercher les erreurs:"
    echo "   grep -i 'error\|exception' $LOG_FILE"
    echo ""
    
    # Afficher les erreurs
    ERROR_LINES=$(grep -i "error\|exception" "$LOG_FILE" 2>/dev/null | head -5)
    if [ ! -z "$ERROR_LINES" ]; then
        echo "Erreurs trouvées:"
        echo "$ERROR_LINES" | sed 's/^/   /'
        echo ""
    fi
    
    exit 1
fi

# ============================================================================
# RESTAURATION DE LA CONFIG
# ============================================================================

# Restaurer la config originale
mv "${CONFIG_FILE}.backup" "$CONFIG_FILE"

echo "✅ Configuration restaurée"
echo ""
