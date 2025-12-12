#!/bin/bash

echo "🧹 NETTOYAGE T10 - LIBÉRATION D'ESPACE DISQUE"
echo "=============================================="

# Vérifier l'espace avant
echo ""
echo "📊 Espace AVANT nettoyage:"
df -h /mnt/new_data | tail -1

# Nettoyer les logs Optuna
echo ""
echo "🗑️  Suppression des logs Optuna..."
rm -rf /mnt/new_data/optuna_results/*/logs/* 2>/dev/null
echo "✅ Logs Optuna supprimés"

# Nettoyer les checkpoints T10
echo ""
echo "🗑️  Suppression des checkpoints T10..."
rm -rf /mnt/new_data/t10_training/checkpoints/* 2>/dev/null
echo "✅ Checkpoints T10 supprimés"

# Nettoyer les logs T10
echo ""
echo "🗑️  Suppression des logs T10..."
rm -rf /mnt/new_data/t10_training/logs/* 2>/dev/null
echo "✅ Logs T10 supprimés"

# Vérifier l'espace après
echo ""
echo "📊 Espace APRÈS nettoyage:"
df -h /mnt/new_data | tail -1

echo ""
echo "✅ Nettoyage terminé!"
