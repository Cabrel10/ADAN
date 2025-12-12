#!/bin/bash

echo "🧹 NETTOYAGE AGRESSIF - LIBÉRATION MAXIMALE D'ESPACE"
echo "===================================================="

# Vérifier l'espace avant
echo ""
echo "📊 Espace AVANT nettoyage:"
df -h /mnt/new_data | tail -1

# Supprimer les fichiers non-essentiels
echo ""
echo "🗑️  Suppression des fichiers non-essentiels..."

# Fichiers ISO (3.7 GB)
rm -f "/mnt/new_data/Windows 7 ORIGINAL Français - Toutes versions  32 & 64 Bits.iso" 2>/dev/null
echo "✅ Fichier ISO supprimé"

# Cours informatique (1.5 GB)
rm -rf /mnt/new_data/Cours_Informatique_L1_L2* 2>/dev/null
echo "✅ Cours informatique supprimés"

# Vérifier l'espace après
echo ""
echo "📊 Espace APRÈS nettoyage:"
df -h /mnt/new_data | tail -1

echo ""
echo "✅ Nettoyage agressif terminé!"
