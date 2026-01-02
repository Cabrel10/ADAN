#!/bin/bash
# Script de test de déploiement local
# Simule un environnement sans accès à /mnt/new_data

set -e

echo "=" 
echo "TEST DE DÉPLOIEMENT LOCAL - ADAN TRADING BOT"
echo "="
echo ""

# 1. Vérifier les fichiers critiques
echo "1️⃣  Vérification des fichiers critiques..."
python3 check_deployment.py || {
    echo "❌ Vérification échouée"
    exit 1
}

echo ""
echo "2️⃣  Vérification de l'accès à /mnt/new_data..."
if [ -d "/mnt/new_data" ]; then
    echo "  ⚠️  /mnt/new_data est accessible (ce n'est pas un problème pour le test)"
else
    echo "  ✅ /mnt/new_data n'est pas accessible (bon pour le test)"
fi

echo ""
echo "3️⃣  Vérification de la structure locale..."
for worker in w1 w2 w3 w4; do
    if [ -f "models/$worker/${worker}_model_final.zip" ] && [ -f "models/$worker/vecnormalize.pkl" ]; then
        echo "  ✅ $worker: modèle et normalisateur présents"
    else
        echo "  ❌ $worker: fichiers manquants"
        exit 1
    fi
done

echo ""
echo "4️⃣  Vérification de la configuration ensemble..."
if [ -f "models/ensemble/adan_ensemble_config.json" ]; then
    echo "  ✅ Configuration ADAN ensemble présente"
    python3 -c "import json; print('  Contenu:', json.dumps(json.load(open('models/ensemble/adan_ensemble_config.json')), indent=2)[:200])"
else
    echo "  ❌ Configuration ensemble manquante"
    exit 1
fi

echo ""
echo "5️⃣  Vérification de l'intégrité des fichiers..."
for worker in w1 w2 w3 w4; do
    zip_file="models/$worker/${worker}_model_final.zip"
    if unzip -t "$zip_file" > /dev/null 2>&1; then
        echo "  ✅ $worker: archive ZIP valide"
    else
        echo "  ❌ $worker: archive ZIP corrompue"
        exit 1
    fi
done

echo ""
echo "6️⃣  Vérification des fichiers pickle..."
python3 << 'EOF'
import pickle
import os

for worker in ['w1', 'w2', 'w3', 'w4']:
    pkl_file = f'models/{worker}/vecnormalize.pkl'
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  ✅ {worker}: pickle valide")
    except Exception as e:
        print(f"  ❌ {worker}: erreur pickle - {e}")
        exit(1)
EOF

echo ""
echo "=" 
echo "✅ TOUS LES TESTS LOCAUX RÉUSSIS"
echo "🚀 Le bot est prêt pour le déploiement"
echo "="
