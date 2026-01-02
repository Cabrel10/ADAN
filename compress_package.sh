#!/bin/bash

echo "📦 COMPRESSION DU PAQUET DE DÉPLOIEMENT"
echo "========================================"
echo ""

# Vérifier que le paquet existe
if [ ! -d "deploy/adan_bot" ]; then
    echo "❌ Dossier deploy/adan_bot non trouvé"
    exit 1
fi

# Créer le dossier de sortie
mkdir -p deploy/packages

# Nom du fichier
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="adan_bot_${TIMESTAMP}.tar.gz"
PACKAGE_PATH="deploy/packages/${PACKAGE_NAME}"

echo "📦 Compression en cours..."
echo "   Source: deploy/adan_bot/"
echo "   Destination: ${PACKAGE_PATH}"
echo ""

# Compresser
cd deploy
tar -czf "packages/${PACKAGE_NAME}" adan_bot/
cd ..

if [ $? -eq 0 ]; then
    echo "✅ Compression réussie"
    echo ""
    
    # Afficher les stats
    SIZE=$(du -sh "${PACKAGE_PATH}" | cut -f1)
    echo "📊 Statistiques:"
    echo "   Taille: ${SIZE}"
    echo "   Fichier: ${PACKAGE_PATH}"
    echo ""
    
    # Créer un checksum
    SHA256=$(sha256sum "${PACKAGE_PATH}" | cut -d' ' -f1)
    echo "🔐 SHA256: ${SHA256}"
    echo ""
    
    # Sauvegarder les infos
    cat > "deploy/packages/${PACKAGE_NAME}.info" << EOF
Package: ${PACKAGE_NAME}
Created: $(date)
Size: ${SIZE}
SHA256: ${SHA256}
Contents:
  - src/ (code source)
  - scripts/ (scripts de lancement)
  - config/ (configurations)
  - models/ (modèles pré-entraînés)
  - requirements.txt (dépendances)
  - .env (variables d'environnement)
  - start.sh (script de démarrage)
  - README.md (documentation)
EOF
    
    echo "📋 Infos sauvegardées dans: deploy/packages/${PACKAGE_NAME}.info"
    echo ""
    echo "========================================"
    echo "✅ PAQUET PRÊT POUR DÉPLOIEMENT"
    echo ""
    echo "📤 Pour envoyer sur le serveur:"
    echo "   scp ${PACKAGE_PATH} user@server:/path/to/deploy/"
    echo ""
    echo "📥 Sur le serveur:"
    echo "   tar -xzf ${PACKAGE_NAME}"
    echo "   cd adan_bot"
    echo "   ./start.sh"
else
    echo "❌ Erreur lors de la compression"
    exit 1
fi
