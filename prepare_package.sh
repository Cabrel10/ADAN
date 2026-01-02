#!/bin/bash

echo "📦 PRÉPARATION DU PAQUET DE DÉPLOIEMENT..."

# 1. Création de la structure propre
rm -rf deploy
mkdir -p deploy/adan_bot
mkdir -p deploy/adan_bot/logs
mkdir -p deploy/adan_bot/phase2_results

# 2. Copie des sources
echo "   - Copie du code source..."
cp -r src deploy/adan_bot/
cp -r scripts deploy/adan_bot/
cp -r config deploy/adan_bot/

# 3. Copie des Modèles (CRITIQUE)
echo "   - Copie des modèles et configurations..."
cp -r models deploy/adan_bot/

# 4. Copie des fichiers racine essentiels
echo "   - Copie des fichiers de configuration..."
cp requirements.txt deploy/adan_bot/
if [ -f .env ]; then
    cp .env deploy/adan_bot/
else
    echo "   ⚠️  Pas de fichier .env trouvé (à configurer sur le serveur)"
fi
if [ -f README_CORRECTIONS.md ]; then
    cp README_CORRECTIONS.md deploy/adan_bot/README.md
fi

# 5. Création du script de lancement universel
echo "   - Création du script de lancement..."
cat > deploy/adan_bot/start.sh << 'SCRIPT'
#!/bin/bash
# Script de lancement ADAN PROD

# Définir le répertoire du script comme répertoire de travail
cd "$(dirname "$0")"

# Vérifier/Créer l'environnement virtuel
if [ ! -d "venv" ]; then
    echo "🔧 Création de l'environnement virtuel..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installation des dépendances..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Variables d'environnement (si non définies dans le système)
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "🚀 DÉMARRAGE ADAN TRADING BOT..."
echo "Logs: logs/adan_trading_bot.log"

# Lancement avec redémarrage automatique en cas de crash (sauf arrêt manuel)
while true; do
    python3 scripts/paper_trading_monitor.py
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "🛑 Arrêt normal du bot."
        break
    else
        echo "⚠️  Crash détecté (Code $EXIT_CODE). Redémarrage dans 10 secondes..."
        sleep 10
    fi
done
SCRIPT

chmod +x deploy/adan_bot/start.sh

# 6. Nettoyage des fichiers temporaires/inutiles dans le déploiement
echo "   - Nettoyage des fichiers cache..."
find deploy/adan_bot -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find deploy/adan_bot -name "*.pyc" -delete 2>/dev/null
find deploy/adan_bot -name ".DS_Store" -delete 2>/dev/null

echo "✅ PAQUET PRÊT DANS : deploy/adan_bot/"
echo ""
echo "📊 Contenu du paquet:"
du -sh deploy/adan_bot/
echo ""
echo "📁 Structure:"
ls -la deploy/adan_bot/
