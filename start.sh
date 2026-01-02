#!/bin/bash
# Script de démarrage du bot ADAN Trading
# À utiliser sur le serveur de déploiement

set -e

# Déterminer le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Créer les répertoires de logs s'ils n'existent pas
mkdir -p logs

# Charger les variables d'environnement si le fichier .env existe
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Vérifier les fichiers critiques
echo "🔍 Vérification pré-démarrage..."
python3 check_deployment.py || {
    echo "❌ Vérification échouée. Impossible de démarrer."
    exit 1
}

# Activer l'environnement virtuel s'il existe
if [ -d venv ]; then
    echo "📦 Activation de l'environnement virtuel..."
    source venv/bin/activate
fi

# Démarrer le bot
echo "🚀 Démarrage du bot ADAN Trading..."
echo "📝 Les logs sont sauvegardés dans: logs/bot.log"
echo ""

python3 scripts/paper_trading_monitor.py >> logs/bot.log 2>&1 &
BOT_PID=$!

echo "✅ Bot démarré avec PID: $BOT_PID"
echo "   Pour arrêter: kill $BOT_PID"
echo "   Pour voir les logs: tail -f logs/bot.log"

# Garder le script actif
wait $BOT_PID
