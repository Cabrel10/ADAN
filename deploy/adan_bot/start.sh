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
