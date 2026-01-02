#!/bin/bash
# Script de lancement ADAN en mode isolation
# Utilise UNIQUEMENT les ressources locales (dossier models/)

set -e

echo "=========================================================="
echo "🚀 DÉMARRAGE ADAN - MODE ISOLATION (RESSOURCES LOCALES)"
echo "=========================================================="
echo ""
echo "Configuration :"
echo "  ✅ Source de données : Binance Testnet (Temps réel)"
echo "  ✅ Modèles : Locaux uniquement (./models/)"
echo "  ✅ Normalisateurs : Locaux uniquement (./models/)"
echo "  ✅ Logique : ADAN Ensemble (Fusion pondérée)"
echo "  ✅ Force Trade : DÉSACTIVÉ (Décisions naturelles)"
echo ""
echo "=========================================================="
echo ""

# Vérification des fichiers critiques
echo "🔍 Vérification des fichiers critiques..."
python3 check_deployment.py || {
    echo "❌ Vérification échouée"
    exit 1
}

echo ""
echo "✅ Tous les fichiers sont présents et valides"
echo ""

# Vérifier les clés API
if [ -z "$BINANCE_TESTNET_API_KEY" ]; then
    echo "⚠️  ATTENTION : Clés API non détectées dans l'environnement."
    echo "   Chargement depuis .env..."
    if [ -f .env ]; then
        export $(cat .env | grep -v '^#' | xargs)
    else
        echo "❌ Erreur : Pas de fichier .env et pas de variables d'environnement."
        exit 1
    fi
fi

echo "🚀 Démarrage du bot ADAN..."
echo "   Mode: ISOLATION (ressources locales uniquement)"
echo "   Logs: paper_trading.log"
echo ""
echo "Signes de bon fonctionnement :"
echo "  • Logs: 'Chargement depuis models/w1/vecnormalize.pkl', etc."
echo "  • Pas d'erreur '/mnt/new_data'"
echo "  • Décisions naturelles (HOLD probable au début)"
echo "  • Mise à jour à chaque nouvelle bougie 5m"
echo ""
echo "=========================================================="
echo ""

# Lancer le moniteur
python3 scripts/paper_trading_monitor.py
