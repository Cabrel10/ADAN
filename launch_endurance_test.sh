#!/bin/bash

echo "🧪 LANCEMENT DU TEST D'ENDURANCE (3 HEURES)"
echo "=================================================="

# Aller dans le dossier de déploiement
cd deploy/adan_bot

# Vérifier que le paquet est complet
if [ ! -f "requirements.txt" ]; then
    echo "❌ Erreur: requirements.txt non trouvé"
    exit 1
fi

if [ ! -f "start.sh" ]; then
    echo "❌ Erreur: start.sh non trouvé"
    exit 1
fi

# Créer un environnement virtuel temporaire pour le test (simule le serveur)
echo "🔧 Création de l'environnement virtuel de test..."
if [ -d "venv_test" ]; then
    rm -rf venv_test
fi

python3 -m venv venv_test
source venv_test/bin/activate

# Installer les dépendances (Test d'installation)
echo "📦 Installation des dépendances pour le test..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation des dépendances"
    deactivate
    exit 1
fi

echo "✅ Dépendances installées avec succès"

# Créer le répertoire logs s'il n'existe pas
mkdir -p logs

# Lancer le test d'endurance en tâche de fond avec log dédié
echo ""
echo "🚀 Lancement du Test d'Endurance (3h)..."
echo "   Logs: logs/endurance_test.log"
echo ""

# Enregistrer l'heure de démarrage
START_TIME=$(date +%s)
echo "⏱️  Démarrage: $(date '+%Y-%m-%d %H:%M:%S')" > logs/endurance_test.log

# Lancer le bot
nohup python3 scripts/paper_trading_monitor.py >> logs/endurance_test.log 2>&1 &

# Sauvegarder le PID
PID=$!
echo $PID > logs/endurance_test.pid
echo "✅ Bot lancé sous PID $PID"

# Revenir à la racine
cd ../..

# Afficher les instructions
echo ""
echo "📝 INSTRUCTIONS DE SUIVI:"
echo "   1. Suivez les logs en temps réel:"
echo "      tail -f deploy/adan_bot/logs/endurance_test.log"
echo ""
echo "   2. Vérifiez le statut du processus:"
echo "      ps aux | grep paper_trading_monitor"
echo ""
echo "   3. Arrêtez le test manuellement (avant 3h):"
echo "      kill $(cat deploy/adan_bot/logs/endurance_test.pid)"
echo ""
echo "🎯 CRITÈRES DE SUCCÈS (à vérifier dans les logs):"
echo "   ✅ Démarrage: 'Pipeline Ready: 4 workers loaded'"
echo "   ✅ Indicateurs: RSI=xx.xx (variable, pas 50.00)"
echo "   ✅ Workers: Chaque worker donne son avis (w1: BUY, w2: HOLD...)"
echo "   ✅ Fusion: 'ADAN: HOLD' (ou autre décision)"
echo "   ✅ Pas de crash: Le processus doit rester vivant"
echo "   ✅ Pas d'erreur API: Pas de déconnexion massive de Binance"
echo ""
echo "⏰ DURÉE: 3 heures (10800 secondes)"
echo "   Fin prévue: $(date -d '+3 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "=================================================="
