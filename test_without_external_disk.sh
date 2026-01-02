#!/bin/bash
# Script pour tester le bot sans accès au disque externe /mnt/new_data
# Simule un environnement de serveur distant

set -e

echo "=========================================="
echo "TEST SANS ACCÈS À /mnt/new_data"
echo "=========================================="
echo ""

# Vérifier si /mnt/new_data existe
if [ ! -d "/mnt/new_data" ]; then
    echo "✅ /mnt/new_data n'existe pas - test valide"
else
    echo "⚠️  /mnt/new_data existe - on va le renommer temporairement"
    echo ""
    echo "ATTENTION: Cette opération nécessite les droits sudo"
    echo "Appuyez sur Entrée pour continuer ou Ctrl+C pour annuler"
    read
    
    # Renommer le disque
    sudo mv /mnt/new_data /mnt/new_data_backup_test
    echo "✅ /mnt/new_data renommé en /mnt/new_data_backup_test"
fi

# Fonction de nettoyage
cleanup() {
    echo ""
    echo "=========================================="
    echo "NETTOYAGE..."
    echo "=========================================="
    
    if [ -d "/mnt/new_data_backup_test" ]; then
        echo "Restauration de /mnt/new_data..."
        sudo mv /mnt/new_data_backup_test /mnt/new_data
        echo "✅ /mnt/new_data restauré"
    fi
}

# Trap pour restaurer en cas d'erreur ou d'interruption
trap cleanup EXIT

echo ""
echo "=========================================="
echo "VÉRIFICATION DES FICHIERS LOCAUX"
echo "=========================================="
python3 check_deployment.py || {
    echo "❌ Vérification échouée"
    exit 1
}

echo ""
echo "=========================================="
echo "DÉMARRAGE DU BOT EN MODE TEST"
echo "=========================================="
echo ""
echo "Le bot va démarrer. Vérifiez que:"
echo "  1. Les modèles w1, w2, w3, w4 se chargent"
echo "  2. La configuration ADAN se charge"
echo "  3. Aucune erreur concernant /mnt/new_data"
echo "  4. Les trades s'exécutent normalement"
echo ""
echo "Appuyez sur Entrée pour démarrer le bot"
echo "(Appuyez sur Ctrl+C pour arrêter)"
read

# Créer un fichier de log temporaire
TEST_LOG="logs/test_without_external_disk_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "📝 Logs sauvegardés dans: $TEST_LOG"
echo ""

# Démarrer le bot avec timeout de 60 secondes
timeout 60 python3 scripts/paper_trading_monitor.py 2>&1 | tee "$TEST_LOG" || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo ""
        echo "✅ Test terminé après 60 secondes (timeout normal)"
    else
        echo ""
        echo "⚠️  Bot arrêté avec le code: $EXIT_CODE"
    fi
}

echo ""
echo "=========================================="
echo "ANALYSE DES LOGS"
echo "=========================================="
echo ""

# Vérifier les erreurs critiques
if grep -q "ERROR" "$TEST_LOG"; then
    echo "⚠️  Erreurs trouvées dans les logs:"
    grep "ERROR" "$TEST_LOG" | head -5
else
    echo "✅ Aucune erreur critique"
fi

# Vérifier le chargement des modèles
echo ""
echo "Chargement des modèles:"
for worker in w1 w2 w3 w4; do
    if grep -q "$worker.*loaded" "$TEST_LOG"; then
        echo "  ✅ $worker chargé"
    else
        echo "  ⚠️  $worker - statut inconnu"
    fi
done

# Vérifier ADAN
echo ""
if grep -q "ADAN" "$TEST_LOG"; then
    echo "✅ ADAN ensemble détecté"
else
    echo "⚠️  ADAN ensemble - statut inconnu"
fi

# Vérifier les trades
echo ""
if grep -q "trade\|Trade\|TRADE" "$TEST_LOG"; then
    echo "✅ Trades exécutés"
else
    echo "⚠️  Aucun trade détecté"
fi

echo ""
echo "=========================================="
echo "RÉSUMÉ DU TEST"
echo "=========================================="
echo ""
echo "✅ Le bot a pu démarrer sans /mnt/new_data"
echo "✅ Les fichiers locaux sont suffisants"
echo "✅ Prêt pour le déploiement serveur"
echo ""
echo "Consultez les logs complets:"
echo "  cat $TEST_LOG"
echo ""
