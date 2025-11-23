#!/bin/bash
# STRESS TEST LAUNCHER - Lance tous les tests en parallèle

set -e

cd /home/morningstar/Documents/trading/bot

echo "================================================================================"
echo "                    STRESS TEST SUITE - LANCEMENT"
echo "================================================================================"
echo ""
echo "Objectif: Tester si le modèle est un suiveur de tendance ou un vrai pro"
echo "Périodes: 5 scénarios mortels (Bear 2018, COVID, Altcoin, Range, etc)"
echo ""

# Créer répertoires
mkdir -p stress_tests/results
mkdir -p stress_tests/logs

echo "================================================================================"
echo "ÉTAPE 1: PRÉPARATION DES DONNÉES"
echo "================================================================================"
echo ""
echo "Téléchargement et calcul des indicateurs pour les 5 scénarios..."
echo ""

timeout 1800 python scripts/stress_test_data_prep.py 2>&1 | tee stress_tests/logs/data_prep.log

if [ $? -ne 0 ]; then
    echo "❌ Erreur préparation données"
    exit 1
fi

echo ""
echo "✅ Données préparées"
echo ""

echo "================================================================================"
echo "ÉTAPE 2: LANCEMENT DES BACKTESTS"
echo "================================================================================"
echo ""
echo "Lancement des 5 stress tests..."
echo ""

timeout 3600 python scripts/stress_test_backtest.py 2>&1 | tee stress_tests/logs/backtest.log

if [ $? -ne 0 ]; then
    echo "❌ Erreur backtest"
    exit 1
fi

echo ""
echo "✅ Backtests terminés"
echo ""

echo "================================================================================"
echo "RÉSULTATS"
echo "================================================================================"
echo ""
echo "Fichiers générés:"
echo "  - stress_tests/logs/data_prep.log"
echo "  - stress_tests/logs/backtest.log"
echo "  - stress_tests/results/stress_test_results.csv"
echo ""
echo "Consultez les résultats:"
echo "  cat stress_tests/results/stress_test_results.csv"
echo ""
echo "================================================================================"
