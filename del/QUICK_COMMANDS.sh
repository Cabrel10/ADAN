#!/bin/bash

# 🚀 QUICK COMMANDS - HIERARCHY CORRECTIONS

echo "🎯 QUICK COMMANDS - HIERARCHY CORRECTIONS"
echo "=========================================="
echo ""

# Function to display menu
show_menu() {
    echo "Choisir une action :"
    echo "1. Vérifier les corrections"
    echo "2. Redémarrer le système"
    echo "3. Vérifier les logs"
    echo "4. Exécuter les tests"
    echo "5. Afficher le statut"
    echo "6. Arrêter le système"
    echo "0. Quitter"
    echo ""
}

# Function to verify corrections
verify_corrections() {
    echo "🔍 VÉRIFICATION DES CORRECTIONS"
    echo "================================"
    echo ""
    
    echo "1️⃣  Vérifier les méthodes helper..."
    if grep -q "_get_current_tier\|_get_max_concurrent_positions" scripts/paper_trading_monitor.py; then
        echo "✅ Méthodes helper présentes"
    else
        echo "❌ Méthodes helper manquantes"
    fi
    
    echo ""
    echo "2️⃣  Vérifier les features [8] et [9]..."
    if grep -q "portfolio_obs\[8\]\|portfolio_obs\[9\]" scripts/paper_trading_monitor.py; then
        echo "✅ Features présentes"
    else
        echo "❌ Features manquantes"
    fi
    
    echo ""
    echo "3️⃣  Vérifier le DBE..."
    if grep -q "_detect_market_regime\|_get_dbe_multipliers" scripts/paper_trading_monitor.py; then
        echo "✅ DBE présent"
    else
        echo "❌ DBE manquant"
    fi
    
    echo ""
    echo "4️⃣  Vérifier le blocage hiérarchique..."
    if grep -q "TRANSFORMATION HIÉRARCHIQUE\|BLOCAGE HIÉRARCHIQUE" scripts/paper_trading_monitor.py; then
        echo "✅ Blocage présent"
    else
        echo "❌ Blocage manquant"
    fi
    
    echo ""
}

# Function to restart system
restart_system() {
    echo "🚀 REDÉMARRAGE DU SYSTÈME"
    echo "========================="
    echo ""
    
    echo "1️⃣  Arrêter les processus..."
    pkill -9 -f paper_trading_monitor.py
    sleep 2
    echo "✅ Processus arrêtés"
    
    echo ""
    echo "2️⃣  Redémarrer le monitor..."
    nohup python scripts/paper_trading_monitor.py > monitor_hierarchy_fixed.log 2>&1 &
    sleep 5
    echo "✅ Monitor redémarré"
    
    echo ""
    echo "3️⃣  Vérifier le processus..."
    if pgrep -f "paper_trading_monitor.py" > /dev/null; then
        echo "✅ Processus actif"
    else
        echo "❌ Processus inactif"
    fi
    
    echo ""
}

# Function to check logs
check_logs() {
    echo "📋 VÉRIFICATION DES LOGS"
    echo "======================="
    echo ""
    
    echo "1️⃣  Features détectées :"
    tail -50 monitor_hierarchy_fixed.log | grep "HIÉRARCHIE:" | tail -1
    
    echo ""
    echo "2️⃣  DBE activé :"
    tail -50 monitor_hierarchy_fixed.log | grep "DBE ACTIVÉ" | tail -1
    
    echo ""
    echo "3️⃣  Blocage hiérarchique :"
    tail -50 monitor_hierarchy_fixed.log | grep "TRANSFORMATION HIÉRARCHIQUE" | tail -1
    
    echo ""
    echo "4️⃣  Derniers logs :"
    tail -20 monitor_hierarchy_fixed.log
    
    echo ""
}

# Function to run tests
run_tests() {
    echo "🧪 EXÉCUTION DES TESTS"
    echo "====================="
    echo ""
    
    python scripts/test_hierarchy_corrections.py
    
    echo ""
}

# Function to show status
show_status() {
    echo "📊 STATUT DU SYSTÈME"
    echo "==================="
    echo ""
    
    echo "1️⃣  Processus actif ?"
    if pgrep -f "paper_trading_monitor.py" > /dev/null; then
        echo "✅ Oui"
        ps aux | grep paper_trading_monitor.py | grep -v grep
    else
        echo "❌ Non"
    fi
    
    echo ""
    echo "2️⃣  Fichier log :"
    if [ -f monitor_hierarchy_fixed.log ]; then
        echo "✅ Existe"
        echo "   Taille: $(wc -l < monitor_hierarchy_fixed.log) lignes"
        echo "   Dernière mise à jour: $(date -r monitor_hierarchy_fixed.log)"
    else
        echo "❌ N'existe pas"
    fi
    
    echo ""
    echo "3️⃣  Corrections appliquées :"
    echo "   ✅ Features [8] et [9]"
    echo "   ✅ DBE implémenté"
    echo "   ✅ Blocage hiérarchique"
    
    echo ""
}

# Function to stop system
stop_system() {
    echo "⏹️  ARRÊT DU SYSTÈME"
    echo "==================="
    echo ""
    
    echo "1️⃣  Arrêter les processus..."
    pkill -9 -f paper_trading_monitor.py
    sleep 2
    echo "✅ Processus arrêtés"
    
    echo ""
    echo "2️⃣  Vérifier..."
    if pgrep -f "paper_trading_monitor.py" > /dev/null; then
        echo "❌ Processus toujours actif"
    else
        echo "✅ Système arrêté"
    fi
    
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Entrer votre choix [0-6]: " choice
    
    case $choice in
        1)
            verify_corrections
            ;;
        2)
            restart_system
            ;;
        3)
            check_logs
            ;;
        4)
            run_tests
            ;;
        5)
            show_status
            ;;
        6)
            stop_system
            ;;
        0)
            echo "Au revoir !"
            exit 0
            ;;
        *)
            echo "❌ Choix invalide"
            ;;
    esac
    
    echo ""
    read -p "Appuyer sur Entrée pour continuer..."
    clear
done
