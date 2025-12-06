#!/bin/bash
# SCRIPT MAÎTRE: Exécute tous les audits d'investigation

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🔬 SUITE COMPLÈTE D'INVESTIGATION - ADAN 2.0              ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Créer le répertoire de résultats
mkdir -p investigation_results

echo ""
echo "📅 Date: $(date)"
echo "📁 Répertoire de sortie: investigation_results"
echo ""

# Phase 1: Audit Statique
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 1: AUDIT STATIQUE DE LA CODEBASE                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "scripts/01_audit_codebase.py" ]; then
    echo "▶️  Exécution: 01_audit_codebase.py"
    python scripts/01_audit_codebase.py
    echo "✅ Phase 1 complétée"
else
    echo "❌ Script 01_audit_codebase.py non trouvé"
fi

echo ""

# Phase 2: Audit du Reward Calculator
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 2: AUDIT DU REWARD CALCULATOR                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "scripts/02_audit_reward_calculator.py" ]; then
    echo "▶️  Exécution: 02_audit_reward_calculator.py"
    python scripts/02_audit_reward_calculator.py
    echo "✅ Phase 2 complétée"
else
    echo "❌ Script 02_audit_reward_calculator.py non trouvé"
fi

echo ""

# Phase 3: Audit des Métriques
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 3: AUDIT DES CALCULS DE MÉTRIQUES                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "scripts/03_audit_metrics.py" ]; then
    echo "▶️  Exécution: 03_audit_metrics.py"
    python scripts/03_audit_metrics.py
    echo "✅ Phase 3 complétée"
else
    echo "❌ Script 03_audit_metrics.py non trouvé"
fi

echo ""

# Phase 4: Audit des Logs
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 4: AUDIT DES LOGS                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ -f "scripts/04_audit_logs.py" ]; then
    echo "▶️  Exécution: 04_audit_logs.py"
    python scripts/04_audit_logs.py
    echo "✅ Phase 4 complétée"
else
    echo "❌ Script 04_audit_logs.py non trouvé"
fi

echo ""

# Phase 5: Consolidation
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 5: CONSOLIDATION DES RÉSULTATS                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python << 'EOF'
import json
from pathlib import Path
from datetime import datetime

print("▶️  Consolidation des résultats...")

output_dir = Path('investigation_results')
consolidated = {
    'timestamp': datetime.now().isoformat(),
    'investigation_status': 'COMPLETE',
    'phases': {}
}

# Charger tous les rapports
reports = {
    'code_audit': 'code_audit.json',
    'reward_calculator': 'reward_calculator_audit.json',
    'metrics': 'metrics_audit.json',
    'logs': 'logs_audit.json'
}

for phase_name, filename in reports.items():
    filepath = output_dir / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            data = json.load(f)
        consolidated['phases'][phase_name] = data
        print(f"  ✅ {phase_name}: chargé")
    else:
        print(f"  ⚠️  {phase_name}: non trouvé")

# Générer le résumé global
print("\n▶️  Génération du résumé global...")

total_findings = 0
critical_findings = 0
major_findings = 0

for phase_name, phase_data in consolidated['phases'].items():
    if 'findings' in phase_data:
        findings = phase_data['findings']
        total_findings += len(findings)
        critical_findings += sum(1 for f in findings if f.get('severity') == 'CRITICAL')
        major_findings += sum(1 for f in findings if f.get('severity') == 'MAJOR')

consolidated['summary'] = {
    'total_findings': total_findings,
    'critical_findings': critical_findings,
    'major_findings': major_findings,
    'investigation_complete': True
}

# Sauvegarder
with open(output_dir / 'consolidated_investigation.json', 'w') as f:
    json.dump(consolidated, f, indent=2)

print(f"  ✅ Consolidation complétée")
print(f"\n📊 RÉSUMÉ GLOBAL")
print(f"  Total Findings: {total_findings}")
print(f"  Critical: {critical_findings}")
print(f"  Major: {major_findings}")

EOF

echo "✅ Phase 5 complétée"

echo ""

# Résumé final
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ ✅ INVESTIGATION COMPLÈTE                                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Résultats disponibles dans: investigation_results/"
echo ""
echo "📄 Fichiers générés:"
echo "  - code_audit.json"
echo "  - reward_calculator_audit.json"
echo "  - metrics_audit.json"
echo "  - logs_audit.json"
echo "  - consolidated_investigation.json"
echo ""
echo "🎯 Prochaines étapes:"
echo "  1. Consulter consolidated_investigation.json pour le résumé"
echo "  2. Analyser les findings critiques"
echo "  3. Exécuter les corrections recommandées"
echo ""
echo "⏱️  Fin: $(date)"
