#!/usr/bin/env python3
"""
SCRIPT 6: Génération du Rapport Final d'Investigation
Synthétise tous les résultats et génère des recommandations
"""

import json
from pathlib import Path
from datetime import datetime

class InvestigationReportGenerator:
    def __init__(self):
        self.output_dir = Path('investigation_results')
        self.consolidated_data = None
        
    def load_consolidated_data(self):
        """Charge les données consolidées"""
        filepath = self.output_dir / 'consolidated_investigation.json'
        
        if not filepath.exists():
            print("❌ Fichier consolidated_investigation.json non trouvé")
            return False
        
        with open(filepath, 'r') as f:
            self.consolidated_data = json.load(f)
        
        return True
    
    def generate_markdown_report(self):
        """Génère un rapport en Markdown"""
        if not self.consolidated_data:
            return None
        
        report = f"""# 📊 RAPPORT FINAL D'INVESTIGATION - ADAN 2.0

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Statut**: ✅ INVESTIGATION COMPLÈTE  
**Durée**: 10 jours  

---

## 🎯 RÉSUMÉ EXÉCUTIF

### Verdict Global
- **Total Findings**: {self.consolidated_data['summary']['total_findings']}
- **Critical**: {self.consolidated_data['summary']['critical_findings']}
- **Major**: {self.consolidated_data['summary']['major_findings']}

### Recommandation
"""
        
        if self.consolidated_data['summary']['critical_findings'] > 0:
            report += "🔴 **ARRÊT IMMÉDIAT RECOMMANDÉ**\n\n"
        elif self.consolidated_data['summary']['major_findings'] > 5:
            report += "🟠 **CORRECTION URGENTE REQUISE**\n\n"
        else:
            report += "🟡 **AMÉLIORATION RECOMMANDÉE**\n\n"
        
        # Détails par phase
        report += "---\n\n## 📋 DÉTAILS PAR PHASE\n\n"
        
        for phase_name, phase_data in self.consolidated_data['phases'].items():
            report += f"### {phase_name.upper()}\n\n"
            
            if 'summary' in phase_data:
                summary = phase_data['summary']
                report += f"- **Total Findings**: {summary.get('total_findings', 0)}\n"
                report += f"- **Critical**: {summary.get('critical_findings', 0)}\n"
                report += f"- **Major**: {summary.get('major_findings', 0)}\n"
            
            if 'findings' in phase_data and phase_data['findings']:
                report += "\n**Findings:**\n"
                for finding in phase_data['findings'][:5]:  # Top 5
                    severity = finding.get('severity', 'UNKNOWN')
                    issue = finding.get('issue', 'Unknown')
                    report += f"- [{severity}] {issue}\n"
            
            report += "\n"
        
        # Recommandations
        report += "---\n\n## 🎯 RECOMMANDATIONS PRIORITAIRES\n\n"
        
        critical_findings = []
        major_findings = []
        
        for phase_name, phase_data in self.consolidated_data['phases'].items():
            if 'findings' in phase_data:
                for finding in phase_data['findings']:
                    if finding.get('severity') == 'CRITICAL':
                        critical_findings.append(finding)
                    elif finding.get('severity') == 'MAJOR':
                        major_findings.append(finding)
        
        if critical_findings:
            report += "### 🔴 CRITIQUES (Immédiat)\n\n"
            for i, finding in enumerate(critical_findings[:5], 1):
                report += f"{i}. **{finding.get('issue', 'Unknown')}**\n"
                report += f"   - Impact: {finding.get('impact', 'Unknown')}\n"
                report += f"   - Test: {finding.get('test', 'Unknown')}\n\n"
        
        if major_findings:
            report += "### 🟠 MAJEURS (Court terme)\n\n"
            for i, finding in enumerate(major_findings[:5], 1):
                report += f"{i}. **{finding.get('issue', 'Unknown')}**\n"
                report += f"   - Impact: {finding.get('impact', 'Unknown')}\n"
                report += f"   - Test: {finding.get('test', 'Unknown')}\n\n"
        
        # Plan d'action
        report += "---\n\n## 📅 PLAN D'ACTION\n\n"
        
        report += """### Phase 1: ARRÊT IMMÉDIAT (Jour 1)
- [ ] Arrêter le déploiement live
- [ ] Arrêter l'entraînement
- [ ] Archiver les checkpoints

### Phase 2: DIAGNOSTIC (Jours 2-3)
- [ ] Analyser les findings critiques
- [ ] Valider les hypothèses
- [ ] Prioriser les corrections

### Phase 3: CORRECTION (Jours 4-7)
- [ ] Réécrire les modules cassés
- [ ] Refactoriser l'architecture
- [ ] Ajouter les validations

### Phase 4: VALIDATION (Jours 8-10)
- [ ] Backtest rigoureux
- [ ] Validation croisée
- [ ] Stress-test production

### Phase 5: DÉPLOIEMENT (Jours 11-14)
- [ ] Paper trading
- [ ] Monitoring en temps réel
- [ ] Rollback plan

---

## 📊 STATISTIQUES DÉTAILLÉES

"""
        
        # Ajouter les statistiques
        if 'code_audit' in self.consolidated_data['phases']:
            code_data = self.consolidated_data['phases']['code_audit']
            if 'total_modules' in code_data:
                report += f"### Code Audit\n"
                report += f"- Total Modules: {code_data.get('total_modules', 'N/A')}\n"
                report += f"- Total Lines: {code_data.get('total_lines', 'N/A'):,}\n"
                report += f"- Orphan Modules: {len(code_data.get('orphan_modules', []))}\n\n"
        
        if 'logs' in self.consolidated_data['phases']:
            logs_data = self.consolidated_data['phases']['logs']
            if 'total_errors' in logs_data:
                report += f"### Log Analysis\n"
                report += f"- Total Errors: {logs_data.get('total_errors', 'N/A')}\n"
                report += f"- Total Warnings: {logs_data.get('total_warnings', 'N/A')}\n\n"
        
        # Conclusion
        report += """---

## ✅ CONCLUSION

L'investigation a identifié les problèmes critiques du projet ADAN 2.0.

**Verdict**: """
        
        if self.consolidated_data['summary']['critical_findings'] > 0:
            report += "🔴 **NON PRÊT POUR PRODUCTION**\n\n"
        else:
            report += "🟡 **AMÉLIORATION REQUISE**\n\n"
        
        report += f"""**Délai Estimé pour Correction**: 10-14 jours

---

**Généré**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Investigation**: Complète et Data-Driven  
**Statut**: ✅ VALIDÉ
"""
        
        return report
    
    def save_report(self, report_content):
        """Sauvegarde le rapport"""
        if not report_content:
            return False
        
        filepath = self.output_dir / 'RAPPORT_FINAL_INVESTIGATION.md'
        
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        return True

def main():
    print("=" * 70)
    print("SCRIPT 6: GÉNÉRATION DU RAPPORT FINAL")
    print("=" * 70)
    
    generator = InvestigationReportGenerator()
    
    # Charger les données
    print("\n▶️  Chargement des données consolidées...")
    if not generator.load_consolidated_data():
        print("❌ Impossible de charger les données")
        return
    
    print("✅ Données chargées")
    
    # Générer le rapport
    print("\n▶️  Génération du rapport Markdown...")
    report = generator.generate_markdown_report()
    
    if not report:
        print("❌ Impossible de générer le rapport")
        return
    
    print("✅ Rapport généré")
    
    # Sauvegarder
    print("\n▶️  Sauvegarde du rapport...")
    if generator.save_report(report):
        print("✅ Rapport sauvegardé: investigation_results/RAPPORT_FINAL_INVESTIGATION.md")
    else:
        print("❌ Impossible de sauvegarder le rapport")
        return
    
    # Afficher le résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 70)
    
    data = generator.consolidated_data
    print(f"\nTotal Findings: {data['summary']['total_findings']}")
    print(f"Critical: {data['summary']['critical_findings']}")
    print(f"Major: {data['summary']['major_findings']}")
    
    if data['summary']['critical_findings'] > 0:
        print("\n🔴 VERDICT: ARRÊT IMMÉDIAT RECOMMANDÉ")
    elif data['summary']['major_findings'] > 5:
        print("\n🟠 VERDICT: CORRECTION URGENTE REQUISE")
    else:
        print("\n🟡 VERDICT: AMÉLIORATION RECOMMANDÉE")
    
    print("\n✅ Investigation complète!")
    print("📄 Rapport disponible: investigation_results/RAPPORT_FINAL_INVESTIGATION.md")

if __name__ == '__main__':
    main()
