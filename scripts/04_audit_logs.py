#!/usr/bin/env python3
"""
SCRIPT 4: Audit des Logs
Analyse les fichiers de logs pour extraire les métriques réelles
"""

import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

class LogAuditor:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.patterns = {
            'sharpe': r'Sharpe[:\s=]+([+-]?\d+\.?\d*)',
            'win_rate': r'Win Rate[:\s=]+([+-]?\d+\.?\d*)%?',
            'profit_factor': r'Profit Factor[:\s=]+([+-]?\d+\.?\d*)',
            'max_dd': r'Max DD[:\s=]+([+-]?\d+\.?\d*)%?',
            'action': r'action[:\s=]+(\d)',
            'reward': r'reward[:\s=]+([+-]?\d+\.?\d*)',
            'error': r'ERROR|CRITICAL|FAILED|Exception',
            'warning': r'WARNING|Warning'
        }
        self.findings = []
        self.errors = []
        
    def parse_log_file(self, log_file):
        """Parse un fichier de log"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self.errors.append(f"Erreur lecture {log_file}: {e}")
            return None
        
        analysis = {
            'file': str(log_file),
            'total_lines': len(content.split('\n')),
            'metrics': defaultdict(list),
            'errors': [],
            'warnings': [],
            'actions': Counter()
        }
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Extraire les métriques
            for metric_name, pattern in self.patterns.items():
                if metric_name in ['error', 'warning']:
                    continue
                
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            value = float(match)
                            analysis['metrics'][metric_name].append(value)
                        except:
                            pass
            
            # Extraire les erreurs
            if re.search(self.patterns['error'], line, re.IGNORECASE):
                analysis['errors'].append({
                    'line': i + 1,
                    'content': line[:200]
                })
            
            # Extraire les warnings
            if re.search(self.patterns['warning'], line, re.IGNORECASE):
                analysis['warnings'].append({
                    'line': i + 1,
                    'content': line[:200]
                })
            
            # Extraire les actions
            action_match = re.search(self.patterns['action'], line, re.IGNORECASE)
            if action_match:
                action = int(action_match.group(1))
                analysis['actions'][action] += 1
        
        return analysis
    
    def analyze_all_logs(self):
        """Analyse tous les fichiers de logs"""
        print("\n📋 Analyse des Fichiers de Logs")
        print("-" * 50)
        
        if not self.log_dir.exists():
            print(f"  ⚠️  Répertoire logs non trouvé: {self.log_dir}")
            return {}
        
        all_analyses = {}
        log_files = list(self.log_dir.glob('*.log'))
        
        if not log_files:
            print(f"  ⚠️  Aucun fichier .log trouvé dans {self.log_dir}")
            return {}
        
        print(f"  Fichiers trouvés: {len(log_files)}")
        
        for log_file in log_files:
            analysis = self.parse_log_file(log_file)
            if analysis:
                all_analyses[log_file.name] = analysis
                print(f"    ✅ {log_file.name}: {analysis['total_lines']} lignes")
        
        return all_analyses
    
    def extract_metrics_statistics(self, all_analyses):
        """Extrait les statistiques des métriques"""
        print("\n📊 Statistiques des Métriques")
        print("-" * 50)
        
        metrics_stats = {}
        
        for metric_name in ['sharpe', 'win_rate', 'profit_factor', 'max_dd', 'reward']:
            all_values = []
            
            for analysis in all_analyses.values():
                all_values.extend(analysis['metrics'].get(metric_name, []))
            
            if all_values:
                import numpy as np
                stats = {
                    'count': len(all_values),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'median': float(np.median(all_values))
                }
                metrics_stats[metric_name] = stats
                
                print(f"  {metric_name.upper()}:")
                print(f"    Count: {stats['count']}")
                print(f"    Min: {stats['min']:.4f}")
                print(f"    Max: {stats['max']:.4f}")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Std: {stats['std']:.4f}")
        
        return metrics_stats
    
    def analyze_action_distribution(self, all_analyses):
        """Analyse la distribution des actions"""
        print("\n📊 Distribution des Actions")
        print("-" * 50)
        
        total_actions = Counter()
        
        for analysis in all_analyses.values():
            total_actions.update(analysis['actions'])
        
        if total_actions:
            total = sum(total_actions.values())
            action_names = ['HOLD', 'BUY', 'SELL']
            
            for action in range(3):
                count = total_actions.get(action, 0)
                pct = count / total * 100 if total > 0 else 0
                print(f"  {action_names[action]}: {count} ({pct:.1f}%)")
            
            # Vérifier le biais
            buy_pct = total_actions.get(1, 0) / total * 100 if total > 0 else 0
            
            if buy_pct > 90:
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Action Distribution',
                    'issue': f'Biais BUY extrême: {buy_pct:.1f}%',
                    'impact': 'Modèle ne fait que des achats'
                })
                print(f"  🔴 ALERTE: Biais BUY extrême ({buy_pct:.1f}%)")
            elif buy_pct > 70:
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Action Distribution',
                    'issue': f'Biais BUY significatif: {buy_pct:.1f}%',
                    'impact': 'Modèle biaisé vers les achats'
                })
                print(f"  🟠 ALERTE: Biais BUY significatif ({buy_pct:.1f}%)")
        
        return total_actions
    
    def analyze_errors_and_warnings(self, all_analyses):
        """Analyse les erreurs et warnings"""
        print("\n⚠️  Erreurs et Warnings")
        print("-" * 50)
        
        total_errors = 0
        total_warnings = 0
        error_types = Counter()
        
        for analysis in all_analyses.values():
            total_errors += len(analysis['errors'])
            total_warnings += len(analysis['warnings'])
            
            for error in analysis['errors']:
                # Extraire le type d'erreur
                error_type = error['content'].split(':')[0][:50]
                error_types[error_type] += 1
        
        print(f"  Total Erreurs: {total_errors}")
        print(f"  Total Warnings: {total_warnings}")
        
        if error_types:
            print(f"\n  Types d'Erreurs les Plus Fréquents:")
            for error_type, count in error_types.most_common(5):
                print(f"    {error_type}: {count}")
        
        if total_errors > 100:
            self.findings.append({
                'severity': 'MAJOR',
                'test': 'Error Analysis',
                'issue': f'Nombre élevé d\'erreurs: {total_errors}',
                'impact': 'Système instable'
            })
            print(f"  🟠 ALERTE: Nombre élevé d'erreurs")
        
        return total_errors, total_warnings
    
    def generate_report(self, all_analyses, metrics_stats, action_dist, errors_warnings):
        """Génère un rapport d'audit"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_name': 'Log Audit',
            'log_files_analyzed': len(all_analyses),
            'metrics_statistics': metrics_stats,
            'action_distribution': dict(action_dist),
            'total_errors': errors_warnings[0],
            'total_warnings': errors_warnings[1],
            'findings': self.findings,
            'errors': self.errors,
            'summary': {
                'total_findings': len(self.findings),
                'critical_findings': sum(1 for f in self.findings if f['severity'] == 'CRITICAL'),
                'major_findings': sum(1 for f in self.findings if f['severity'] == 'MAJOR'),
                'total_errors': len(self.errors)
            }
        }
        return report

def main():
    print("=" * 70)
    print("SCRIPT 4: AUDIT DES LOGS")
    print("=" * 70)
    
    auditor = LogAuditor()
    
    # Analyser tous les logs
    all_analyses = auditor.analyze_all_logs()
    
    if not all_analyses:
        print("\n❌ Aucun log à analyser")
        return
    
    # Extraire les statistiques
    metrics_stats = auditor.extract_metrics_statistics(all_analyses)
    
    # Analyser la distribution des actions
    action_dist = auditor.analyze_action_distribution(all_analyses)
    
    # Analyser les erreurs
    errors_warnings = auditor.analyze_errors_and_warnings(all_analyses)
    
    # Générer le rapport
    report = auditor.generate_report(all_analyses, metrics_stats, action_dist, errors_warnings)
    
    print(f"\n📊 RÉSUMÉ")
    print(f"  Total Findings: {report['summary']['total_findings']}")
    print(f"  Critical: {report['summary']['critical_findings']}")
    print(f"  Major: {report['summary']['major_findings']}")
    
    # Sauvegarder
    output_dir = Path('investigation_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'logs_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: investigation_results/logs_audit.json")
    
    return report

if __name__ == '__main__':
    main()
