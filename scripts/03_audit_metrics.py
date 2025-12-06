#!/usr/bin/env python3
"""
SCRIPT 3: Audit des Calculs de Métriques
Valide Sharpe, Sortino, Max DD, etc.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, 'src')

class MetricsAuditor:
    def __init__(self):
        self.findings = []
        self.errors = []
        self.validation_results = {}
        
    def test_sharpe_calculation(self):
        """Test 1: Valider le calcul de Sharpe Ratio"""
        print("\n📋 TEST 1: Validation Sharpe Ratio")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            
            metrics = PerformanceMetrics()
            
            # Générer des données de test réalistes
            test_returns = np.random.normal(0.0005, 0.02, 1000)
            
            # Ajouter les returns
            for ret in test_returns:
                metrics.returns.append(ret)
            
            # Calculer avec la méthode Kiro
            sharpe_kiro = metrics.calculate_sharpe_ratio()
            
            # Calculer manuellement
            excess_returns = test_returns - 0.0
            sharpe_manual = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Comparer
            diff = abs(sharpe_kiro - sharpe_manual)
            
            print(f"  Sharpe (Kiro): {sharpe_kiro:.4f}")
            print(f"  Sharpe (Manuel): {sharpe_manual:.4f}")
            print(f"  Différence: {diff:.6f}")
            
            if diff > 0.01:
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Sharpe Calculation',
                    'issue': f'Différence significative: {diff:.6f}',
                    'impact': 'Calcul de Sharpe incorrect'
                })
                print(f"  🟠 ALERTE: Différence significative")
            else:
                print(f"  ✅ Calcul correct")
            
            # Vérifier les valeurs extrêmes
            if np.isnan(sharpe_kiro):
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Sharpe Calculation',
                    'issue': 'Sharpe est NaN',
                    'impact': 'Métrique invalide'
                })
                print(f"  🔴 ALERTE: Sharpe est NaN")
            
            if np.isinf(sharpe_kiro):
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Sharpe Calculation',
                    'issue': 'Sharpe est Inf',
                    'impact': 'Métrique invalide'
                })
                print(f"  🔴 ALERTE: Sharpe est Inf")
            
            # Vérifier si Sharpe est réaliste
            if sharpe_kiro > 5.0:
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Sharpe Calculation',
                    'issue': f'Sharpe trop élevé: {sharpe_kiro:.2f}',
                    'impact': 'Suggère overfitting ou data leakage'
                })
                print(f"  🟠 ALERTE: Sharpe trop élevé ({sharpe_kiro:.2f})")
            
            self.validation_results['sharpe'] = {
                'kiro': sharpe_kiro,
                'manual': sharpe_manual,
                'difference': diff,
                'is_valid': not (np.isnan(sharpe_kiro) or np.isinf(sharpe_kiro))
            }
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def test_max_drawdown_calculation(self):
        """Test 2: Valider le calcul de Max Drawdown"""
        print("\n📋 TEST 2: Validation Max Drawdown")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            
            metrics = PerformanceMetrics()
            
            # Générer une courbe d'équité réaliste
            test_returns = np.random.normal(0.0005, 0.02, 1000)
            test_equity = 10000 * np.exp(np.cumsum(test_returns))
            
            # Ajouter à metrics
            for eq in test_equity:
                metrics.equity_curve.append(eq)
            
            # Calculer avec la méthode Kiro
            max_dd_kiro = metrics.calculate_max_drawdown()
            
            # Calculer manuellement
            peak = np.maximum.accumulate(test_equity)
            dd = (peak - test_equity) / peak
            max_dd_manual = np.max(dd) * 100
            
            # Comparer
            diff = abs(max_dd_kiro - max_dd_manual)
            
            print(f"  Max DD (Kiro): {max_dd_kiro:.2f}%")
            print(f"  Max DD (Manuel): {max_dd_manual:.2f}%")
            print(f"  Différence: {diff:.2f}%")
            
            if diff > 1.0:
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Max Drawdown Calculation',
                    'issue': f'Différence significative: {diff:.2f}%',
                    'impact': 'Calcul de Max DD incorrect'
                })
                print(f"  🟠 ALERTE: Différence significative")
            else:
                print(f"  ✅ Calcul correct")
            
            # Vérifier les valeurs extrêmes
            if np.isnan(max_dd_kiro):
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Max Drawdown Calculation',
                    'issue': 'Max DD est NaN',
                    'impact': 'Métrique invalide'
                })
                print(f"  🔴 ALERTE: Max DD est NaN")
            
            self.validation_results['max_dd'] = {
                'kiro': max_dd_kiro,
                'manual': max_dd_manual,
                'difference': diff,
                'is_valid': not np.isnan(max_dd_kiro)
            }
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def test_sortino_calculation(self):
        """Test 3: Valider le calcul de Sortino Ratio"""
        print("\n📋 TEST 3: Validation Sortino Ratio")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            
            metrics = PerformanceMetrics()
            
            # Générer des données de test
            test_returns = np.random.normal(0.0005, 0.02, 1000)
            
            # Ajouter les returns
            for ret in test_returns:
                metrics.returns.append(ret)
            
            # Calculer avec la méthode Kiro
            sortino_kiro = metrics.calculate_sortino_ratio()
            
            # Calculer manuellement
            excess_returns = test_returns - 0.0
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_manual = np.mean(excess_returns) / downside_std * np.sqrt(252)
            else:
                sortino_manual = 0.0
            
            print(f"  Sortino (Kiro): {sortino_kiro:.4f}")
            print(f"  Sortino (Manuel): {sortino_manual:.4f}")
            
            if sortino_kiro == 0.0:
                print(f"  ⚠️  Sortino non calculé (peut être normal)")
            else:
                print(f"  ✅ Sortino calculé")
            
            self.validation_results['sortino'] = {
                'kiro': sortino_kiro,
                'manual': sortino_manual,
                'is_valid': not (np.isnan(sortino_kiro) or np.isinf(sortino_kiro))
            }
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def test_data_integrity(self):
        """Test 4: Vérifier l'intégrité des données"""
        print("\n📋 TEST 4: Vérification de l'Intégrité des Données")
        print("-" * 50)
        
        try:
            from adan_trading_bot.performance.metrics import PerformanceMetrics
            
            metrics = PerformanceMetrics()
            
            # Ajouter des données
            test_returns = np.random.normal(0.0005, 0.02, 100)
            test_equity = 10000 * np.exp(np.cumsum(test_returns))
            
            for ret in test_returns:
                metrics.returns.append(ret)
            
            for eq in test_equity:
                metrics.equity_curve.append(eq)
            
            # Vérifier les NaN/Inf
            nan_count = sum(1 for r in metrics.returns if np.isnan(r) or np.isinf(r))
            inf_count = sum(1 for r in metrics.returns if np.isinf(r))
            
            print(f"  Returns avec NaN: {nan_count}")
            print(f"  Returns avec Inf: {inf_count}")
            
            if nan_count > 0 or inf_count > 0:
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Data Integrity',
                    'issue': f'Données corrompues: {nan_count} NaN, {inf_count} Inf',
                    'impact': 'Métriques invalides'
                })
                print(f"  🔴 ALERTE: Données corrompues")
            else:
                print(f"  ✅ Données intègres")
            
            # Vérifier la cohérence
            if len(metrics.returns) != len(test_returns):
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Data Integrity',
                    'issue': f'Nombre de returns incorrect',
                    'impact': 'Données perdues'
                })
                print(f"  🟠 ALERTE: Nombre de returns incorrect")
            else:
                print(f"  ✅ Nombre de returns correct")
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def generate_report(self):
        """Génère un rapport d'audit"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_name': 'Metrics Audit',
            'findings': self.findings,
            'errors': self.errors,
            'validation_results': self.validation_results,
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
    print("SCRIPT 3: AUDIT DES CALCULS DE MÉTRIQUES")
    print("=" * 70)
    
    auditor = MetricsAuditor()
    
    # Exécuter les tests
    auditor.test_sharpe_calculation()
    auditor.test_max_drawdown_calculation()
    auditor.test_sortino_calculation()
    auditor.test_data_integrity()
    
    # Générer le rapport
    report = auditor.generate_report()
    
    print(f"\n📊 RÉSUMÉ")
    print(f"  Total Findings: {report['summary']['total_findings']}")
    print(f"  Critical: {report['summary']['critical_findings']}")
    print(f"  Major: {report['summary']['major_findings']}")
    print(f"  Errors: {report['summary']['total_errors']}")
    
    # Sauvegarder
    output_dir = Path('investigation_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'metrics_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: investigation_results/metrics_audit.json")
    
    return report

if __name__ == '__main__':
    main()
