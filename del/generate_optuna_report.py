#!/usr/bin/env python3
"""
Génère un rapport de l'optimisation Optuna
"""
import sys
import yaml
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

def generate_report(output_file: str = "optuna_results/T8_REPORT.md"):
    """Génère un rapport complet"""
    
    workers = ["W1", "W2", "W3", "W4"]
    results_dir = Path("optuna_results")
    
    report = []
    report.append("# T8 : Rapport d'Optimisation Optuna\n")
    report.append(f"**Généré** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Résumé
    report.append("## 📊 Résumé\n")
    
    completed = []
    pending = []
    
    for worker in workers:
        result_file = results_dir / f"{worker}_ppo_best_params.yaml"
        if result_file.exists():
            completed.append(worker)
        else:
            pending.append(worker)
    
    report.append(f"- **Complétés** : {', '.join(completed) if completed else 'Aucun'}\n")
    report.append(f"- **En attente** : {', '.join(pending) if pending else 'Aucun'}\n\n")
    
    # Résultats détaillés
    report.append("## 🎯 Résultats Détaillés\n\n")
    
    for worker in workers:
        result_file = results_dir / f"{worker}_ppo_best_params.yaml"
        
        if result_file.exists():
            with open(result_file, 'r') as f:
                results = yaml.safe_load(f)
            
            report.append(f"### {worker} - ✅ COMPLÉTÉ\n\n")
            report.append(f"**Score** : {results['score']:.2f}\n\n")
            
            # Métriques
            metrics = results['metrics']
            report.append("#### Métriques\n")
            report.append(f"| Métrique | Valeur |\n")
            report.append(f"|----------|--------|\n")
            report.append(f"| Sharpe Ratio | {metrics['sharpe']:.2f} |\n")
            report.append(f"| Max Drawdown | {metrics['drawdown']:.1%} |\n")
            report.append(f"| Win Rate | {metrics['win_rate']:.1%} |\n")
            report.append(f"| Total Return | {metrics['total_return']:.1%} |\n")
            report.append(f"| Profit Factor | {metrics['profit_factor']:.2f} |\n")
            report.append(f"| Total Trades | {metrics['trades']} |\n\n")
            
            # Hyperparamètres PPO
            report.append("#### Hyperparamètres PPO\n")
            report.append("```yaml\n")
            for key, value in results['ppo_parameters'].items():
                report.append(f"{key}: {value}\n")
            report.append("```\n\n")
        
        else:
            report.append(f"### {worker} - ⏳ EN ATTENTE\n\n")
            report.append("Optimisation en cours...\n\n")
    
    # Écrire le rapport
    with open(output_file, 'w') as f:
        f.write(''.join(report))
    
    print(f"✅ Rapport généré : {output_file}")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "optuna_results/T8_REPORT.md"
    generate_report(output)
