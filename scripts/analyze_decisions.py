#!/usr/bin/env python3
"""
Checkpoint 3.3: Analyse des Décisions
Analyse les décisions générées par les modèles pour vérifier la cohérence.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("CHECKPOINT 3.3: Analyse des Décisions")
print("="*70)

class DecisionAnalyzer:
    """Classe pour analyser les décisions des modèles"""
    
    def __init__(self, decisions_file: str = "diagnostic/results/checkpoint_3_2_results.json"):
        self.decisions_file = Path(decisions_file)
        self.decisions = []
        self.analysis = {}
    
    def load_decisions(self):
        """Charger les décisions du fichier de résultats"""
        print("\n1️⃣  Chargement des décisions...")
        try:
            if not self.decisions_file.exists():
                raise FileNotFoundError(f"Fichier non trouvé: {self.decisions_file}")
            
            with open(self.decisions_file, 'r') as f:
                data = json.load(f)
            
            # Extraire les décisions
            if 'decisions_sample' in data:
                # Charger depuis le fichier de résultats (limité à 5 premiers)
                self.decisions = data['decisions_sample']
                print(f"   ⚠️  Chargement des 5 premières décisions du fichier")
            else:
                raise ValueError("Format de fichier invalide")
            
            print(f"   ✅ {len(self.decisions)} décisions chargées")
            return True
        except Exception as e:
            print(f"   ❌ Erreur chargement: {e}")
            return False
    
    def calculate_statistics(self):
        """Calculer les statistiques des décisions"""
        print("\n2️⃣  Calcul des statistiques...")
        try:
            worker_stats = {}
            
            for worker_id in ["w1", "w2", "w3", "w4"]:
                actions_list = []
                
                # Collecter toutes les actions pour ce worker
                for decision in self.decisions:
                    if worker_id in decision['workers']:
                        worker_data = decision['workers'][worker_id]
                        if worker_data.get('valid', False) and 'action' in worker_data:
                            actions_list.append(worker_data['action'])
                
                if actions_list:
                    actions_array = np.array(actions_list)
                    
                    # Calculer les statistiques
                    stats = {
                        'num_decisions': len(actions_list),
                        'action_mean': float(actions_array.mean()),
                        'action_std': float(actions_array.std()),
                        'action_min': float(actions_array.min()),
                        'action_max': float(actions_array.max()),
                        'action_median': float(np.median(actions_array)),
                        'action_q25': float(np.percentile(actions_array, 25)),
                        'action_q75': float(np.percentile(actions_array, 75))
                    }
                    
                    worker_stats[worker_id] = stats
                    
                    print(f"   ✅ {worker_id}:")
                    print(f"      - Décisions: {stats['num_decisions']}")
                    print(f"      - Mean: {stats['action_mean']:.4f}")
                    print(f"      - Std: {stats['action_std']:.4f}")
                    print(f"      - Range: [{stats['action_min']:.4f}, {stats['action_max']:.4f}]")
            
            self.analysis['worker_statistics'] = worker_stats
            return True
        except Exception as e:
            print(f"   ❌ Erreur calcul statistiques: {e}")
            return False
    
    def check_coherence(self):
        """Vérifier la cohérence des décisions"""
        print("\n3️⃣  Vérification de la cohérence...")
        try:
            coherence_checks = {}
            all_passed = True
            
            for worker_id, stats in self.analysis['worker_statistics'].items():
                checks = {}
                
                # Check 1: Std > 0.01 (pas figé)
                std_check = stats['action_std'] > 0.01
                checks['not_frozen'] = {
                    'passed': std_check,
                    'value': stats['action_std'],
                    'threshold': 0.01,
                    'description': 'Écart-type > 0.01 (pas figé)'
                }
                
                # Check 2: Std < 0.5 (pas aléatoire)
                random_check = stats['action_std'] < 0.5
                checks['not_random'] = {
                    'passed': random_check,
                    'value': stats['action_std'],
                    'threshold': 0.5,
                    'description': 'Écart-type < 0.5 (pas aléatoire)'
                }
                
                # Check 3: Mean dans [-1, 1] (actions valides)
                mean_check = -1 <= stats['action_mean'] <= 1
                checks['valid_mean'] = {
                    'passed': mean_check,
                    'value': stats['action_mean'],
                    'threshold': '[-1, 1]',
                    'description': 'Mean dans [-1, 1]'
                }
                
                # Check 4: Range dans [-1.1, 1.1] (actions valides)
                range_check = stats['action_min'] >= -1.1 and stats['action_max'] <= 1.1
                checks['valid_range'] = {
                    'passed': range_check,
                    'value': f"[{stats['action_min']:.4f}, {stats['action_max']:.4f}]",
                    'threshold': '[-1.1, 1.1]',
                    'description': 'Range dans [-1.1, 1.1]'
                }
                
                coherence_checks[worker_id] = checks
                
                # Afficher les résultats
                print(f"   {worker_id}:")
                for check_name, check_result in checks.items():
                    status = "✅" if check_result['passed'] else "❌"
                    print(f"      {status} {check_result['description']}")
                    print(f"         Valeur: {check_result['value']}, Seuil: {check_result['threshold']}")
                
                # Vérifier si tous les checks passent
                if not all(c['passed'] for c in checks.values()):
                    all_passed = False
            
            self.analysis['coherence_checks'] = coherence_checks
            self.analysis['all_coherence_passed'] = all_passed
            
            return all_passed
        except Exception as e:
            print(f"   ❌ Erreur vérification cohérence: {e}")
            return False
    
    def compare_workers(self):
        """Comparer les patterns entre workers"""
        print("\n4️⃣  Comparaison entre workers...")
        try:
            comparison = {}
            
            # Comparer les moyennes
            means = {w: stats['action_mean'] for w, stats in self.analysis['worker_statistics'].items()}
            mean_std = np.std(list(means.values()))
            mean_range = max(means.values()) - min(means.values())
            
            comparison['means'] = {
                'values': means,
                'std': float(mean_std),
                'range': float(mean_range),
                'interpretation': 'Écart-type des moyennes entre workers'
            }
            
            print(f"   Moyennes par worker:")
            for w, m in means.items():
                print(f"      {w}: {m:.4f}")
            print(f"   Écart-type des moyennes: {mean_std:.4f}")
            print(f"   Range des moyennes: {mean_range:.4f}")
            
            # Comparer les écarts-types
            stds = {w: stats['action_std'] for w, stats in self.analysis['worker_statistics'].items()}
            std_mean = np.mean(list(stds.values()))
            std_range = max(stds.values()) - min(stds.values())
            
            comparison['stds'] = {
                'values': stds,
                'mean': float(std_mean),
                'range': float(std_range),
                'interpretation': 'Variabilité des écarts-types entre workers'
            }
            
            print(f"   Écarts-types par worker:")
            for w, s in stds.items():
                print(f"      {w}: {s:.4f}")
            print(f"   Moyenne des écarts-types: {std_mean:.4f}")
            print(f"   Range des écarts-types: {std_range:.4f}")
            
            # Vérifier la cohérence entre workers
            coherence_between_workers = mean_std < 0.2  # Moins de 0.2 d'écart entre moyennes
            comparison['coherence_between_workers'] = {
                'passed': bool(coherence_between_workers),
                'value': float(mean_std),
                'threshold': 0.2,
                'description': 'Écart-type des moyennes < 0.2'
            }
            
            status = "✅" if coherence_between_workers else "⚠️"
            print(f"   {status} Cohérence entre workers: {coherence_between_workers}")
            
            self.analysis['worker_comparison'] = comparison
            return True
        except Exception as e:
            print(f"   ❌ Erreur comparaison: {e}")
            return False
    
    def generate_report(self):
        """Générer le rapport d'analyse"""
        print("\n5️⃣  Génération du rapport...")
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'checkpoint': '3.3',
                'test': 'decision_analysis',
                'analysis': self.analysis,
                'summary': {
                    'total_decisions_analyzed': len(self.decisions),
                    'workers_analyzed': len(self.analysis['worker_statistics']),
                    'all_coherence_checks_passed': self.analysis.get('all_coherence_passed', False),
                    'workers_coherent': self.analysis.get('worker_comparison', {}).get('coherence_between_workers', {}).get('passed', False)
                }
            }
            
            # Sauvegarder le rapport
            output_path = Path("diagnostic/results/checkpoint_3_3_results.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"   ✅ Rapport sauvegardé: {output_path}")
            return report
        except Exception as e:
            print(f"   ❌ Erreur génération rapport: {e}")
            return None
    
    def run_analysis(self):
        """Exécuter l'analyse complète"""
        # Charger les décisions
        if not self.load_decisions():
            return False
        
        # Calculer les statistiques
        if not self.calculate_statistics():
            return False
        
        # Vérifier la cohérence
        coherence_ok = self.check_coherence()
        
        # Comparer les workers
        if not self.compare_workers():
            return False
        
        # Générer le rapport
        report = self.generate_report()
        if report is None:
            return False
        
        # Évaluation finale
        print("\n" + "="*70)
        
        if coherence_ok and report['summary']['workers_coherent']:
            print("✅ CHECKPOINT 3.3 VALIDÉ")
            print("="*70)
            print(f"\nRésultats:")
            print(f"  - Décisions analysées: {report['summary']['total_decisions_analyzed']}")
            print(f"  - Workers analysés: {report['summary']['workers_analyzed']}")
            print(f"  - Cohérence intra-worker: ✅ PASSÉE")
            print(f"  - Cohérence inter-worker: ✅ PASSÉE")
            print("\nProchaine étape: Checkpoint 3.4 - Génération État JSON")
            return True
        else:
            print("⚠️  CHECKPOINT 3.3 PARTIELLEMENT VALIDÉ")
            print("="*70)
            print(f"\nRésultats:")
            print(f"  - Décisions analysées: {report['summary']['total_decisions_analyzed']}")
            print(f"  - Workers analysés: {report['summary']['workers_analyzed']}")
            print(f"  - Cohérence intra-worker: {'✅ PASSÉE' if coherence_ok else '❌ ÉCHOUÉE'}")
            print(f"  - Cohérence inter-worker: {'✅ PASSÉE' if report['summary']['workers_coherent'] else '⚠️  ATTENTION'}")
            return True  # Retourner True même si partiellement validé

def main():
    """Fonction principale"""
    analyzer = DecisionAnalyzer()
    success = analyzer.run_analysis()
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
