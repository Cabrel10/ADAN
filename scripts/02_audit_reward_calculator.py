#!/usr/bin/env python3
"""
SCRIPT 2: Audit du Reward Calculator
Détecte les biais, les bugs d'initialisation et les asymétries
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, 'src')

class RewardCalculatorAuditor:
    def __init__(self):
        self.calls = []
        self.results = []
        self.errors = []
        self.findings = []
        
    def test_initialization(self):
        """Test 1: Vérifier les attributs manquants"""
        print("\n📋 TEST 1: Vérification des Attributs")
        print("-" * 50)
        
        try:
            from adan_trading_bot.environment.reward_calculator import RewardCalculator
            
            config = {
                'reward_shaping': {
                    'realized_pnl_multiplier': 1.0,
                    'unrealized_pnl_multiplier': 0.1,
                    'inaction_penalty': -0.0001,
                    'reward_clipping_range': [-5.0, 5.0]
                }
            }
            
            calc = RewardCalculator(config)
            
            # Vérifier les attributs critiques
            attributes = [
                'returns_dates',
                'current_chunk_id',
                'current_episode_id',
                'returns_history',
                'weights'
            ]
            
            missing = []
            for attr in attributes:
                if not hasattr(calc, attr):
                    missing.append(attr)
                    print(f"  ❌ MANQUANT: {attr}")
                else:
                    print(f"  ✅ PRÉSENT: {attr}")
            
            if missing:
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Initialization',
                    'issue': f'Attributs manquants: {missing}',
                    'impact': 'AttributeError lors de _update_returns_history()'
                })
            
            return calc, missing
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
            return None, []
    
    def test_action_bias(self, calc):
        """Test 2: Détecter le biais BUY 100%"""
        print("\n📋 TEST 2: Détection du Biais d'Actions")
        print("-" * 50)
        
        if calc is None:
            print("  ❌ Reward calculator non disponible")
            return
        
        try:
            actions = []
            rewards = []
            
            # Simuler 1000 appels
            for i in range(1000):
                action = np.random.randint(0, 3)  # 0=HOLD, 1=BUY, 2=SELL
                
                portfolio_metrics = {
                    'total_commission': 0.01,
                    'drawdown': -0.05,
                    'win_rate': 0.5,
                    'closed_positions': []
                }
                
                try:
                    reward = calc.calculate(
                        portfolio_metrics=portfolio_metrics,
                        trade_pnl=0.001 if action == 1 else -0.001,
                        action=action,
                        chunk_id=0,
                        optimal_chunk_pnl=0.01,
                        performance_ratio=0.9
                    )
                    actions.append(action)
                    rewards.append(reward)
                except Exception as e:
                    self.errors.append(f"Erreur à l'itération {i}: {e}")
                    break
            
            # Analyser la distribution
            unique, counts = np.unique(actions, return_counts=True)
            action_dist = dict(zip(unique, counts / len(actions) * 100))
            
            print(f"  Distribution des actions ({len(actions)} appels):")
            for action in range(3):
                action_name = ['HOLD', 'BUY', 'SELL'][action]
                pct = action_dist.get(action, 0)
                print(f"    {action_name}: {pct:.1f}%")
            
            # Vérifier le biais
            buy_pct = action_dist.get(1, 0)
            if buy_pct > 90:
                self.findings.append({
                    'severity': 'CRITICAL',
                    'test': 'Action Bias',
                    'issue': f'Biais BUY extrême: {buy_pct:.1f}%',
                    'impact': 'Modèle ne fait que des achats'
                })
                print(f"  🔴 ALERTE: Biais BUY extrême ({buy_pct:.1f}%)")
            elif buy_pct > 70:
                self.findings.append({
                    'severity': 'MAJOR',
                    'test': 'Action Bias',
                    'issue': f'Biais BUY significatif: {buy_pct:.1f}%',
                    'impact': 'Modèle biaisé vers les achats'
                })
                print(f"  🟠 ALERTE: Biais BUY significatif ({buy_pct:.1f}%)")
            else:
                print(f"  ✅ Distribution acceptable")
            
            # Analyser les récompenses par action
            print(f"\n  Récompenses moyennes par action:")
            for action in range(3):
                action_name = ['HOLD', 'BUY', 'SELL'][action]
                mask = np.array(actions) == action
                if np.any(mask):
                    avg_reward = np.mean(np.array(rewards)[mask])
                    std_reward = np.std(np.array(rewards)[mask])
                    print(f"    {action_name}: {avg_reward:.4f} ± {std_reward:.4f}")
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def test_reward_asymmetry(self, calc):
        """Test 3: Détecter l'asymétrie des récompenses"""
        print("\n📋 TEST 3: Détection de l'Asymétrie des Récompenses")
        print("-" * 50)
        
        if calc is None:
            print("  ❌ Reward calculator non disponible")
            return
        
        try:
            # Tester avec différentes actions
            rewards_by_action = defaultdict(list)
            
            for action in range(3):
                for _ in range(100):
                    portfolio_metrics = {
                        'total_commission': 0.01,
                        'drawdown': -0.05,
                        'win_rate': 0.5,
                        'closed_positions': []
                    }
                    
                    try:
                        reward = calc.calculate(
                            portfolio_metrics=portfolio_metrics,
                            trade_pnl=0.001,
                            action=action
                        )
                        rewards_by_action[action].append(reward)
                    except:
                        pass
            
            # Analyser l'asymétrie
            print(f"  Asymétrie des récompenses:")
            means = {}
            for action in range(3):
                action_name = ['HOLD', 'BUY', 'SELL'][action]
                if rewards_by_action[action]:
                    mean = np.mean(rewards_by_action[action])
                    means[action] = mean
                    print(f"    {action_name}: {mean:.4f}")
            
            # Vérifier l'asymétrie
            if len(means) >= 2:
                max_mean = max(means.values())
                min_mean = min(means.values())
                asymmetry = abs(max_mean - min_mean)
                
                if asymmetry > 0.1:
                    self.findings.append({
                        'severity': 'MAJOR',
                        'test': 'Reward Asymmetry',
                        'issue': f'Asymétrie significative: {asymmetry:.4f}',
                        'impact': 'Récompenses biaisées par action'
                    })
                    print(f"  🟠 ALERTE: Asymétrie significative ({asymmetry:.4f})")
                else:
                    print(f"  ✅ Asymétrie acceptable ({asymmetry:.4f})")
            
        except Exception as e:
            self.errors.append(str(e))
            print(f"  ❌ ERREUR: {e}")
    
    def generate_report(self):
        """Génère un rapport d'audit"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_name': 'Reward Calculator Audit',
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
    print("SCRIPT 2: AUDIT DU REWARD CALCULATOR")
    print("=" * 70)
    
    auditor = RewardCalculatorAuditor()
    
    # Test 1: Initialisation
    calc, missing = auditor.test_initialization()
    
    # Test 2: Biais d'actions
    if calc:
        auditor.test_action_bias(calc)
    
    # Test 3: Asymétrie des récompenses
    if calc:
        auditor.test_reward_asymmetry(calc)
    
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
    
    with open(output_dir / 'reward_calculator_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: investigation_results/reward_calculator_audit.json")
    
    return report

if __name__ == '__main__':
    main()
