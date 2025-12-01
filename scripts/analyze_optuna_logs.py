#!/usr/bin/env python3
"""
Analyse approfondie des logs Optuna pour diagnostiquer le blocage des trades.
"""

import re
import sys
from collections import defaultdict
import numpy as np

def parse_log_file(log_path):
    """Parse le fichier de log et extrait les métriques clés."""
    
    results = {
        'actions': [],
        'rewards': [],
        'portfolio_values': [],
        'trade_counts': [],
        'force_trade_blocks': 0,
        'daily_resets': 0,
        'steps_analyzed': 0
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Extraire les actions
            if 'Executing step with action:' in line:
                match = re.search(r'action: \[(.*?)\]', line)
                if match:
                    action_str = match.group(1)
                    actions = [float(x.strip()) for x in action_str.split()]
                    results['actions'].append(actions)
                    results['steps_analyzed'] += 1
            
            # Extraire les rewards
            if '[REWARD Worker' in line and 'Total:' in line:
                match = re.search(r'Total: ([+-]?\d+\.\d+)', line)
                if match:
                    results['rewards'].append(float(match.group(1)))
                
                # Extraire aussi les counts
                match_counts = re.search(r"Counts: \{.*'daily_total': (\d+)", line)
                if match_counts:
                    results['trade_counts'].append(int(match_counts.group(1)))
            
            # Extraire portfolio value
            if 'Portfolio value:' in line:
                match = re.search(r'Portfolio value: ([0-9.]+)', line)
                if match:
                    results['portfolio_values'].append(float(match.group(1)))
            
            # Compter les blocages force trade
            if 'FORCE_TRADE_CAP' in line:
                results['force_trade_blocks'] += 1
            
            # Compter les resets journaliers
            if 'DAILY RESET' in line:
                results['daily_resets'] += 1
    
    return results

def analyze_actions(actions, threshold=0.05):
    """Analyse les actions par rapport au threshold."""
    
    if not actions:
        return None
    
    # Flatten all actions
    all_actions = [a for step_actions in actions for a in step_actions]
    
    # Analyser uniquement les actions principales (chaque 3ème valeur = décision principale)
    main_actions = [actions[i][j] for i in range(len(actions)) 
                    for j in range(0, len(actions[i]), 3)]
    
    analysis = {
        'total_actions': len(main_actions),
        'mean': np.mean(main_actions),
        'std': np.std(main_actions),
        'min': np.min(main_actions),
        'max': np.max(main_actions),
        'above_threshold': sum(1 for a in main_actions if abs(a) > threshold),
        'below_threshold': sum(1 for a in main_actions if abs(a) <= threshold),
        'pct_above': sum(1 for a in main_actions if abs(a) > threshold) / len(main_actions) * 100
    }
    
    return analysis

def generate_report(results, log_path):
    """Génère un rapport diagnostique."""
    
    print("="*70)
    print("DIAGNOSTIC APPROFONDI DES LOGS OPTUNA")
    print("="*70)
    print(f"\nFichier analysé: {log_path}")
    print(f"Steps analysés: {results['steps_analyzed']}")
    
    # Analyse des actions
    print("\n" + "="*70)
    print("1. ANALYSE DES ACTIONS DU MODÈLE")
    print("="*70)
    
    if results['actions']:
        action_analysis = analyze_actions(results['actions'], threshold=0.05)
        
        print(f"\nTotal d'actions analysées: {action_analysis['total_actions']}")
        print(f"Moyenne: {action_analysis['mean']:.4f}")
        print(f"Écart-type: {action_analysis['std']:.4f}")
        print(f"Range: [{action_analysis['min']:.4f}, {action_analysis['max']:.4f}]")
        print(f"\n🔍 DIAGNOSTIC ACTION_THRESHOLD:")
        print(f"  - Actions |a| > 0.05: {action_analysis['above_threshold']} ({action_analysis['pct_above']:.1f}%)")
        print(f"  - Actions |a| ≤ 0.05: {action_analysis['below_threshold']} ({100-action_analysis['pct_above']:.1f}%)")
        
        if action_analysis['pct_above'] < 10:
            print(f"\n  ⚠️  PROBLÈME CRITIQUE: Seulement {action_analysis['pct_above']:.1f}% des actions dépassent le threshold!")
            print(f"     → Le threshold de 0.05 filtre {100-action_analysis['pct_above']:.1f}% des actions du modèle")
            print(f"     → RECOMMANDATION: Réduire action_threshold à 0.01 ou moins")
    
    # Analyse des rewards
    print("\n" + "="*70)
    print("2. ANALYSE DES RÉCOMPENSES")
    print("="*70)
    
    if results['rewards']:
        print(f"\nTotal de rewards: {len(results['rewards'])}")
        print(f"Moyenne: {np.mean(results['rewards']):.4f}")
        print(f"Écart-type: {np.std(results['rewards']):.4f}")
        print(f"Range: [{np.min(results['rewards']):.4f}, {np.max(results['rewards']):.4f}]")
        
        # Compter les rewards constants
        unique_rewards = len(set(results['rewards']))
        print(f"\nValeurs uniques de reward: {unique_rewards}")
        
        if unique_rewards < 5:
            print("\n  ⚠️  PROBLÈME: Rewards quasi-constants!")
            print(f"     → Le modèle reçoit toujours la même punition")
            print(f"     → Aucun signal d'apprentissage clair")
            print(f"     → RECOMMANDATION: Ajuster les poids de reward")
    
    # Analyse du portfolio
    print("\n" + "="*70)
    print("3. ANALYSE DU PORTFOLIO")
    print("="*70)
    
    if results['portfolio_values']:
        pv_unique = len(set(results['portfolio_values']))
        print(f"\nValeurs uniques: {pv_unique}")
        print(f"Range: [{np.min(results['portfolio_values']):.2f}, {np.max(results['portfolio_values']):.2f}]")
        
        if pv_unique == 1:
            print("\n  ⚠️  PROBLÈME CRITIQUE: Portfolio complètement figé!")
            print(f"     → Aucun trade exécuté (portfolio = {results['portfolio_values'][0]:.2f} constant)")
        elif pv_unique < 5:
            print("\n  ⚠️  Portfolio presque figé (très peu de variation)")
    
    # Analyse des trades
    print("\n" + "="*70)
    print("4. ANALYSE DES TRADES")
    print("="*70)
    
    if results['trade_counts']:
        max_trades = max(results['trade_counts'])
        print(f"\nTrades naturels maximum observé: {max_trades}")
        
        if max_trades == 0:
            print("\n  🚨 BLOCAGE TOTAL: Aucun trade naturel exécuté!")
        
    print(f"\nForce trade blocks: {results['force_trade_blocks']}")
    print(f"Daily resets: {results['daily_resets']}")
    
    if results['force_trade_blocks'] > 100:
        print("\n  ⚠️  Force trades constamment bloqués par la limite journalière")
        print(f"     → {results['force_trade_blocks']} tentatives bloquées")
    
    # Conclusions
    print("\n" + "="*70)
    print("5. CONCLUSIONS & RECOMMANDATIONS")
    print("="*70)
    
    issues_found = []
    
    if results['actions']:
        action_analysis = analyze_actions(results['actions'])
        if action_analysis['pct_above'] < 20:
            issues_found.append({
                'severity': 'CRITIQUE',
                'issue': 'action_threshold trop élevé',
                'fix': 'Réduire action_threshold de 0.05 à 0.01'
            })
    
    if results['rewards'] and np.std(results['rewards']) < 0.5:
        issues_found.append({
            'severity': 'HAUTE',
            'issue': 'Rewards trop constants/punitifs',
            'fix': 'Ajuster base_reward_scale et frequency_penalty'
        })
    
    if results['portfolio_values'] and len(set(results['portfolio_values'])) == 1:
        issues_found.append({
            'severity': 'CRITIQUE',
            'issue': 'Portfolio figé - aucun trade',
            'fix': 'Appliquer TOUS les ajustements ci-dessus'
        })
    
    if issues_found:
        print("\n🔴 PROBLÈMES IDENTIFIÉS:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. [{issue['severity']}] {issue['issue']}")
            print(f"   → {issue['fix']}\n")
    
    # Recommandations d'actions immédiates
    print("\n📋 ACTIONS IMMÉDIATES RECOMMANDÉES:")
    print("\n1. Ajuster config/config.yaml:")
    print("   ```yaml")
    print("   action_threshold: 0.01  # Au lieu de 0.05")
    print("   reward:")
    print("     base_reward_scale: 50.0")
    print("     frequency_penalty: 0.05")
    print("   ```")
    print("\n2. Tester avec un run court (3000 steps)")
    print("\n3. Vérifier que des trades apparaissent")
    print("\n4. Si succès → Relancer Optuna avec config ajustée")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Utiliser le dernier log par défaut
        import os
        if os.path.exists('logs/optuna_optimization.log'):
            log_path = 'logs/optuna_optimization.log'
        else:
            print("Usage: python analyze_optuna_logs.py <path_to_log>")
            sys.exit(1)
    
    results = parse_log_file(log_path)
    generate_report(results, log_path)
