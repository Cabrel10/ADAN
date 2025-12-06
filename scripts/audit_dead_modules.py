#!/usr/bin/env python3
"""
🔍 AUDIT AUTOMATISÉ - Analyser tous les modules "morts" pour identifier leur valeur
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import json

# Liste des 53 modules morts
DEAD_MODULES = [
    "src/adan_trading_bot/agent/__init__.py",
    "src/adan_trading_bot/agent/custom_recurrent_policy.py",
    "src/adan_trading_bot/evaluation/__init__.py",
    "src/adan_trading_bot/evaluation/decision_quality_analyzer.py",
    "src/adan_trading_bot/exchange_api/__init__.py",
    "src/adan_trading_bot/exchange_api/connector.py",
    "src/adan_trading_bot/live_trading/__init__.py",
    "src/adan_trading_bot/live_trading/experience_buffer.py",
    "src/adan_trading_bot/live_trading/online_reward_calculator.py",
    "src/adan_trading_bot/live_trading/safety_manager.py",
    "src/adan_trading_bot/monitoring/alert_system.py",
    "src/adan_trading_bot/monitoring/system_health_monitor.py",
    "src/adan_trading_bot/monitoring/worker_monitor.py",
    "src/adan_trading_bot/optimization/__init__.py",
    "src/adan_trading_bot/optimization/config/__init__.py",
    "src/adan_trading_bot/optimization/config/experiment_config.py",
    "src/adan_trading_bot/optimization/hyperparameter_optimizer.py",
    "src/adan_trading_bot/optimization/hyperparameter_optimizer_fixed.py",
    "src/adan_trading_bot/optimization/monitoring/__init__.py",
    "src/adan_trading_bot/optimization/monitoring/experiment_tracker.py",
    "src/adan_trading_bot/optimization/optimize_attention.py",
    "src/adan_trading_bot/optimization/scripts/__init__.py",
    "src/adan_trading_bot/optimization/tests/__init__.py",
    "src/adan_trading_bot/optimization/tests/load_testing.py",
    "src/adan_trading_bot/patches/gugu_march_excellence_rewards.py",
    "src/adan_trading_bot/performance/metrics.py",
    "src/adan_trading_bot/portfolio/__init__.py",
    "src/adan_trading_bot/portfolio/portfolio_manager.py",
    "src/adan_trading_bot/risk_management/__init__.py",
    "src/adan_trading_bot/risk_management/position_sizer.py",
    "src/adan_trading_bot/risk_management/risk_manager.py",
    "src/adan_trading_bot/trading/__init__.py",
    "src/adan_trading_bot/trading/action_translator.py",
    "src/adan_trading_bot/trading/action_validator.py",
    "src/adan_trading_bot/trading/fee_manager.py",
    "src/adan_trading_bot/trading/manual_trading_interface.py",
    "src/adan_trading_bot/trading/order_manager.py",
    "src/adan_trading_bot/trading/position_sizer.py",
    "src/adan_trading_bot/trading/safety_manager.py",
    "src/adan_trading_bot/trading/secure_api_manager.py",
    "src/adan_trading_bot/training/__init__.py",
    "src/adan_trading_bot/training/callbacks.py",
    "src/adan_trading_bot/training/dynamic_training_callback.py",
    "src/adan_trading_bot/training/hyperparam_modulator.py",
    "src/adan_trading_bot/training/shared_experience_buffer.py",
    "src/adan_trading_bot/training/trainer.py",
    "src/adan_trading_bot/training/training_orchestrator.py",
    "src/adan_trading_bot/visualization/gradcam_1d.py",
    "src/adan_trading_bot/visualization/plotting_styles.py",
    "src/adan_trading_bot/workflows/workflow_orchestrator.py",
    "src/adan_trading_bot/constants.py",
    "src/adan_trading_bot/main.py",
    "src/adan_trading_bot/online_learning_agent.py",
]

def analyze_module(path: str) -> dict:
    """Analyser un module pour extraire ses caractéristiques"""
    
    if not os.path.exists(path):
        return {
            'exists': False,
            'lines': 0,
            'classes': 0,
            'functions': 0,
            'value': 'MISSING'
        }
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            
        # Compter les classes et fonctions
        classes = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))
        functions = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
        
        # Évaluer la valeur
        value = 'EMPTY'
        if len(lines) > 100:
            value = 'SUBSTANTIAL'
        elif len(lines) > 50:
            value = 'MEDIUM'
        elif len(lines) > 10:
            value = 'SMALL'
        
        # Vérifier si c'est juste un __init__.py vide
        if path.endswith('__init__.py') and len(lines) < 5:
            value = 'INIT_EMPTY'
        
        # Vérifier la complexité
        if classes > 0 or functions > 5:
            value = 'COMPLEX'
        
        return {
            'exists': True,
            'lines': len(lines),
            'classes': classes,
            'functions': functions,
            'value': value,
            'path': path
        }
    except Exception as e:
        return {
            'exists': False,
            'error': str(e),
            'value': 'ERROR'
        }

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  🔍 AUDIT AUTOMATISÉ - Analyse des 53 modules morts           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Analyser tous les modules
    results = {}
    categories = defaultdict(list)
    
    for module_path in DEAD_MODULES:
        analysis = analyze_module(module_path)
        results[module_path] = analysis
        
        # Catégoriser par valeur
        value = analysis['value']
        categories[value].append({
            'path': module_path,
            'lines': analysis.get('lines', 0),
            'classes': analysis.get('classes', 0),
            'functions': analysis.get('functions', 0)
        })
    
    # Afficher les résultats par catégorie
    print("📊 RÉSUMÉ PAR CATÉGORIE")
    print("═" * 70)
    print()
    
    # Modules COMPLEXES (à réintégrer)
    if categories['COMPLEX']:
        print(f"🔴 MODULES COMPLEXES À RÉINTÉGRER ({len(categories['COMPLEX'])})")
        print("─" * 70)
        for item in sorted(categories['COMPLEX'], key=lambda x: x['lines'], reverse=True):
            print(f"  📄 {item['path']}")
            print(f"     Lines: {item['lines']}, Classes: {item['classes']}, Functions: {item['functions']}")
        print()
    
    # Modules SUBSTANTIELS
    if categories['SUBSTANTIAL']:
        print(f"🟡 MODULES SUBSTANTIELS ({len(categories['SUBSTANTIAL'])})")
        print("─" * 70)
        for item in sorted(categories['SUBSTANTIAL'], key=lambda x: x['lines'], reverse=True):
            print(f"  📄 {item['path']}")
            print(f"     Lines: {item['lines']}, Classes: {item['classes']}, Functions: {item['functions']}")
        print()
    
    # Modules MEDIUM
    if categories['MEDIUM']:
        print(f"🟠 MODULES MEDIUM ({len(categories['MEDIUM'])})")
        print("─" * 70)
        for item in sorted(categories['MEDIUM'], key=lambda x: x['lines'], reverse=True)[:5]:
            print(f"  📄 {item['path']}")
            print(f"     Lines: {item['lines']}, Classes: {item['classes']}, Functions: {item['functions']}")
        if len(categories['MEDIUM']) > 5:
            print(f"  ... et {len(categories['MEDIUM']) - 5} autres")
        print()
    
    # Modules VIDES
    if categories['INIT_EMPTY'] or categories['EMPTY']:
        empty_count = len(categories['INIT_EMPTY']) + len(categories['EMPTY'])
        print(f"⚪ MODULES VIDES ({empty_count})")
        print("─" * 70)
        print(f"  {empty_count} modules vides ou __init__.py (SAFE TO DELETE)")
        print()
    
    # Résumé
    print("=" * 70)
    print("📈 RÉSUMÉ GLOBAL")
    print("=" * 70)
    print()
    
    total_lines = sum(r.get('lines', 0) for r in results.values())
    total_classes = sum(r.get('classes', 0) for r in results.values())
    total_functions = sum(r.get('functions', 0) for r in results.values())
    
    print(f"Total modules: {len(DEAD_MODULES)}")
    print(f"Total lignes: {total_lines}")
    print(f"Total classes: {total_classes}")
    print(f"Total fonctions: {total_functions}")
    print()
    
    print("Répartition:")
    for value in ['COMPLEX', 'SUBSTANTIAL', 'MEDIUM', 'SMALL', 'INIT_EMPTY', 'EMPTY', 'MISSING', 'ERROR']:
        count = len(categories[value])
        if count > 0:
            print(f"  {value}: {count} modules")
    print()
    
    # Recommandations
    print("=" * 70)
    print("🎯 RECOMMANDATIONS")
    print("=" * 70)
    print()
    
    complex_count = len(categories['COMPLEX'])
    substantial_count = len(categories['SUBSTANTIAL'])
    empty_count = len(categories['INIT_EMPTY']) + len(categories['EMPTY'])
    
    print(f"✅ À RÉINTÉGRER: {complex_count + substantial_count} modules")
    print(f"   - {complex_count} modules complexes (priorité haute)")
    print(f"   - {substantial_count} modules substantiels (priorité moyenne)")
    print()
    
    print(f"🗑️  SAFE TO DELETE: {empty_count} modules")
    print(f"   - Modules vides ou __init__.py sans contenu")
    print()
    
    print(f"❓ À ANALYSER: {len(categories['SMALL'])} modules")
    print(f"   - Modules petits (10-50 lignes)")
    print()
    
    # Sauvegarder les résultats
    output_file = 'investigation_results/dead_modules_audit.json'
    os.makedirs('investigation_results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_modules': len(DEAD_MODULES),
                'total_lines': total_lines,
                'total_classes': total_classes,
                'total_functions': total_functions,
                'complex': complex_count,
                'substantial': substantial_count,
                'empty': empty_count,
            },
            'categories': {k: v for k, v in categories.items()},
            'details': results
        }, f, indent=2)
    
    print(f"✅ Résultats sauvegardés: {output_file}")
    print()

if __name__ == '__main__':
    main()

