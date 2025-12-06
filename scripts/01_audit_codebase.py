#!/usr/bin/env python3
"""
SCRIPT 1: Audit Statique Complet de la Codebase
Analyse tous les fichiers Python et génère des métriques
"""

import ast
import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

class CodeAuditor:
    def __init__(self, src_path="src/adan_trading_bot"):
        self.src_path = Path(src_path)
        self.stats = defaultdict(dict)
        self.total_lines = 0
        self.total_functions = 0
        self.total_classes = 0
        self.imports_map = defaultdict(set)
        self.orphan_modules = []
        self.dead_code_lines = 0
        
    def analyze_module(self, file_path):
        """Analyse statique approfondie d'un module"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            imports = []
            functions = []
            classes = []
            lines = len(content.split('\n'))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'lines': node.end_lineno - node.lineno if node.end_lineno else 0
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'lines': node.end_lineno - node.lineno if node.end_lineno else 0
                    })
            
            return {
                'imports': len(imports),
                'functions': len(functions),
                'classes': len(classes),
                'lines': lines,
                'import_list': imports,
                'function_names': [f['name'] for f in functions],
                'class_names': [c['name'] for c in classes],
                'functions_detail': functions,
                'classes_detail': classes,
                'complexity': self.estimate_complexity(tree)
            }
        except Exception as e:
            return {
                'error': str(e),
                'lines': len(content.split('\n'))
            }
    
    def estimate_complexity(self, tree):
        """Estime la complexité cyclomatique"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def run_complete_audit(self):
        """Audit complet de toute la codebase"""
        results = {}
        all_imports = set()
        
        for root, dirs, files in os.walk(self.src_path):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.src_path)
                    results[rel_path] = self.analyze_module(full_path)
                    
                    if 'lines' in results[rel_path]:
                        self.total_lines += results[rel_path]['lines']
                    if 'functions' in results[rel_path]:
                        self.total_functions += results[rel_path]['functions']
                    if 'classes' in results[rel_path]:
                        self.total_classes += results[rel_path]['classes']
                    
                    # Collecter les imports
                    if 'import_list' in results[rel_path]:
                        for imp in results[rel_path]['import_list']:
                            all_imports.add(imp)
        
        return results, all_imports
    
    def identify_orphan_modules(self, results):
        """Identifie les modules orphelins"""
        all_modules = set(results.keys())
        used_modules = set()
        
        for module, data in results.items():
            if 'import_list' in data:
                for imp in data['import_list']:
                    # Convertir l'import en chemin de module
                    if 'adan_trading_bot' in imp:
                        module_path = imp.replace('adan_trading_bot.', '').replace('.', '/')
                        used_modules.add(module_path + '.py')
        
        orphans = all_modules - used_modules
        return list(orphans)
    
    def generate_report(self, results):
        """Génère un rapport d'audit"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(results),
            'total_lines': self.total_lines,
            'total_functions': self.total_functions,
            'total_classes': self.total_classes,
            'avg_lines_per_module': self.total_lines / len(results) if results else 0,
            'avg_complexity': sum(r.get('complexity', 0) for r in results.values()) / len(results) if results else 0,
            'modules': results,
            'orphan_modules': self.identify_orphan_modules(results),
            'statistics': {
                'modules_with_errors': sum(1 for r in results.values() if 'error' in r),
                'modules_over_500_lines': sum(1 for r in results.values() if r.get('lines', 0) > 500),
                'modules_over_1000_lines': sum(1 for r in results.values() if r.get('lines', 0) > 1000),
                'high_complexity_modules': sum(1 for r in results.values() if r.get('complexity', 0) > 20),
            }
        }
        return report

def main():
    print("=" * 70)
    print("SCRIPT 1: AUDIT STATIQUE COMPLET DE LA CODEBASE")
    print("=" * 70)
    
    auditor = CodeAuditor()
    results, all_imports = auditor.run_complete_audit()
    report = auditor.generate_report(results)
    
    # Afficher le résumé
    print(f"\n📊 RÉSUMÉ DE L'AUDIT")
    print(f"  Total Modules: {report['total_modules']}")
    print(f"  Total Lines: {report['total_lines']:,}")
    print(f"  Total Functions: {report['total_functions']}")
    print(f"  Total Classes: {report['total_classes']}")
    print(f"  Avg Lines/Module: {report['avg_lines_per_module']:.0f}")
    print(f"  Avg Complexity: {report['avg_complexity']:.2f}")
    
    print(f"\n⚠️  STATISTIQUES")
    print(f"  Modules avec erreurs: {report['statistics']['modules_with_errors']}")
    print(f"  Modules > 500 lignes: {report['statistics']['modules_over_500_lines']}")
    print(f"  Modules > 1000 lignes: {report['statistics']['modules_over_1000_lines']}")
    print(f"  Modules haute complexité: {report['statistics']['high_complexity_modules']}")
    
    print(f"\n🔴 MODULES ORPHELINS ({len(report['orphan_modules'])})")
    for orphan in sorted(report['orphan_modules'])[:10]:
        print(f"  - {orphan}")
    if len(report['orphan_modules']) > 10:
        print(f"  ... et {len(report['orphan_modules']) - 10} autres")
    
    # Sauvegarder le rapport
    output_dir = Path('investigation_results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'code_audit.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Rapport sauvegardé: investigation_results/code_audit.json")
    
    return report

if __name__ == '__main__':
    main()
