#!/usr/bin/env python3
"""
Script de correction complète de tous les bugs connus
"""
import os
import re
from pathlib import Path

def fix_datetime_timezone_comprehensive(root_dir='src'):
    """Corrige TOUTES les occurrences de datetime.timezone"""
    print("🔧 Correction de datetime.timezone...")
    
    fixes = []
    for path in Path(root_dir).rglob('*.py'):
        with open(path, 'r') as f:
            content = f.read()
        
        original = content
        
        # Pattern 1: datetime.now(datetime.timezone.utc)
        content = re.sub(
            r'datetime\.now\(datetime\.timezone\.utc\)',
            'datetime.now(timezone.utc)',
            content
        )
        
        # Pattern 2: datetime.timezone.utc seul
        content = re.sub(
            r'datetime\.timezone\.utc',
            'timezone.utc',
            content
        )
        
        # Pattern 3: datetime.now(datetime.timezone.utc) - self.last_flush
        content = re.sub(
            r'datetime\.now\(datetime\.timezone\.utc\) - self\.last_flush',
            'datetime.now(timezone.utc) - self.last_flush',
            content
        )
        
        # Ajouter l'import si nécessaire
        if 'timezone.utc' in content:
            # Chercher les imports existants
            has_timezone_import = 'from datetime import' in content and 'timezone' in content
            
            if not has_timezone_import:
                # Ajouter timezone aux imports existants
                if 'from datetime import datetime' in content:
                    content = content.replace(
                        'from datetime import datetime',
                        'from datetime import datetime, timezone',
                        1  # Une seule fois
                    )
                elif 'import datetime' in content:
                    # Ajouter une ligne d'import
                    lines = content.split('\n')
                    import_idx = next((i for i, line in enumerate(lines) if 'import datetime' in line), -1)
                    if import_idx >= 0:
                        lines.insert(import_idx + 1, 'from datetime import timezone')
                        content = '\n'.join(lines)
        
        if content != original:
            with open(path, 'w') as f:
                f.write(content)
            fixes.append(str(path))
    
    return fixes

def verify_fixes(root_dir='src'):
    """Vérifie que les corrections sont appliquées"""
    print("\n🔍 Vérification des corrections...")
    
    issues = []
    
    # Vérifier datetime.timezone
    for path in Path(root_dir).rglob('*.py'):
        with open(path, 'r') as f:
            content = f.read()
        
        if 'datetime.timezone' in content:
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if 'datetime.timezone' in line and not line.strip().startswith('#'):
                    issues.append(f"{path}:{i} - {line.strip()}")
    
    return issues

if __name__ == '__main__':
    print("="*60)
    print("CORRECTION COMPLÈTE DES BUGS")
    print("="*60)
    
    # 1. Corriger datetime.timezone
    fixes = fix_datetime_timezone_comprehensive()
    print(f"✅ {len(fixes)} fichiers corrigés pour datetime.timezone")
    for f in fixes[:10]:
        print(f"  - {f}")
    if len(fixes) > 10:
        print(f"  ... et {len(fixes) - 10} autres")
    
    # 2. Vérification
    print()
    issues = verify_fixes()
    
    if issues:
        print(f"❌ {len(issues)} problème(s) restant(s):")
        for issue in issues[:10]:
            print(f"  {issue}")
    else:
        print("✅ Aucun problème détecté!")
    
    print("="*60)
