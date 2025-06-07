#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script wrapper pour lancer train_rl_agent.py sans Rich.
Ce script désactive Rich pour éviter les problèmes d'affichage dans certains terminaux.
"""
import os
import sys
import subprocess

def main():
    # Désactiver Rich via les variables d'environnement
    env = os.environ.copy()
    env['NO_COLOR'] = '1'
    env['TERM'] = 'dumb'
    env['FORCE_COLOR'] = '0'
    env['RICH_FORCE_TERMINAL'] = '0'
    
    # Construire la commande pour lancer train_rl_agent.py
    cmd = [sys.executable, 'scripts/train_rl_agent.py'] + sys.argv[1:]
    
    # Lancer le script avec les variables d'environnement modifiées
    try:
        result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nEntraînement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du lancement de l'entraînement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()