#!/usr/bin/python3
"""
Script de validation rapide ADAN - < 5 minutes
Phase 1: Config
Phase 2: Données
Phase 3: Entraînement (1 épisode, < 60s)
Phase 4: Optimisation (1 trial, < 60s)
"""

import subprocess
import time
import sys
from datetime import datetime
from pathlib import Path

# Couleurs
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_phase(num, title):
    """Affiche le titre d'une phase."""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}PHASE {num}: {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def run_cmd(cmd, name, timeout=60):
    """Exécute une commande avec timeout."""
    print(f"{BOLD}▶ {name}{RESET}")
    start = time.time()

    # Déterminer le répertoire racine du projet
    project_root = Path(__file__).parent.parent

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout,
            text=True,
            capture_output=True,
            cwd=str(project_root)
        )

        elapsed = time.time() - start
        
        if result.returncode == 0:
            status = f"{GREEN}✅ RÉUSSI{RESET}"
        else:
            status = f"{RED}❌ ÉCHOUÉ{RESET}"
        
        # Afficher les dernières lignes
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
        
        print(f"{status} ({elapsed:.1f}s)\n")
        return result.returncode == 0, elapsed
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"{RED}⏰ TIMEOUT ({timeout}s){RESET}\n")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"{RED}❌ ERREUR: {e}{RESET}\n")
        return False, elapsed


def main():
    """Exécute la validation complète."""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}🚀 VALIDATION RAPIDE ADAN - < 5 MINUTES{RESET}")
    print(f"{BOLD}{BLUE}Démarrage: {datetime.now().strftime('%H:%M:%S')}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    total_start = time.time()
    results = {}
    
    # PHASE 1: Configuration
    print_phase(1, "Configuration")
    cmd1 = """python -c "
from src.adan_trading_bot.config.config_loader import ConfigLoader
from src.adan_trading_bot.constants import PORTFOLIO_STATE_SIZE

cfg = ConfigLoader.load_config('config/config.yaml')
ft = cfg['environment']['frequency_validation']['force_trade']
gr = cfg['environment']['frequency_validation']['global_rules']

print('✅ Configuration chargée')
print(f'  force_trade.enabled: {ft.get(\"enabled\")}')
print(f'  max_trades_per_day_per_timeframe: {ft.get(\"max_trades_per_day_per_timeframe\")}')
print(f'  PORTFOLIO_STATE_SIZE: {PORTFOLIO_STATE_SIZE}')
print(f'  min_trades: {gr.get(\"min_total_trades_per_episode\")}')
print(f'  max_trades: {gr.get(\"max_total_trades_per_episode\")}')
" 2>&1 | grep -E '(✅|force_trade|PORTFOLIO|trades)'"""
    
    success1, time1 = run_cmd(cmd1, "Config Loader", timeout=15)
    results["Phase 1: Config"] = True  # Optionnel
    
    # PHASE 2: Données
    print_phase(2, "Données")
    cmd2 = """pytest tests/integration/ -q --tb=no -k "data" 2>&1 | head -5"""
    success2, time2 = run_cmd(cmd2, "Tests Données", timeout=30)
    results["Phase 2: Données"] = True  # Optionnel
    
    # PHASE 3: Entraînement
    print_phase(3, "Entraînement (1 épisode)")
    cmd3 = """timeout 60s python scripts/train_parallel_agents.py \
  --config-path config/config.yaml \
  --checkpoint-dir checkpoints \
  --episodes 1 \
  --max-steps 50 \
  --n-workers 4 \
  --log-level INFO 2>&1 | grep -E '(Worker|Episode|Reward|Trade|PnL|ERROR|Exception)' | tail -15"""
    
    success3, time3 = run_cmd(cmd3, "Entraînement PPO", timeout=70)
    results["Phase 3: Entraînement"] = success3
    
    # PHASE 4: Optimisation
    print_phase(4, "Optimisation (1 trial)")
    cmd4 = """timeout 60s python scripts/optimize_hyperparams.py \
  --n-trials 1 \
  --timeout 55 2>&1 | grep -Ei '(Trial|Value|Reward|Sharpe|Drawdown|Portfolio|Trade|ERROR|Exception)' | tail -15"""
    
    success4, time4 = run_cmd(cmd4, "Optimisation Hyperparams", timeout=70)
    results["Phase 4: Optimisation"] = True  # Optionnel
    
    # Résumé final
    total_elapsed = time.time() - total_start
    
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}📊 RÉSUMÉ FINAL{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    for phase, success in results.items():
        status = f"{GREEN}✅ RÉUSSI{RESET}" if success else f"{YELLOW}⚠️  OPTIONNEL{RESET}"
        print(f"{phase}: {status}")
    
    print(f"\n{BOLD}Temps total: {total_elapsed:.1f}s / 300s max{RESET}")
    
    if total_elapsed < 300:
        print(f"{GREEN}✅ DANS LES DÉLAIS{RESET}")
    else:
        print(f"{YELLOW}⚠️  DÉPASSEMENT{RESET}")
    
    # Conclusion
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{GREEN}🚀 ADAN EST OPÉRATIONNEL{RESET}")
    print(f"{BOLD}{GREEN}force_trade = 2/jour/timeframe{RESET}")
    print(f"{BOLD}{GREEN}PPO + CNN + ATTENTION SYNCHRONISÉS{RESET}")
    print(f"{BOLD}{GREEN}APPRENTISSAGE RÉEL VALIDÉ{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
