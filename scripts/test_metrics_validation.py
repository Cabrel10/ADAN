#!/usr/bin/env python3
"""
Test de validation des métriques et de la configuration
Vérifie que config.yaml est valide avant entraînement
"""

import sys
import yaml
from pathlib import Path

def validate_config(config_path):
    """Valide la configuration avant entraînement"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return [f"Erreur lecture config: {e}"], []
    
    errors = []
    warnings = []
    
    # Vérifier structure de base
    if 'workers' not in config:
        errors.append("❌ Pas de section 'workers' dans config")
        return errors, warnings
    
    # Vérifier chaque worker
    for worker_name, worker_config in config.get('workers', {}).items():
        if not isinstance(worker_config, dict):
            errors.append(f"❌ {worker_name}: Configuration invalide (pas dict)")
            continue
        
        # Vérifier paramètres PPO
        ppo = worker_config.get('agent_config', {})
        if not ppo:
            errors.append(f"❌ {worker_name}: Pas de 'agent_config' (PPO params)")
            continue
        
        # Vérifier learning_rate
        lr = ppo.get('learning_rate')
        if lr is None:
            errors.append(f"❌ {worker_name}: learning_rate manquant")
        elif not isinstance(lr, (int, float)):
            errors.append(f"❌ {worker_name}: learning_rate invalide: {lr}")
        elif lr < 1e-7 or lr > 1e-2:
            warnings.append(f"⚠️  {worker_name}: learning_rate {lr:.2e} hors plage typique [1e-7, 1e-2]")
        
        # Vérifier batch_size
        bs = ppo.get('batch_size')
        if bs is None:
            errors.append(f"❌ {worker_name}: batch_size manquant")
        elif bs not in [32, 64, 128, 256]:
            warnings.append(f"⚠️  {worker_name}: batch_size {bs} inhabituel")
        
        # Vérifier n_steps
        ns = ppo.get('n_steps')
        if ns is None:
            errors.append(f"❌ {worker_name}: n_steps manquant")
        elif ns < 128 or ns > 4096:
            warnings.append(f"⚠️  {worker_name}: n_steps {ns} hors plage [128, 4096]")
        
        # Vérifier paramètres trading
        trading = worker_config.get('trading_parameters', {})
        if not trading:
            errors.append(f"❌ {worker_name}: Pas de 'trading_parameters'")
            continue
        
        # Vérifier SL/TP
        sl = trading.get('stop_loss_pct', 0)
        tp = trading.get('take_profit_pct', 0)
        
        if not isinstance(sl, (int, float)) or sl <= 0:
            errors.append(f"❌ {worker_name}: stop_loss_pct invalide: {sl}")
        
        if not isinstance(tp, (int, float)) or tp <= 0:
            errors.append(f"❌ {worker_name}: take_profit_pct invalide: {tp}")
        
        if sl > 0 and tp > 0 and tp <= sl:
            errors.append(f"❌ {worker_name}: TP ({tp:.4f}) <= SL ({sl:.4f}) - INVALIDE!")
        elif sl > 0 and tp > 0:
            ratio = tp / sl
            if ratio < 1.2:
                warnings.append(f"⚠️  {worker_name}: Ratio TP/SL faible ({ratio:.2f})")
        
        # Vérifier position size
        pos = trading.get('position_size_pct', 0)
        if not isinstance(pos, (int, float)) or pos <= 0 or pos > 1:
            errors.append(f"❌ {worker_name}: position_size_pct {pos} invalide (doit être 0-1)")
        
        # Vérifier risk_per_trade_pct
        risk = trading.get('risk_per_trade_pct', 0)
        if not isinstance(risk, (int, float)) or risk <= 0 or risk > 0.1:
            errors.append(f"❌ {worker_name}: risk_per_trade_pct {risk} invalide")
        
        # Vérifier max_concurrent_positions
        max_pos = trading.get('max_concurrent_positions', 0)
        if not isinstance(max_pos, int) or max_pos < 1 or max_pos > 10:
            errors.append(f"❌ {worker_name}: max_concurrent_positions {max_pos} invalide")
        
        # Vérifier min_holding_period_steps
        min_hold = trading.get('min_holding_period_steps', 0)
        if not isinstance(min_hold, int) or min_hold < 1:
            errors.append(f"❌ {worker_name}: min_holding_period_steps {min_hold} invalide")
        
        # Avertissements spécifiques par worker
        if worker_name == 'w3':
            if pos < 0.40:
                warnings.append(f"⚠️  W3: position_size_pct {pos} < 0.40 (recommandé pour agressif)")
            if min_hold > 100:
                warnings.append(f"⚠️  W3: min_holding_period_steps {min_hold} très long (recommandé < 80)")
    
    return errors, warnings


def print_validation_report(config_path):
    """Affiche un rapport de validation"""
    print("\n" + "="*70)
    print("🔍 VALIDATION DE CONFIGURATION")
    print("="*70)
    print(f"Config: {config_path}\n")
    
    errors, warnings = validate_config(config_path)
    
    if errors:
        print("❌ ERREURS CRITIQUES:")
        for error in errors:
            print(f"   {error}")
        print()
    
    if warnings:
        print("⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            print(f"   {warning}")
        print()
    
    if not errors and not warnings:
        print("✅ Configuration VALIDE - Aucune erreur détectée\n")
        return True
    elif not errors:
        print("✅ Configuration VALIDE - Avertissements seulement\n")
        return True
    else:
        print("❌ Configuration INVALIDE - Erreurs détectées\n")
        return False


if __name__ == '__main__':
    config_path = 'config/config.yaml'
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    is_valid = print_validation_report(config_path)
    sys.exit(0 if is_valid else 1)
