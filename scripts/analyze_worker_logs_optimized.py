import re
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

W2_LOG = "/home/morningstar/Documents/trading/bot/logs/workers/optuna_w2_20251109_223833.log"
W3_LOG = "/home/morningstar/Documents/trading/bot/logs/workers/optuna_w3_20251109_223837.log"

@dataclass
class TrialResult:
    trial_number: int
    sharpe_ratio: float = 0.0
    pnl: float = 0.0
    final_balance: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    
    # Hyperparamètres PPO
    learning_rate: float = 0.0
    batch_size: int = 0
    n_steps: int = 0
    gamma: float = 0.0
    clip_range: float = 0.0
    ent_coef: float = 0.0
    n_epochs: int = 0
    
    # Risk Management
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Trading Rules
    max_trades_5m: int = 0
    max_trades_1h: int = 0
    max_trades_4h: int = 0
    patience: int = 0
    min_confidence: float = 0.0
    risk_mult: float = 0.0
    
    # Reward Weights
    pnl_weight: float = 0.0
    win_rate_bonus: float = 0.0
    take_profit_bonus: float = 0.0
    stop_loss_penalty: float = 0.0
    consecutive_loss_penalty: float = 0.0
    
    status: str = "unknown"

def extract_trials_streaming(filepath: str, worker_name: str) -> List[TrialResult]:
    """Extrait tous les trials en streaming sans charger le fichier entier"""
    
    print(f"\n{'='*80}")
    print(f"🔍 EXTRACTION {worker_name}")
    print(f"{'='*80}\n")
    
    trials = {}
    current_trial = None
    lines_processed = 0
    
    # Patterns de détection
    trial_start_pattern = re.compile(r'Trial (\d+)')
    trial_finish_pattern = re.compile(r'Trial (\d+) finished')
    
    # Patterns pour métriques
    sharpe_pattern = re.compile(r'Sharpe[:\s]+([+-]?\d+\.?\d*)')
    pnl_pattern = re.compile(r'PnL[:\s]+\$?([+-]?\d+\.?\d*)')
    balance_pattern = re.compile(r'(?:Final |)Balance[:\s]+\$?(\d+\.?\d*)')
    trades_pattern = re.compile(r'(?:Total |)Trades[:\s]+(\d+)')
    win_rate_pattern = re.compile(r'Win Rate[:\s]+(\d+\.?\d*)%?')
    drawdown_pattern = re.compile(r'(?:Max |)Drawdown[:\s]+([+-]?\d+\.?\d*)%?')
    
    # Patterns pour hyperparamètres
    lr_pattern = re.compile(r'learning_rate[:\s]+([0-9.e-]+)')
    batch_pattern = re.compile(r'batch_size[:\s]+(\d+)')
    nsteps_pattern = re.compile(r'n_steps[:\s]+(\d+)')
    gamma_pattern = re.compile(r'gamma[:\s]+([0-9.]+)')
    clip_pattern = re.compile(r'clip_range[:\s]+([0-9.]+)')
    ent_pattern = re.compile(r'ent_coef[:\s]+([0-9.]+)')
    epochs_pattern = re.compile(r'n_epochs[:\s]+(\d+)')
    
    # Risk management
    pos_size_pattern = re.compile(r'position_size[:\s]+([0-9.]+)%?')
    sl_pattern = re.compile(r'stop_loss(?:_pct)?[:\s]+([0-9.]+)%?')
    tp_pattern = re.compile(r'take_profit(?:_pct)?[:\s]+([0-9.]+)%?')
    
    # Trading rules
    max_trades_pattern = re.compile(r'max_trades.*?5m[:\s]+(\d+).*?1h[:\s]+(\d+).*?4h[:\s]+(\d+)', re.IGNORECASE)
    patience_pattern = re.compile(r'patience[:\s]+(\d+)')
    confidence_pattern = re.compile(r'min_confidence[:\s]+([0-9.]+)')
    risk_mult_pattern = re.compile(r'risk_mult[:\s]+([0-9.]+)')
    
    # Reward weights
    pnl_weight_pattern = re.compile(r'pnl_weight[:\s]+([0-9.]+)')
    win_bonus_pattern = re.compile(r'win_rate_bonus[:\s]+([+-]?[0-9.]+)')
    tp_bonus_pattern = re.compile(r'take_profit_bonus[:\s]+([+-]?[0-9.]+)')
    sl_penalty_pattern = re.compile(r'stop_loss_penalty[:\s]+([+-]?[0-9.]+)')
    loss_penalty_pattern = re.compile(r'consecutive_loss_penalty[:\s]+([+-]?[0-9.]+)')
    
    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            lines_processed += 1
            
            if lines_processed % 1000000 == 0:
                print(f"  📊 Traité: {lines_processed:,} lignes - Trials: {len(trials)}")
            
            # Détecter nouveau trial
            match = trial_start_pattern.search(line)
            if match:
                trial_num = int(match.group(1))
                if trial_num not in trials:
                    trials[trial_num] = TrialResult(trial_number=trial_num)
                current_trial = trial_num
                continue
            
            if current_trial is None:
                continue
            
            trial = trials[current_trial]
            
            # Status
            if trial_finish_pattern.search(line):
                trial.status = "finished"
            
            # Extraire métriques
            if match := sharpe_pattern.search(line):
                trial.sharpe_ratio = float(match.group(1))
            elif match := pnl_pattern.search(line):
                trial.pnl = float(match.group(1))
            elif match := balance_pattern.search(line):
                trial.final_balance = float(match.group(1))
            elif match := trades_pattern.search(line):
                trial.total_trades = int(match.group(1))
            elif match := win_rate_pattern.search(line):
                trial.win_rate = float(match.group(1))
            elif match := drawdown_pattern.search(line):
                trial.max_drawdown = float(match.group(1))
            
            # Hyperparamètres PPO
            elif match := lr_pattern.search(line):
                trial.learning_rate = float(match.group(1))
            elif match := batch_pattern.search(line):
                trial.batch_size = int(match.group(1))
            elif match := nsteps_pattern.search(line):
                trial.n_steps = int(match.group(1))
            elif match := gamma_pattern.search(line):
                trial.gamma = float(match.group(1))
            elif match := clip_pattern.search(line):
                trial.clip_range = float(match.group(1))
            elif match := ent_pattern.search(line):
                trial.ent_coef = float(match.group(1))
            elif match := epochs_pattern.search(line):
                trial.n_epochs = int(match.group(1))
            
            # Risk Management
            elif match := pos_size_pattern.search(line):
                trial.position_size = float(match.group(1))
            elif match := sl_pattern.search(line):
                trial.stop_loss = float(match.group(1))
            elif match := tp_pattern.search(line):
                trial.take_profit = float(match.group(1))
            
            # Trading Rules
            elif match := max_trades_pattern.search(line):
                trial.max_trades_5m = int(match.group(1))
                trial.max_trades_1h = int(match.group(2))
                trial.max_trades_4h = int(match.group(3))
            elif match := patience_pattern.search(line):
                trial.patience = int(match.group(1))
            elif match := confidence_pattern.search(line):
                trial.min_confidence = float(match.group(1))
            elif match := risk_mult_pattern.search(line):
                trial.risk_mult = float(match.group(1))
            
            # Reward Weights
            elif match := pnl_weight_pattern.search(line):
                trial.pnl_weight = float(match.group(1))
            elif match := win_bonus_pattern.search(line):
                trial.win_rate_bonus = float(match.group(1))
            elif match := tp_bonus_pattern.search(line):
                trial.take_profit_bonus = float(match.group(1))
            elif match := sl_penalty_pattern.search(line):
                trial.stop_loss_penalty = float(match.group(1))
            elif match := loss_penalty_pattern.search(line):
                trial.consecutive_loss_penalty = float(match.group(1))
    
    print(f"  ✅ Extraction terminée: {lines_processed:,} lignes traitées")
    print(f"  📊 {len(trials)} trials détectés\n")
    
    return sorted(trials.values(), key=lambda x: x.trial_number)

def display_top_trials(trials: List[TrialResult], worker_name: str, top_n: int = 10):
    """Affiche les meilleurs trials par Sharpe Ratio"""
    
    finished_trials = [t for t in trials if t.status == "finished" and t.sharpe_ratio > 0]
    
    if not finished_trials:
        print(f"⚠️ Aucun trial fini trouvé pour {worker_name}")
        return
    
    top_trials = sorted(finished_trials, key=lambda x: x.sharpe_ratio, reverse=True)[:top_n]
    
    print(f"\n{'='*80}")
    print(f"🏆 TOP {top_n} TRIALS - {worker_name} (triés par Sharpe Ratio)")
    print(f"{'='*80}\n")
    
    for i, trial in enumerate(top_trials, 1):
        print(f"{'─'*80}")
        print(f"#{i} - Trial {trial.trial_number} | Sharpe: {trial.sharpe_ratio:.4f} | Status: {trial.status}")
        print(f"{'─'*80}")
        
        print(f"\n📊 PERFORMANCES:")
        print(f"  • PnL: ${trial.pnl:.2f}")
        print(f"  • Balance Finale: ${trial.final_balance:.2f}")
        print(f"  • Total Trades: {trial.total_trades}")
        print(f"  • Win Rate: {trial.win_rate:.2f}%")
        print(f"  • Max Drawdown: {trial.max_drawdown:.2f}%")
        
        print(f"\n⚙️ HYPERPARAMÈTRES PPO:")
        print(f"  • Learning Rate: {trial.learning_rate:.6f}")
        print(f"  • Batch Size: {trial.batch_size}")
        print(f"  • N Steps: {trial.n_steps}")
        print(f"  • Gamma: {trial.gamma:.4f}")
        print(f"  • Clip Range: {trial.clip_range:.4f}")
        print(f"  • Entropy Coef: {trial.ent_coef:.4f}")
        print(f"  • N Epochs: {trial.n_epochs}")
        
        print(f"\n💰 RISK MANAGEMENT:")
        print(f"  • Position Size: {trial.position_size:.2f}%")
        print(f"  • Stop Loss: {trial.stop_loss:.2f}%")
        print(f"  • Take Profit: {trial.take_profit:.2f}%")
        
        print(f"\n📈 TRADING RULES:")
        print(f"  • Max Trades: 5m={trial.max_trades_5m} | 1h={trial.max_trades_1h} | 4h={trial.max_trades_4h}")
        print(f"  • Patience: {trial.patience}")
        print(f"  • Min Confidence: {trial.min_confidence:.4f}")
        print(f"  • Risk Multiplier: {trial.risk_mult:.4f}")
        
        print(f"\n🎯 REWARD WEIGHTS:")
        print(f"  • PnL Weight: {trial.pnl_weight:.2f}")
        print(f"  • Win Rate Bonus: {trial.win_rate_bonus:.4f}")
        print(f"  • Take Profit Bonus: {trial.take_profit_bonus:.4f}")
        print(f"  • Stop Loss Penalty: {trial.stop_loss_penalty:.4f}")
        print(f"  • Consecutive Loss Penalty: {trial.consecutive_loss_penalty:.4f}")
        print()

def export_to_json(trials: List[TrialResult], worker_name: str, filename: str):
    """Exporte tous les trials en JSON"""
    data = {
        "worker": worker_name,
        "total_trials": len(trials),
        "finished_trials": sum(1 for t in trials if t.status == "finished"),
        "trials": [asdict(t) for t in trials]
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Exporté vers: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("╔════════════════════════════════════════════════════════════════╗")
print("║    📊 EXTRACTION COMPLÈTE HYPERPARAMÈTRES W2 & W3           ║")
print("╚════════════════════════════════════════════════════════════════╝")

# Extraire W2
w2_trials = extract_trials_streaming(W2_LOG, "W2")
display_top_trials(w2_trials, "W2", top_n=10)
export_to_json(w2_trials, "W2", "w2_trials_complete.json")

print("\n" + "="*80 + "\n")

# Extraire W3
w3_trials = extract_trials_streaming(W3_LOG, "W3")
display_top_trials(w3_trials, "W3", top_n=10)
export_to_json(w3_trials, "W3", "w3_trials_complete.json")

# Statistiques globales
print(f"\n{'='*80}")
print(f"📊 STATISTIQUES GLOBALES")
print(f"{'='*80}")
print(f"\nW2:")
print(f"  • Total trials: {len(w2_trials)}")
print(f"  • Trials terminés: {sum(1 for t in w2_trials if t.status == 'finished')}")
print(f"  • Meilleur Sharpe: {max((t.sharpe_ratio for t in w2_trials if t.sharpe_ratio > 0), default=0):.4f}")

print(f"\nW3:")
print(f"  • Total trials: {len(w3_trials)}")
print(f"  • Trials terminés: {sum(1 for t in w3_trials if t.status == 'finished')}")
print(f"  • Meilleur Sharpe: {max((t.sharpe_ratio for t in w3_trials if t.sharpe_ratio > 0), default=0):.4f}")

print(f"\n✅ Extraction terminée! Fichiers JSON générés.")
