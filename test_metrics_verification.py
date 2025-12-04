#!/usr/bin/env python3
"""Test diagnostic pour vérifier Win Rate, Profit Factor, Calmar"""
import numpy as np
import yaml
from src.adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Force trades agressifs
config['trading_rules']['frequency']['force_trade_steps'] = {'5m': 30, '1h': 50, '4h': 100}
config['trading_rules']['frequency']['daily_forced_trades_limit'] = 15

env = MultiAssetChunkedEnv(config=config)
obs = env.reset()
np.random.seed(42)

print('=== VÉRIFICATION DÉTAILLÉE DES MÉTRIQUES ===\n')

portfolio_values = [env.portfolio.portfolio_value]
winning_pnl = 0.0
losing_pnl = 0.0
winning_count = 0
losing_count = 0
total_trades = 0
positions_opened = 0
positions_closed = 0

for step in range(500):
    action = np.random.normal(0, 0.1, env.action_space.shape)
    if step % 30 == 0 and step > 0:
        action[0] = 0.5 * np.random.choice([-1, 1])
    
    result = env.step(action)
    info = result[3] if len(result) == 4 else result[-1]
    
    portfolio_values.append(env.portfolio.portfolio_value)
    total_trades = info.get('trades', 0)
    
    # Compter positions actuelles
    num_pos = info.get('num_positions', 0)
    
    # Vérifier fermetures
    closed = info.get('closed_positions', [])
    if closed:
        positions_closed += len(closed)
        print(f'Step {step}: {len(closed)} positions fermées')
        for pos in closed:
            if hasattr(pos, 'realized_pnl'):
                pnl = pos.realized_pnl
                print(f'  - PnL réalisé: ${pnl:.4f}')
                if pnl > 0:
                    winning_pnl += pnl
                    winning_count += 1
                else:
                    losing_pnl += abs(pnl)
                    losing_count += 1
            elif isinstance(pos, dict) and 'realized_pnl' in pos:
                pnl = pos['realized_pnl']
                print(f'  - PnL réalisé (dict): ${pnl:.4f}')
                if pnl > 0:
                    winning_pnl += pnl
                    winning_count += 1
                else:
                    losing_pnl += abs(pnl)
                    losing_count += 1
            else:
                print(f'  - Type position: {type(pos)}, attrs: {dir(pos) if hasattr(pos, "__dict__") else pos}')

# Calculs finaux
returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
returns = returns[~np.isnan(returns)]

print(f'\n{"="*60}')
print(f'RÉSULTATS APRÈS 500 STEPS')
print(f'{"="*60}')
print(f'Portfolio: ${portfolio_values[0]:.2f} → ${portfolio_values[-1]:.2f}')
print(f'')
print(f'TRADES:')
print(f'  Trades total (compteur): {total_trades}')
print(f'  Positions fermées: {positions_closed}')
print(f'  Gagnantes: {winning_count}')
print(f'  Perdantes: {losing_count}')
print(f'')

if (winning_count + losing_count) > 0:
    win_rate = winning_count / (winning_count + losing_count) * 100
    print(f'WIN RATE: {win_rate:.1f}%')
else:
    print(f'WIN RATE: N/A (aucune position fermée)')
print(f'')

print(f'PnL:')
print(f'  Gross Profit: ${winning_pnl:.4f}')
print(f'  Gross Loss: ${losing_pnl:.4f}')
if losing_pnl > 0:
    print(f'  Profit Factor: {winning_pnl / losing_pnl:.2f}')
else:
    print(f'  Profit Factor: N/A')
print(f'')

# Sharpe
if np.std(returns) > 0:
    sharpe = np.sqrt(252*24) * np.mean(returns) / np.std(returns)
else:
    sharpe = 0
print(f'RATIOS:')
print(f'  Sharpe Ratio: {sharpe:.4f}')

# Calmar
max_dd = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values))
total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
calmar = total_return / max_dd if max_dd > 0 else 0
print(f'  Max Drawdown: {max_dd*100:.2f}%')
print(f'  Total Return: {total_return*100:.2f}%')
print(f'  Calmar Ratio: {calmar:.4f}')

print(f'\n{"="*60}')
print(f'DIAGNOSTIC:')
if positions_closed == 0:
    print(f'  ⚠️  AUCUNE POSITION FERMÉE!')
    print(f'  Raisons possibles:')
    print(f'    - TP/SL jamais atteints')
    print(f'    - Timeout positions non configuré')
    print(f'    - Positions restent ouvertes indéfiniment')
print(f'{"="*60}')
