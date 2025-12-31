# 🎯 ADAN REWARD FUNCTION - COMPLETE ANALYSIS

**Date**: 2025-12-13  
**Status**: Comprehensive Analysis of Training Reward Mechanisms  
**Purpose**: Understanding why models saturate at BUY action

---

## EXECUTIVE SUMMARY

The ADAN reward function is a **multi-objective optimization system** that combines:
1. **PnL-based rewards** (25% weight)
2. **Risk-adjusted metrics** (Sharpe, Sortino, Calmar ratios - 75% weight)
3. **Behavioral penalties** (inaction, duration, invalid trades)
4. **Adaptive bonuses** (chunk performance, Kelly criterion, risk parity)

**Critical Issue**: The reward function heavily incentivizes **BUY actions** during bull markets, leading to saturation.

---

## PART 1: CORE REWARD FORMULA

### Main Reward Calculation (RewardCalculator.calculate())

```python
# STEP 1: Base Reward from PnL
base_reward = (trade_pnl - commission_penalty) * pnl_multiplier

# STEP 2: Commission Penalty
commission_penalty = commission * commission_penalty_multiplier  # Default: 1.5x
if trade_pnl > 0 and trade_pnl < min_profit_multiplier * commission:
    drawdown_penalty = (min_profit_multiplier - trade_pnl) * 2
    base_reward -= drawdown_penalty

# STEP 3: Chunk Bonus (if performance >= 80% of optimal)
chunk_bonus = 0.0
if performance_ratio >= performance_threshold (0.8):
    chunk_bonus = optimal_trade_bonus * (performance_ratio - 0.8)

# STEP 4: Multi-Objective Composite Score (if returns_history >= 5)
composite_score = (
    0.25 * base_reward                    # PnL component
    + 0.30 * sharpe_ratio                 # Risk-adjusted return
    + 0.30 * sortino_ratio                # Downside risk
    + 0.15 * calmar_ratio                 # Drawdown-adjusted
    + chunk_bonus                         # Performance bonus
    + kelly_bonus                         # Kelly criterion bonus
    + risk_parity_bonus                   # Risk parity bonus
    + stress_var_penalty                  # Stress VaR penalty
)

# STEP 5: Drawdown Penalty
if drawdown < -0.05:
    drawdown_penalty = abs(drawdown) * 10
    composite_score -= drawdown_penalty

# STEP 6: Inaction Penalty
if action == 0 and not is_hunting:
    inaction_penalty = -0.0001
    composite_score += inaction_penalty

# STEP 7: Duration Penalty
if trade_reason == "MaxDuration":
    duration_penalty = -1.0 * (1.0 - risk_horizon)
    composite_score += duration_penalty

# STEP 8: Tutor Bonus (one-time discovery bonus)
final_reward += tutor_bonus

# STEP 9: Clip to Range
final_reward = clip(final_reward, -5.0, 5.0)
```

---

## PART 2: DETAILED REWARD COMPONENTS

### 1. **PnL-Based Reward** (25% weight)

**Formula**:
```
base_reward = (realized_pnl - commission_penalty) * pnl_multiplier

where:
  realized_pnl = profit/loss from closed trade
  commission_penalty = commission * 1.5  (penalizes trading costs)
  pnl_multiplier = 1.0 (default)
```

**Example**:
```
Trade PnL: +$10
Commission: $1
commission_penalty = $1 * 1.5 = $1.50
base_reward = ($10 - $1.50) * 1.0 = $8.50
```

**Minimum Profit Threshold**:
```
min_profit = min_profit_multiplier * commission  # Default: 3.0x
if trade_pnl > 0 and trade_pnl < min_profit:
    drawdown_penalty = (min_profit - trade_pnl) * 2
    base_reward -= drawdown_penalty

Example:
  Commission: $1
  min_profit = 3.0 * $1 = $3
  If trade_pnl = $2 (less than $3):
    drawdown_penalty = ($3 - $2) * 2 = $2
    base_reward -= $2
```

### 2. **Sharpe Ratio** (30% weight)

**Formula**:
```
Sharpe = (weighted_mean_return - risk_free_rate) / weighted_std * sqrt(365)

where:
  weighted_mean_return = Σ(return_i * weight_i)
  weight_i = decay_factor^i  (exponential decay, recent = 1.0)
  weighted_std = sqrt(Σ((return_i - mean)^2 * weight_i))
  risk_free_rate = 0.01 (1% annual)
  365 = annual trading days (crypto 24/7)
```

**Time Weighting**:
```
weights = [0.99^n, 0.99^(n-1), ..., 0.99^1, 1.0]
Normalized: weights / sum(weights)

Effect: Recent returns weighted 100x more than old returns
```

**Example**:
```
Last 5 returns: [0.5%, 0.3%, -0.2%, 0.4%, 0.6%]
Weights: [0.95, 0.96, 0.97, 0.98, 1.0] (normalized)

weighted_mean = 0.5*0.95 + 0.3*0.96 + (-0.2)*0.97 + 0.4*0.98 + 0.6*1.0
              = 0.475 + 0.288 - 0.194 + 0.392 + 0.6
              = 1.561% (weighted)

weighted_std = sqrt(variance of weighted returns)
Sharpe = (1.561% - 0.01%/365) / weighted_std * sqrt(365)
```

### 3. **Sortino Ratio** (30% weight)

**Formula**:
```
Sortino = (weighted_mean_return - risk_free_rate) / downside_std * sqrt(365)

where:
  downside_std = sqrt(Σ(min(return_i, 0)^2 * weight_i))
  Only negative returns contribute to downside risk
```

**Key Difference from Sharpe**:
- Sharpe penalizes ALL volatility (up and down)
- Sortino only penalizes DOWNSIDE volatility (losses)

**Example**:
```
Returns: [+1%, +2%, -0.5%, +1.5%, -0.2%]

Downside returns: [0, 0, -0.5%, 0, -0.2%]
downside_std = sqrt(weighted variance of downside)

Sortino = (mean - rf) / downside_std * sqrt(365)
Higher Sortino = better risk-adjusted returns with less downside
```

### 4. **Calmar Ratio** (15% weight)

**Formula**:
```
Calmar = annualized_return / max_drawdown

where:
  annualized_return = (1 + cumulative_return)^365 - 1
  max_drawdown = maximum peak-to-trough decline
```

**Example**:
```
Cumulative return over 36 months: +15%
Max drawdown: -5%

annualized_return = (1 + 0.15)^(365/1095) - 1 ≈ 4.7%
Calmar = 4.7% / 5% = 0.94
```

---

## PART 3: BEHAVIORAL PENALTIES & BONUSES

### Inaction Penalty

**Formula**:
```
if action == 0 (HOLD) and not is_hunting:
    inaction_penalty = -0.0001
    reward += inaction_penalty
```

**Effect**: Discourages the model from doing nothing
- Very small penalty (-0.0001)
- Only applies when NOT in "hunting" mode
- Encourages exploration of BUY/SELL actions

### Duration Penalty

**Formula**:
```
if trade_reason == "MaxDuration":
    duration_penalty = -1.0 * (1.0 - risk_horizon)
    reward += duration_penalty

where:
  risk_horizon = time_remaining / max_trade_duration
```

**Example**:
```
Max trade duration: 100 steps
Trade closed at step 95:
  risk_horizon = 5 / 100 = 0.05
  duration_penalty = -1.0 * (1.0 - 0.05) = -0.95

Trade closed at step 50:
  risk_horizon = 50 / 100 = 0.50
  duration_penalty = -1.0 * (1.0 - 0.50) = -0.50
```

**Effect**: Penalizes trades that run to maximum duration

### Drawdown Penalty

**Formula**:
```
if drawdown < -0.05:
    drawdown_penalty = abs(drawdown) * 10
    composite_score -= drawdown_penalty
```

**Example**:
```
Drawdown: -10%
drawdown_penalty = 0.10 * 10 = 1.0
reward -= 1.0

Drawdown: -5%
drawdown_penalty = 0.05 * 10 = 0.5
reward -= 0.5
```

### Chunk Performance Bonus

**Formula**:
```
if performance_ratio >= 0.8:
    chunk_bonus = optimal_trade_bonus * (performance_ratio - 0.8)
    
where:
  performance_ratio = actual_pnl / optimal_pnl
  optimal_trade_bonus = 1.0 (default)
```

**Example**:
```
Optimal PnL for chunk: $100
Actual PnL: $90
performance_ratio = 90 / 100 = 0.90

chunk_bonus = 1.0 * (0.90 - 0.80) = 0.10
reward += 0.10
```

### Kelly Criterion Bonus

**Formula**:
```
if kelly_respected:
    kelly_bonus = kelly_bonus_weight  # 0.1
    reward += kelly_bonus
```

**Kelly Criterion**:
```
f* = (bp - q) / b

where:
  f* = fraction of capital to bet
  b = odds (profit/loss ratio)
  p = probability of win
  q = probability of loss (1 - p)
```

### Risk Parity Bonus

**Formula**:
```
if risk_respected:
    risk_parity_bonus = risk_parity_bonus_weight  # 0.05
    reward += risk_parity_bonus
```

**Risk Parity**: Each position contributes equally to portfolio risk

### Stress VaR Penalty

**Formula**:
```
stress_var_99 = portfolio_metrics.get("stress_var_0.99", 0.0)
var_threshold = 0.1  # 10%

if stress_var_99 > var_threshold:
    excess_var = stress_var_99 - var_threshold
    stress_var_penalty = -stress_var_penalty_weight * excess_var
    reward += stress_var_penalty
```

---

## PART 4: MULTI-ASSET ENVIRONMENT REWARD

### Additional Components in Multi-Asset Environment

```python
# 1. Frequency Reward
frequency_reward = _calculate_frequency_reward()
# Bonus for maintaining optimal trade frequency

# 2. Position Limit Penalty
pos_limit_penalty = calculate_position_limit_penalty()
# Penalty for exceeding position limits

# 3. Outcome Reward
outcome_reward = calculate_outcome_reward()
# Reward for successful trade outcomes

# 4. Early Close Bonus
early_close_bonus = calculate_early_close_bonus()
# Bonus for closing trades early with profit

# 5. Invalid Trade Penalty
invalid_trade_penalty = -invalid_trade_penalty_weight * invalid_trade_attempts
# Penalty for attempting invalid trades

# 6. Multi-Track Bonus
multi_bonus = 0.0
if current_equity > initial_capital * 1.5 and open_trades_count >= 2:
    multi_bonus = multi_traque_bonus_value
# Bonus for managing multiple positions profitably
```

---

## PART 5: WEIGHT CONFIGURATION

### Default Weights (Balanced)

```yaml
reward_shaping:
  weights:
    pnl: 0.25        # 25% - PnL component
    sharpe: 0.30     # 30% - Risk-adjusted return
    sortino: 0.30    # 30% - Downside risk
    calmar: 0.15     # 15% - Drawdown-adjusted
```

### Worker-Specific Profiles

```yaml
reward_shaping:
  profiles:
    Conservative:
      pnl_weight: 0.5
      missed_opportunity_penalty: -0.05
      multi_traque_bonus: 0.1
    
    Moderate:
      pnl_weight: 1.0
      missed_opportunity_penalty: -0.1
      multi_traque_bonus: 0.2
    
    Aggressive:
      pnl_weight: 2.0
      missed_opportunity_penalty: -0.2
      multi_traque_bonus: 0.5
```

---

## PART 6: WHY MODELS SATURATE AT BUY

### Root Cause Analysis

**Problem**: All models return 1.0 (BUY) always

**Contributing Factors**:

1. **Bull Market Training Data**
   - Training period: Likely bull market
   - BUY action = +reward (market goes up)
   - SELL action = -reward (market goes up, you miss gains)
   - HOLD action = -reward (inaction penalty)
   - **Result**: Model learns BUY is always optimal

2. **Inaction Penalty**
   ```
   if action == 0 (HOLD):
       reward -= 0.0001
   ```
   - Discourages HOLD
   - Encourages BUY/SELL
   - In bull market: BUY is rewarded, SELL is punished
   - **Result**: Model converges to BUY

3. **PnL-Dominant Reward**
   ```
   composite_score = 0.25 * base_reward + ...
   ```
   - 25% weight on PnL
   - In bull market: BUY = +PnL
   - Risk metrics (Sharpe, Sortino) also favor BUY in bull market
   - **Result**: All components reward BUY

4. **Insufficient Exploration**
   - No entropy bonus to encourage exploration
   - No forced diversity in training
   - Model converges to local optimum (always BUY)
   - **Result**: Saturation at 1.0

5. **No Negative Reward for BUY in Bull Market**
   - BUY in bull market = profit
   - No penalty for over-trading
   - No penalty for concentration risk
   - **Result**: Model learns to always BUY

---

## PART 7: PROPOSED FIXES

### Fix 1: Rebalance Reward Weights

**Current**:
```yaml
weights:
  pnl: 0.25
  sharpe: 0.30
  sortino: 0.30
  calmar: 0.15
```

**Proposed**:
```yaml
weights:
  pnl: 0.15        # Reduce PnL weight
  sharpe: 0.30
  sortino: 0.30
  calmar: 0.15
  diversity: 0.10  # Add diversity bonus
```

### Fix 2: Add Entropy Bonus

```python
# Encourage exploration
entropy_bonus = -entropy_coefficient * action_entropy
# Higher entropy (more diverse actions) = higher bonus
```

### Fix 3: Add Action Diversity Penalty

```python
# Penalize repetitive actions
if last_action == current_action:
    diversity_penalty = -0.1
    reward += diversity_penalty
```

### Fix 4: Balanced Training Data

```python
# Ensure equal distribution of market regimes
training_data = {
    'bull_market': 33%,
    'bear_market': 33%,
    'sideways_market': 34%
}
```

### Fix 5: Add Forced Diversity in Ensemble

```python
# Force at least one worker to vote differently
if all_workers_vote_same_action:
    override_random_worker()
    recalculate_consensus()
```

---

## PART 8: EQUATIONS SUMMARY TABLE

| Component | Formula | Weight | Purpose |
|-----------|---------|--------|---------|
| **Base PnL** | `(pnl - commission) * multiplier` | 25% | Profit incentive |
| **Sharpe Ratio** | `(mean_return - rf) / std * sqrt(365)` | 30% | Risk-adjusted return |
| **Sortino Ratio** | `(mean_return - rf) / downside_std * sqrt(365)` | 30% | Downside risk focus |
| **Calmar Ratio** | `annualized_return / max_drawdown` | 15% | Drawdown-adjusted |
| **Inaction Penalty** | `-0.0001` if HOLD | - | Discourage inaction |
| **Duration Penalty** | `-1.0 * (1 - risk_horizon)` | - | Penalize long trades |
| **Drawdown Penalty** | `-abs(drawdown) * 10` if < -5% | - | Penalize large losses |
| **Chunk Bonus** | `optimal_bonus * (perf_ratio - 0.8)` | - | Reward performance |
| **Kelly Bonus** | `+0.1` if respected | - | Reward position sizing |
| **Risk Parity Bonus** | `+0.05` if respected | - | Reward balanced risk |
| **Stress VaR Penalty** | `-0.15 * excess_var` | - | Penalize tail risk |

---

## PART 9: CONFIGURATION PARAMETERS

### Key Parameters in config.yaml

```yaml
reward_shaping:
  # PnL parameters
  realized_pnl_multiplier: 1.0
  unrealized_pnl_multiplier: 0.1
  commission_penalty: 1.5
  min_profit_multiplier: 3.0
  
  # Chunk parameters
  optimal_trade_bonus: 1.0
  performance_threshold: 0.8
  
  # Behavioral parameters
  inaction_penalty: -0.0001
  reward_clipping_range: [-5.0, 5.0]
  
  # Risk management
  kelly_bonus_weight: 0.1
  risk_parity_bonus_weight: 0.05
  stress_var_penalty_weight: 0.15
  
  # Exploration tutor
  exploration_tutor:
    enabled: false
    discovery_bonus: 0.0
    exit_on_successful_trades: {}
```

---

## CONCLUSION

The ADAN reward function is sophisticated but **biased towards BUY actions in bull markets**. The saturation at 1.0 is a direct result of:

1. ✅ Training on bull market data
2. ✅ Inaction penalty discouraging HOLD
3. ✅ PnL-dominant reward in bull market
4. ✅ Insufficient exploration mechanisms
5. ✅ No forced diversity

**Solution**: Implement the emergency patch (noise injection + forced diversity) as temporary fix, then retrain with:
- Balanced market regime data
- Entropy bonus for exploration
- Action diversity penalties
- Reduced PnL weight
- Forced voting diversity in ensemble

---

**Last Updated**: 2025-12-13 18:30:00 UTC  
**Related Documents**:
- `CRITICAL_MODEL_SATURATION_REPORT.md`
- `SATURATION_FIX_SUMMARY.md`
- `EMERGENCY_ACTION_PLAN.md`
