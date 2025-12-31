# 📊 ADAN REWARD FUNCTION - VISUAL GUIDE

---

## REWARD CALCULATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: COLLECT INPUTS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  • realized_pnl: Profit/Loss from closed trade                 │
│  • commission: Trading fees paid                               │
│  • action: 0=HOLD, 1=BUY, 2=SELL                              │
│  • portfolio_metrics: Risk metrics, drawdown, etc.             │
│  • returns_history: Last N returns for ratio calculation       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 2: BASE REWARD (PnL)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  base_reward = (realized_pnl - commission_penalty) * multiplier│
│                                                                 │
│  Example:                                                       │
│    realized_pnl = +$10                                         │
│    commission = $1                                             │
│    commission_penalty = $1 * 1.5 = $1.50                      │
│    base_reward = ($10 - $1.50) * 1.0 = $8.50                 │
│                                                                 │
│  ⚠️  ISSUE: In bull market, BUY = +reward, SELL = -reward     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: RISK-ADJUSTED METRICS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  IF returns_history.length >= 5:                               │
│                                                                 │
│    ┌─ Sharpe Ratio (30% weight) ─────────────────────────┐   │
│    │ (mean_return - rf) / std * sqrt(365)                │   │
│    │ Measures: Risk-adjusted return                      │   │
│    │ Higher = Better                                     │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                                 │
│    ┌─ Sortino Ratio (30% weight) ────────────────────────┐   │
│    │ (mean_return - rf) / downside_std * sqrt(365)       │   │
│    │ Measures: Downside risk only                        │   │
│    │ Higher = Better (less downside)                     │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                                 │
│    ┌─ Calmar Ratio (15% weight) ─────────────────────────┐   │
│    │ annualized_return / max_drawdown                    │   │
│    │ Measures: Return per unit of drawdown              │   │
│    │ Higher = Better                                     │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                                 │
│  ⚠️  ISSUE: In bull market, all ratios favor BUY              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            STEP 4: COMPOSITE SCORE (Multi-Objective)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  composite_score = (                                           │
│      0.25 * base_reward                                        │
│    + 0.30 * sharpe_ratio                                       │
│    + 0.30 * sortino_ratio                                      │
│    + 0.15 * calmar_ratio                                       │
│    + chunk_bonus                                               │
│    + kelly_bonus                                               │
│    + risk_parity_bonus                                         │
│    + stress_var_penalty                                        │
│  )                                                              │
│                                                                 │
│  ⚠️  ISSUE: All components reward BUY in bull market           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: BEHAVIORAL ADJUSTMENTS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─ Inaction Penalty ──────────────────────────────────────┐  │
│  │ if action == 0 (HOLD):                                 │  │
│  │     composite_score -= 0.0001                          │  │
│  │ Effect: Discourages HOLD                               │  │
│  │ ⚠️  In bull market: Encourages BUY                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Drawdown Penalty ──────────────────────────────────────┐  │
│  │ if drawdown < -5%:                                     │  │
│  │     composite_score -= abs(drawdown) * 10              │  │
│  │ Effect: Penalizes large losses                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─ Duration Penalty ──────────────────────────────────────┐  │
│  │ if trade_reason == "MaxDuration":                      │  │
│  │     composite_score -= 1.0 * (1 - risk_horizon)        │  │
│  │ Effect: Penalizes long-running trades                  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ⚠️  ISSUE: Inaction penalty + bull market = BUY bias         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 6: FINAL REWARD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  final_reward = clip(composite_score, -5.0, 5.0)              │
│                                                                 │
│  Return: float value in range [-5.0, 5.0]                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## REWARD COMPONENT WEIGHTS

```
┌──────────────────────────────────────────────────────────────┐
│                    REWARD COMPOSITION                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  PnL Component              ████░░░░░░░░░░░░░░░░  25%       │
│  Sharpe Ratio               ██████░░░░░░░░░░░░░░░  30%       │
│  Sortino Ratio              ██████░░░░░░░░░░░░░░░  30%       │
│  Calmar Ratio               ███░░░░░░░░░░░░░░░░░░  15%       │
│                                                              │
│  Total: 100%                                                │
│                                                              │
│  ⚠️  ISSUE: PnL weight (25%) too high for bull market       │
│      Should be reduced to 15% to avoid BUY bias             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SHARPE RATIO CALCULATION (Time-Weighted)

```
┌─────────────────────────────────────────────────────────────┐
│              SHARPE RATIO FORMULA                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sharpe = (weighted_mean_return - rf) / weighted_std       │
│           * sqrt(365)                                      │
│                                                             │
│  where:                                                     │
│    weighted_mean_return = Σ(return_i * weight_i)          │
│    weight_i = 0.99^i (exponential decay)                  │
│    weighted_std = sqrt(Σ((return_i - mean)^2 * weight_i))│
│    rf = 0.01 (1% annual risk-free rate)                   │
│    365 = annual trading days                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

EXAMPLE: Last 5 returns
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Returns:     [0.5%,  0.3%, -0.2%,  0.4%,  0.6%]          │
│  Weights:     [0.95,  0.96,  0.97,  0.98,  1.00]          │
│  (normalized)                                              │
│                                                             │
│  Weighted Mean = 0.5*0.95 + 0.3*0.96 + (-0.2)*0.97        │
│                + 0.4*0.98 + 0.6*1.00                       │
│                = 1.561% (weighted)                         │
│                                                             │
│  Weighted Std = sqrt(variance of weighted returns)         │
│               ≈ 0.35%                                      │
│                                                             │
│  Sharpe = (1.561% - 0.01%/365) / 0.35% * sqrt(365)       │
│         ≈ 1.56% / 0.35% * 19.1                            │
│         ≈ 85.0                                             │
│                                                             │
│  ✅ High Sharpe = Good risk-adjusted return               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SORTINO RATIO CALCULATION (Downside Risk Only)

```
┌─────────────────────────────────────────────────────────────┐
│              SORTINO RATIO FORMULA                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sortino = (weighted_mean_return - rf) / downside_std      │
│            * sqrt(365)                                     │
│                                                             │
│  where:                                                     │
│    downside_std = sqrt(Σ(min(return_i, 0)^2 * weight_i))  │
│    Only NEGATIVE returns contribute                        │
│                                                             │
│  Key Difference from Sharpe:                               │
│    Sharpe: Penalizes ALL volatility (up and down)         │
│    Sortino: Penalizes ONLY downside volatility (losses)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

EXAMPLE: Same 5 returns
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Returns:           [0.5%,  0.3%, -0.2%,  0.4%,  0.6%]    │
│  Downside returns:  [0,     0,    -0.2%,  0,     0]        │
│                                                             │
│  downside_std = sqrt(weighted variance of downside)        │
│               ≈ 0.08%  (much lower than Sharpe's 0.35%)   │
│                                                             │
│  Sortino = (1.561% - 0.01%/365) / 0.08% * sqrt(365)      │
│          ≈ 1.56% / 0.08% * 19.1                           │
│          ≈ 371.0  (much higher than Sharpe's 85.0)        │
│                                                             │
│  ✅ High Sortino = Good return with minimal downside      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## CALMAR RATIO CALCULATION (Drawdown-Adjusted)

```
┌─────────────────────────────────────────────────────────────┐
│              CALMAR RATIO FORMULA                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Calmar = annualized_return / max_drawdown                 │
│                                                             │
│  where:                                                     │
│    annualized_return = (1 + cumulative_return)^365 - 1    │
│    max_drawdown = maximum peak-to-trough decline           │
│                                                             │
│  Interpretation:                                            │
│    How much return per unit of drawdown risk               │
│    Higher = Better                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

EXAMPLE:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Cumulative return over 36 months: +15%                    │
│  Max drawdown: -5%                                         │
│                                                             │
│  annualized_return = (1 + 0.15)^(365/1095) - 1            │
│                    ≈ 4.7%                                  │
│                                                             │
│  Calmar = 4.7% / 5% = 0.94                                │
│                                                             │
│  Interpretation:                                            │
│    For every 1% of drawdown risk, we get 0.94% return     │
│    ✅ Good Calmar ratio (> 0.5 is good)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## WHY MODELS SATURATE AT BUY

```
┌──────────────────────────────────────────────────────────────┐
│                  SATURATION MECHANISM                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TRAINING SCENARIO: Bull Market (BTC rising)                │
│                                                              │
│  ┌─ Action: BUY ──────────────────────────────────────────┐ │
│  │ • Market goes up                                       │ │
│  │ • Trade PnL = +$10                                     │ │
│  │ • base_reward = +$8.50                                 │ │
│  │ • Sharpe ratio = +85.0 (high return, low volatility)  │ │
│  │ • Sortino ratio = +371.0 (minimal downside)           │ │
│  │ • Calmar ratio = +0.94 (good return/drawdown)         │ │
│  │ • Total reward = 0.25*8.5 + 0.30*85 + 0.30*371 + ...  │ │
│  │                = +2.125 + 25.5 + 111.3 + ...          │ │
│  │                = +139.0 (HUGE REWARD)                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ Action: SELL ──────────────────────────────────────────┐ │
│  │ • Market goes up (you miss gains)                      │ │
│  │ • Trade PnL = -$5 (loss)                               │ │
│  │ • base_reward = -$6.50                                 │ │
│  │ • Sharpe ratio = -50.0 (negative return)              │ │
│  │ • Sortino ratio = -200.0 (downside)                   │ │
│  │ • Calmar ratio = -0.50 (negative return/drawdown)     │ │
│  │ • Total reward = 0.25*(-6.5) + 0.30*(-50) + ...       │ │
│  │                = -1.625 - 15 - 60 - ...               │ │
│  │                = -76.0 (HUGE PENALTY)                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ Action: HOLD ──────────────────────────────────────────┐ │
│  │ • No trade                                             │ │
│  │ • Trade PnL = 0                                        │ │
│  │ • base_reward = 0                                      │ │
│  │ • Inaction penalty = -0.0001                           │ │
│  │ • Total reward = -0.0001 (SMALL PENALTY)              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  RESULT:                                                     │
│    BUY:  +139.0  ✅ BEST                                    │
│    HOLD: -0.0001 ❌ WORST                                   │
│    SELL: -76.0   ❌ WORST                                   │
│                                                              │
│  Model learns: ALWAYS BUY = +139.0 reward                  │
│  Model converges to: action = 1.0 (BUY)                    │
│  Result: SATURATION AT 1.0                                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SOLUTION: REBALANCED WEIGHTS

```
┌──────────────────────────────────────────────────────────────┐
│              CURRENT vs PROPOSED WEIGHTS                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CURRENT (Biased to BUY):                                   │
│    PnL:      ████░░░░░░░░░░░░░░░░  25%                     │
│    Sharpe:   ██████░░░░░░░░░░░░░░░  30%                     │
│    Sortino:  ██████░░░░░░░░░░░░░░░  30%                     │
│    Calmar:   ███░░░░░░░░░░░░░░░░░░  15%                     │
│                                                              │
│  PROPOSED (Balanced):                                       │
│    PnL:      ███░░░░░░░░░░░░░░░░░░  15%  ← Reduced         │
│    Sharpe:   ██████░░░░░░░░░░░░░░░  30%                     │
│    Sortino:  ██████░░░░░░░░░░░░░░░  30%                     │
│    Calmar:   ███░░░░░░░░░░░░░░░░░░  15%                     │
│    Diversity:██░░░░░░░░░░░░░░░░░░░  10%  ← New             │
│                                                              │
│  EFFECT:                                                     │
│    • Reduces PnL bias (25% → 15%)                           │
│    • Adds diversity bonus (0% → 10%)                        │
│    • Encourages exploration                                 │
│    • Prevents saturation                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## EMERGENCY PATCH: NOISE INJECTION

```
┌──────────────────────────────────────────────────────────────┐
│            ANTI-SATURATION PATCH MECHANISM                   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  BEFORE PATCH:                                              │
│    Model output: 1.0 (always BUY)                           │
│    Variance: 0.0 (no diversity)                             │
│    Result: System broken                                    │
│                                                              │
│  PATCH LOGIC:                                               │
│    if abs(action_value) > 0.90:  # Detect saturation       │
│        noise = random_normal(0, 0.50)                       │
│        action_value = action_value + noise                  │
│        action_value = clip(action_value, -0.85, 0.85)      │
│                                                              │
│  AFTER PATCH:                                               │
│    Model output: 1.0 + noise                                │
│    Example: 1.0 + 0.35 = 1.35 → clipped to 0.85           │
│    Variance: > 0.03 (diversity created)                     │
│    Result: System operational                              │
│                                                              │
│  EFFECT:                                                     │
│    • Breaks saturation by adding randomness                 │
│    • Creates variance in model outputs                      │
│    • Enables diverse voting in ensemble                     │
│    • Temporary fix (1-2 weeks)                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## SUMMARY TABLE

| Metric | Formula | Weight | Bull Market | Bear Market |
|--------|---------|--------|-------------|------------|
| **PnL** | `(pnl - commission) * mult` | 25% | ✅ BUY wins | ❌ SELL wins |
| **Sharpe** | `(mean - rf) / std * √365` | 30% | ✅ BUY wins | ❌ SELL wins |
| **Sortino** | `(mean - rf) / downside * √365` | 30% | ✅ BUY wins | ❌ SELL wins |
| **Calmar** | `return / drawdown` | 15% | ✅ BUY wins | ❌ SELL wins |
| **Inaction** | `-0.0001` if HOLD | - | ❌ Discourages | ❌ Discourages |

**Conclusion**: In bull market, ALL metrics reward BUY → Saturation at 1.0

---

**Last Updated**: 2025-12-13 18:35:00 UTC
