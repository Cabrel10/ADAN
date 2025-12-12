# ADAN BTC/USDT Terminal Dashboard - Requirements

## Introduction

The ADAN Trading Bot requires a professional, real-time terminal dashboard to monitor BTC/USDT paper trading operations. This dashboard must expose ADAN's decision-making intelligence, not just metrics. It serves as an expert analysis tool showing WHY ADAN makes each decision, not just WHAT it does. The dashboard displays portfolio state, active positions, trade history, market context, system health, and ADAN's intentions with professional trading aesthetics using Rich library.

## Glossary

- **ADAN**: Autonomous Decision-making Agent Network - the ensemble trading bot
- **BTC/USDT**: Bitcoin trading pair against Tether stablecoin
- **Position**: Active long/short trade with entry price, SL, TP
- **P&L**: Profit and Loss (realized for closed trades, unrealized for open positions)
- **Worker**: Individual neural network agent in the ensemble (W1, W2, W3, W4)
- **Signal**: ADAN's trading intention (BUY, SELL, HOLD) with confidence score
- **Confidence**: Score 0.0-1.0 representing ADAN's conviction in current signal
- **SL (Stop Loss)**: Price level to exit losing position
- **TP (Take Profit)**: Price level to exit winning position
- **Volatility**: Market price fluctuation (ATR-based)
- **RSI**: Relative Strength Index (0-100, oversold <30, overbought >70)
- **ADX**: Average Directional Index (trend strength 0-100)
- **Slippage**: Difference between intended and actual execution price
- **Latency**: Time from decision to execution in milliseconds
- **Win Rate**: Percentage of profitable closed trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted return metric
- **Drawdown**: Peak-to-trough decline in portfolio value
- **Paper Trading**: Simulated trading without real money
- **Rich**: Python library for terminal UI with colors and tables

## Requirements

### Requirement 1: Global Portfolio State Display

**User Story:** As a trading expert, I want to see the overall portfolio health at a glance, so that I can quickly assess total exposure and performance.

#### Acceptance Criteria

1. WHEN the dashboard loads THEN the system SHALL display portfolio total value in USDT with 2 decimal places
2. WHEN the dashboard updates THEN the system SHALL display available capital (non-engaged) in USDT with 2 decimal places
3. WHEN positions exist THEN the system SHALL display count of open positions as integer
4. WHEN trades have closed THEN the system SHALL display total P&L (realized + unrealized) in USDT with 2 decimals and percentage with 2 decimals
5. WHEN the dashboard refreshes THEN the system SHALL display win rate as percentage with 1 decimal place
6. WHEN the system is running THEN the system SHALL display runtime duration in format HH:MM:SS

### Requirement 2: Active Positions Detailed Table

**User Story:** As a trader monitoring positions, I want detailed information about each open position, so that I can understand entry points, risk levels, and current performance.

#### Acceptance Criteria

1. WHEN positions are open THEN the system SHALL display one row per position with all required columns
2. WHEN a position exists THEN the system SHALL display asset pair (BTCUSDT) as text
3. WHEN a position exists THEN the system SHALL display position size in BTC with 4 decimal places
4. WHEN a position exists THEN the system SHALL display position value in USDT with 2 decimal places
5. WHEN a position exists THEN the system SHALL display average entry price in USDT with 2 decimal places
6. WHEN a position exists THEN the system SHALL display current market price in USDT with 2 decimal places
7. WHEN a position has SL THEN the system SHALL display SL price in USDT with 2 decimal places and distance as percentage
8. WHEN a position has TP THEN the system SHALL display TP price in USDT with 2 decimal places and distance as percentage
9. WHEN a position exists THEN the system SHALL display unrealized P&L in USDT with 2 decimals and percentage with 2 decimals
10. WHEN a position exists THEN the system SHALL display position open time in HH:MM:SS format
11. WHEN a position exists THEN the system SHALL display position duration in HH:MM format
12. WHEN no positions exist THEN the system SHALL display "No active positions" message

### Requirement 3: Recent Closed Trades History

**User Story:** As a performance analyst, I want to see recent closed trades with full details, so that I can understand trade outcomes and identify patterns.

#### Acceptance Criteria

1. WHEN trades have closed THEN the system SHALL display last 5 closed trades in reverse chronological order
2. WHEN a trade is closed THEN the system SHALL display trade outcome as symbol (✅ WIN, ❌ LOSS, ⚠️ BREAKEVEN)
3. WHEN a trade is closed THEN the system SHALL display trade duration in HH:MM format
4. WHEN a trade is closed THEN the system SHALL display position size in BTC with 4 decimal places
5. WHEN a trade is closed THEN the system SHALL display entry price in USDT with 2 decimal places
6. WHEN a trade is closed THEN the system SHALL display exit price in USDT with 2 decimal places
7. WHEN a trade is closed THEN the system SHALL display realized P&L in USDT with 2 decimals and percentage with 2 decimals
8. WHEN a trade is closed THEN the system SHALL display close reason as text (TP Hit, SL Hit, Manual, Time-based)
9. WHEN a trade is closed THEN the system SHALL display ADAN confidence at entry as decimal 0.0-1.0 with 2 decimals
10. WHEN no closed trades exist THEN the system SHALL display "No closed trades" message

### Requirement 4: ADAN Decision Matrix (Intentions)

**User Story:** As a system analyst, I want to understand ADAN's current decision-making state, so that I can see what signal is being generated and why.

#### Acceptance Criteria

1. WHEN the dashboard updates THEN the system SHALL display current signal as text (BUY, SELL, HOLD)
2. WHEN a signal exists THEN the system SHALL display signal confidence as decimal 0.0-1.0 with 2 decimals
3. WHEN a signal exists THEN the system SHALL display decision horizon as timeframe (5m, 1h, 4h, 1d)
4. WHEN workers vote THEN the system SHALL display individual worker votes as [W1:X, W2:Y, W3:Z, W4:W] format with 2 decimals each
5. WHEN a signal exists THEN the system SHALL display decision driver as text (Trend, Mean Reversion, Breakout, Other)
6. WHEN market data available THEN the system SHALL display current volatility (ATR) as percentage with 2 decimals
7. WHEN market data available THEN the system SHALL display RSI value as integer 0-100 with level text (Oversold, Neutral, Overbought)
8. WHEN market data available THEN the system SHALL display ADX value as integer 0-100 with trend strength text (Weak, Moderate, Strong)
9. WHEN API connected THEN the system SHALL display API latency in milliseconds as integer
10. WHEN order execution occurs THEN the system SHALL display slippage as percentage with 3 decimals

### Requirement 5: Performance Analytics

**User Story:** As a performance reviewer, I want comprehensive trading statistics, so that I can evaluate ADAN's overall effectiveness.

#### Acceptance Criteria

1. WHEN trades exist THEN the system SHALL display win rate as percentage with 1 decimal place
2. WHEN trades exist THEN the system SHALL display profit factor as decimal with 2 decimals
3. WHEN trades exist THEN the system SHALL display total trade count as integer
4. WHEN metrics available THEN the system SHALL display Sharpe ratio as decimal with 2 decimals
5. WHEN metrics available THEN the system SHALL display Sortino ratio as decimal with 2 decimals
6. WHEN metrics available THEN the system SHALL display maximum drawdown as percentage with 1 decimal place
7. WHEN trades exist THEN the system SHALL display best trade P&L in USDT with 2 decimals and percentage
8. WHEN trades exist THEN the system SHALL display worst trade P&L in USDT with 2 decimals and percentage
9. WHEN trades exist THEN the system SHALL display average holding time in HH:MM format
10. WHEN trades exist THEN the system SHALL display most profitable timeframe with win rate percentage

### Requirement 6: System Health Monitoring

**User Story:** As a system operator, I want to monitor system health and connectivity, so that I can detect issues before they impact trading.

#### Acceptance Criteria

1. WHEN system running THEN the system SHALL display API connection status as symbol (✅ or ❌) with latency in milliseconds
2. WHEN system running THEN the system SHALL display data feed status as symbol (✅ or ❌) with lag time
3. WHEN system running THEN the system SHALL display model status as symbol (✅ or ❌) with inference time in milliseconds
4. WHEN system running THEN the system SHALL display database status as symbol (✅ or ❌)
5. WHEN system running THEN the system SHALL display CPU usage as percentage 0-100
6. WHEN system running THEN the system SHALL display memory usage as GB/GB format
7. WHEN system running THEN the system SHALL display active threads as count
8. WHEN system running THEN the system SHALL display uptime as percentage with 1 decimal place
9. WHEN alerts exist THEN the system SHALL display alert list with severity level (INFO, WARNING, CRITICAL)
10. WHEN no alerts exist THEN the system SHALL display "No alerts" message

### Requirement 7: Market Context at Trade Entry

**User Story:** As a trade analyst, I want to see market conditions when each trade was entered, so that I can understand the context of ADAN's decisions.

#### Acceptance Criteria

1. WHEN a position is open THEN the system SHALL display market regime at entry as text (Trending, Ranging, Breakout)
2. WHEN a position is open THEN the system SHALL display volatility regime at entry as text (Low, Medium, High)
3. WHEN a position is open THEN the system SHALL display RSI value at entry as integer 0-100
4. WHEN a position is open THEN the system SHALL display volume spike at entry as percentage change
5. WHEN a position is open THEN the system SHALL display ADAN signal strength at entry as decimal 0.0-1.0 with 2 decimals
6. WHEN a position is open THEN the system SHALL display support/resistance distance as percentage

### Requirement 8: Color-Coded Visual Indicators

**User Story:** As a visual analyst, I want color-coded information, so that I can quickly identify profitable trades, risks, and system issues.

#### Acceptance Criteria

1. WHEN P&L is positive >2% THEN the system SHALL display in bright green color
2. WHEN P&L is positive 0-2% THEN the system SHALL display in light green color
3. WHEN P&L is breakeven -0.5% to +0.5% THEN the system SHALL display in yellow color
4. WHEN P&L is negative 0-2% THEN the system SHALL display in light red color
5. WHEN P&L is negative >2% THEN the system SHALL display in dark red color
6. WHEN confidence is 0.9-1.0 THEN the system SHALL display in dark blue color
7. WHEN confidence is 0.8-0.9 THEN the system SHALL display in blue color
8. WHEN confidence is 0.7-0.8 THEN the system SHALL display in green color
9. WHEN confidence is 0.6-0.7 THEN the system SHALL display in yellow color
10. WHEN confidence is 0.0-0.6 THEN the system SHALL display in red color
11. WHEN risk is <1% THEN the system SHALL display in green color
12. WHEN risk is 1-2% THEN the system SHALL display in yellow color
13. WHEN risk is 2-3% THEN the system SHALL display in orange color
14. WHEN risk is >3% THEN the system SHALL display in red color
15. WHEN signal is BUY THEN the system SHALL display with green background
16. WHEN signal is SELL THEN the system SHALL display with red background
17. WHEN signal is HOLD THEN the system SHALL display with yellow background

### Requirement 9: Real-Time Data Updates

**User Story:** As a live trader, I want the dashboard to update in real-time, so that I see current market conditions and position status.

#### Acceptance Criteria

1. WHEN dashboard running THEN the system SHALL refresh price data every 500 milliseconds
2. WHEN dashboard running THEN the system SHALL refresh position metrics every 5 seconds
3. WHEN dashboard running THEN the system SHALL refresh performance metrics every 30 seconds
4. WHEN dashboard running THEN the system SHALL refresh system health every 10 seconds
5. WHEN data unavailable THEN the system SHALL display "N/A" or last known value with stale indicator
6. WHEN data is stale >90 seconds THEN the system SHALL display visual stale indicator (⚠️ STALE)

### Requirement 10: Layout and Organization

**User Story:** As a dashboard user, I want information organized logically, so that I can find what I need quickly.

#### Acceptance Criteria

1. WHEN dashboard loads THEN the system SHALL display header section with title and global metrics
2. WHEN dashboard loads THEN the system SHALL display decision matrix section with ADAN intentions
3. WHEN dashboard loads THEN the system SHALL display active positions section below decision matrix
4. WHEN dashboard loads THEN the system SHALL display closed trades section beside active positions
5. WHEN dashboard loads THEN the system SHALL display performance analytics section below positions
6. WHEN dashboard loads THEN the system SHALL display system health section below analytics
7. WHEN dashboard loads THEN the system SHALL organize sections in logical hierarchy (most important first)
8. WHEN terminal resizes THEN the system SHALL adapt layout to available width (max 80 columns)
9. WHEN dashboard loads THEN the system SHALL use consistent spacing and alignment
10. WHEN dashboard loads THEN the system SHALL display all sections without horizontal scrolling on 80+ column terminal

### Requirement 11: Professional Trading Aesthetics

**User Story:** As a professional trader, I want the dashboard to look professional, so that I can use it in production environments.

#### Acceptance Criteria

1. WHEN dashboard renders THEN the system SHALL use professional trading color palette (not bright neon)
2. WHEN dashboard renders THEN the system SHALL use consistent typography and spacing
3. WHEN dashboard renders THEN the system SHALL use appropriate icons and symbols (✅, ❌, ⚠️, 🚀, 📈, etc.)
4. WHEN dashboard renders THEN the system SHALL use box drawing characters for table borders
5. WHEN dashboard renders THEN the system SHALL display numbers right-aligned in columns
6. WHEN dashboard renders THEN the system SHALL display text left-aligned in columns
7. WHEN dashboard renders THEN the system SHALL use subtle backgrounds for sections (not overwhelming)
8. WHEN dashboard renders THEN the system SHALL maintain readability on both light and dark terminals

### Requirement 12: Data Persistence and Logging

**User Story:** As a system administrator, I want trade data and events logged, so that I can audit and analyze trading history.

#### Acceptance Criteria

1. WHEN a trade opens THEN the system SHALL log trade entry with timestamp, price, size, signal strength
2. WHEN a trade closes THEN the system SHALL log trade exit with timestamp, price, P&L, close reason
3. WHEN an alert occurs THEN the system SHALL log alert with severity and timestamp
4. WHEN system starts THEN the system SHALL log startup with configuration and initial state
5. WHEN system stops THEN the system SHALL log shutdown with final statistics
6. WHEN data collected THEN the system SHALL store logs in structured format (JSON or CSV)
7. WHEN logs requested THEN the system SHALL provide access to historical trade data
8. WHEN logs requested THEN the system SHALL provide access to historical performance metrics

