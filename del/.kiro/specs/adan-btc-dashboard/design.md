# ADAN BTC/USDT Terminal Dashboard - Design Document

## Overview

The ADAN BTC/USDT Terminal Dashboard is a professional real-time monitoring interface built with Rich library. It displays the complete trading intelligence of ADAN, including portfolio state, active positions, trade history, market context, system health, and decision-making process. The dashboard is organized in a hierarchical layout with multiple sections, each serving a specific analytical purpose.

**Key Design Principles:**
- Information hierarchy: Most critical data first
- Visual density: Max 80 columns, logical grouping
- Actionable insights: Every metric has a "so what?"
- Expert-ready: Professional terminology, precise formatting
- Real-time responsiveness: Updates at appropriate intervals

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Terminal Dashboard                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Data Collection Layer                                   │ │
│  │ • Exchange API (Binance WebSocket)                      │ │
│  │ • ADAN Engine (Signals, Intentions)                     │ │
│  │ • Portfolio Manager (Positions, P&L)                    │ │
│  │ • Metrics Database (Performance, History)               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Data Processing Layer                                   │ │
│  │ • Real-time price updates (500ms)                       │ │
│  │ • Position calculations (5s)                            │ │
│  │ • Performance aggregation (30s)                         │ │
│  │ • Health checks (10s)                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Rendering Layer (Rich)                                  │ │
│  │ • Layout management                                     │ │
│  │ • Table generation                                      │ │
│  │ • Color coding                                          │ │
│  │ • Live updates                                          │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Terminal Output                                         │ │
│  │ • 80+ column display                                    │ │
│  │ • Real-time refresh                                     │ │
│  │ • Professional aesthetics                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Exchange API** → Real-time BTC/USDT price, order book, execution
2. **ADAN Engine** → Current signal, confidence, worker votes, decision driver
3. **Portfolio Manager** → Active positions, entry prices, SL/TP, P&L
4. **Metrics DB** → Trade history, performance stats, system health
5. **Dashboard Processor** → Aggregate, calculate, format data
6. **Rich Renderer** → Display with colors, tables, layout
7. **Terminal** → User sees live dashboard

## Components and Interfaces

### 1. Header Section
**Purpose:** Display global portfolio state and system status at a glance

**Data Required:**
- Portfolio total value (USDT)
- Available capital (USDT)
- Open positions count
- Total P&L (USDT + %)
- Win rate (%)
- Runtime (HH:MM:SS)
- Mode (Paper Trading)

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│ ADAN v1.0 - BTC/USDT MONITOR                                │
│ Portfolio: $1,243.75 (+24.3%) │ Positions: 2 │ Win Rate: 67% │
│ Capital: $456.20 │ P&L: +$187.55 │ Runtime: 45h 23m 12s      │
└─────────────────────────────────────────────────────────────┘
```

### 2. Decision Matrix Section
**Purpose:** Show ADAN's current decision-making state and market context

**Data Required:**
- Current signal (BUY/SELL/HOLD)
- Signal confidence (0.0-1.0)
- Decision horizon (5m/1h/4h/1d)
- Worker votes [W1, W2, W3, W4]
- Decision driver (Trend/MeanReversion/Breakout)
- Volatility (ATR %)
- RSI (0-100 + level)
- ADX (0-100 + strength)
- API latency (ms)
- Slippage (%)

**Layout:**
```
┌─ DECISION MATRIX ──────────────────────────────────────────┐
│ Signal: BUY │ Confidence: 0.87 │ Horizon: 4h │ Driver: Trend │
│ Workers: [W1:0.82, W2:0.91, W3:0.85, W4:0.88]              │
├────────────────────────────────────────────────────────────┤
│ Volatility: 2.1% │ RSI: 42.3 (Neutral) │ ADX: 28.4 (Mod)   │
│ Latency: 47ms │ Slippage: 0.03%                             │
└────────────────────────────────────────────────────────────┘
```

### 3. Active Positions Section
**Purpose:** Display detailed information about each open position

**Data Required per Position:**
- Asset pair (BTCUSDT)
- Position size (BTC)
- Position value (USDT)
- Entry price (USDT)
- Current price (USDT)
- SL price + distance (%)
- TP price + distance (%)
- Unrealized P&L (USDT + %)
- Open time (HH:MM:SS)
- Duration (HH:MM)
- Market context at entry

**Layout:**
```
┌─ ACTIVE POSITIONS ─────────────────────────────────────────┐
│ # │ Pair    │ Size   │ Entry    │ Current  │ SL      │ TP   │
├───┼─────────┼────────┼──────────┼──────────┼─────────┼──────┤
│ 1 │ BTC/USD │ 0.0245 │ 43,217.5 │ 43,850.2 │ 41,890  │ 44,57│
│   │ Value: $1,074.33 │ P&L: +$15.51 (+1.44%) │ Time: 2h14m  │
│   │ Context: RSI 38.2 (Oversold) │ Vol: 1.8% │ Signal: 0.91 │
└───┴─────────────────────────────────────────────────────────┘
```

### 4. Closed Trades Section
**Purpose:** Show recent trade history with outcomes and analysis

**Data Required per Trade:**
- Outcome (WIN/LOSS/BREAKEVEN)
- Duration (HH:MM)
- Size (BTC)
- Entry price (USDT)
- Exit price (USDT)
- Realized P&L (USDT + %)
- Close reason (TP/SL/Manual/Time)
- Entry confidence (0.0-1.0)

**Layout:**
```
┌─ LAST 5 CLOSED TRADES ─────────────────────────────────────┐
│ # │ Out │ Dur   │ Size   │ Entry    │ Exit     │ P&L    │ Rsn│
├───┼─────┼───────┼────────┼──────────┼──────────┼────────┼────┤
│ 1 │ ✅  │ 3h22m │ 0.0180 │ 42,150.0 │ 43,217.5 │ +192   │ TP │
│ 2 │ ❌  │ 1h45m │ 0.0150 │ 43,500.0 │ 42,890.0 │ -91    │ SL │
│ 3 │ ✅  │ 5h10m │ 0.0220 │ 41,780.0 │ 42,650.0 │ +191   │ TP │
└───┴─────┴───────┴────────┴──────────┴──────────┴────────┴────┘
```

### 5. Performance Analytics Section
**Purpose:** Display comprehensive trading statistics

**Data Required:**
- Win rate (%)
- Profit factor
- Total trades (count)
- Sharpe ratio
- Sortino ratio
- Max drawdown (%)
- Best trade (USDT + %)
- Worst trade (USDT + %)
- Avg holding time (HH:MM)
- Most profitable timeframe

**Layout:**
```
┌─ PERFORMANCE ANALYTICS ────────────────────────────────────┐
│ Win Rate: 67.3% │ PF: 1.63 │ Sharpe: 2.14 │ Sortino: 3.01 │
│ Max DD: 4.2% │ Trades: 45 │ Avg Hold: 2.8h                │
│ Best: +$312 (3.7%) │ Worst: -$184 (2.1%)                   │
└────────────────────────────────────────────────────────────┘
```

### 6. System Health Section
**Purpose:** Monitor system connectivity and resource usage

**Data Required:**
- API status (✅/❌) + latency
- Data feed status (✅/❌) + lag
- Model status (✅/❌) + inference time
- Database status (✅/❌)
- CPU usage (%)
- Memory usage (GB/GB)
- Active threads (count)
- Uptime (%)
- Alerts (list with severity)

**Layout:**
```
┌─ SYSTEM HEALTH ────────────────────────────────────────────┐
│ API: ✅ 12ms │ Feed: ✅ 0.1s │ Model: ✅ 87ms │ DB: ✅      │
│ CPU: 34% │ Memory: 1.2GB/4GB │ Threads: 8/12 │ Uptime: 99.7%│
│ Alerts: ⚠️ High volatility (3.2%) │ ⚠️ Spread widening     │
└────────────────────────────────────────────────────────────┘
```

## Data Models

### Position Model
```python
@dataclass
class Position:
    pair: str                    # "BTCUSDT"
    side: str                    # "LONG" or "SHORT"
    size_btc: float              # 0.0245
    entry_price: float           # 43217.50
    current_price: float         # 43850.25
    sl_price: float              # 41890.98
    tp_price: float              # 44579.63
    open_time: datetime          # 2024-12-12 10:30:00
    entry_signal_strength: float # 0.91
    entry_market_regime: str     # "Trending"
    entry_volatility: float      # 1.8
    entry_rsi: int               # 38
    
    @property
    def unrealized_pnl_usd(self) -> float:
        return (self.current_price - self.entry_price) * self.size_btc
    
    @property
    def unrealized_pnl_pct(self) -> float:
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> timedelta:
        return datetime.now() - self.open_time
```

### Trade Model
```python
@dataclass
class ClosedTrade:
    pair: str                    # "BTCUSDT"
    side: str                    # "LONG" or "SHORT"
    size_btc: float              # 0.0180
    entry_price: float           # 42150.00
    exit_price: float            # 43217.50
    open_time: datetime          # 2024-12-12 08:00:00
    close_time: datetime         # 2024-12-12 11:22:00
    close_reason: str            # "TP", "SL", "Manual", "Time"
    entry_confidence: float      # 0.91
    
    @property
    def realized_pnl_usd(self) -> float:
        return (self.exit_price - self.entry_price) * self.size_btc
    
    @property
    def realized_pnl_pct(self) -> float:
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def duration(self) -> timedelta:
        return self.close_time - self.open_time
    
    @property
    def is_win(self) -> bool:
        return self.realized_pnl_usd > 0
```

### Signal Model
```python
@dataclass
class Signal:
    direction: str               # "BUY", "SELL", "HOLD"
    confidence: float            # 0.87 (0.0-1.0)
    horizon: str                 # "5m", "1h", "4h", "1d"
    worker_votes: dict           # {"W1": 0.82, "W2": 0.91, ...}
    decision_driver: str         # "Trend", "MeanReversion", "Breakout"
    timestamp: datetime
```

### MarketContext Model
```python
@dataclass
class MarketContext:
    price: float                 # 43850.25
    volatility_atr: float        # 2.1 (%)
    rsi: int                     # 42 (0-100)
    adx: int                     # 28 (0-100)
    trend_strength: str          # "Weak", "Moderate", "Strong"
    market_regime: str           # "Trending", "Ranging", "Breakout"
    volume_change: float         # +18.0 (%)
    timestamp: datetime
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Portfolio Value Consistency
*For any* portfolio state, the sum of all position values plus available capital SHALL equal the total portfolio value.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Position P&L Calculation Accuracy
*For any* open position, the unrealized P&L calculated as (current_price - entry_price) * size SHALL match the value stored in the system.

**Validates: Requirements 2.9**

### Property 3: Trade History Completeness
*For any* closed trade, all required fields (entry price, exit price, size, time, reason) SHALL be present and non-null.

**Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

### Property 4: Signal Confidence Bounds
*For any* signal generated, the confidence value SHALL be between 0.0 and 1.0 inclusive.

**Validates: Requirements 4.2**

### Property 5: Worker Vote Consistency
*For any* signal with worker votes, the average of all worker votes SHALL be within 0.05 of the signal confidence (allowing for rounding).

**Validates: Requirements 4.4**

### Property 6: Performance Metrics Non-Negativity
*For any* performance metric (win rate, profit factor, Sharpe ratio), the value SHALL be mathematically valid (win rate 0-100%, profit factor >0, Sharpe can be negative).

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

### Property 7: Timestamp Ordering
*For any* trade history, the close_time of trade N SHALL be greater than or equal to the close_time of trade N-1 when sorted in reverse chronological order.

**Validates: Requirements 3.1**

### Property 8: Color Coding Correctness
*For any* P&L value displayed, the color SHALL match the P&L range (green for >2%, yellow for -0.5% to +0.5%, red for <-2%).

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

### Property 9: Data Staleness Detection
*For any* data point with timestamp, if current_time - timestamp > 90 seconds, the system SHALL display a stale indicator (⚠️ STALE).

**Validates: Requirements 9.6**

### Property 10: Layout Responsiveness
*For any* terminal width >= 80 columns, all dashboard sections SHALL fit without horizontal scrolling.

**Validates: Requirements 10.8**

### Property 11: Real-Time Update Frequency
*For any* dashboard refresh cycle, price data SHALL be updated at least every 500ms, position metrics every 5s, performance metrics every 30s.

**Validates: Requirements 9.1, 9.2, 9.3**

### Property 12: System Health Status Accuracy
*For any* system component (API, Feed, Model, DB), the status symbol (✅/❌) SHALL accurately reflect the component's connectivity and functionality.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

## Error Handling

### Data Unavailability
- **Scenario:** Exchange API disconnected, no price data available
- **Handling:** Display "N/A" with stale indicator, use last known value if available
- **User Impact:** Dashboard remains visible but shows outdated data

### Calculation Errors
- **Scenario:** Division by zero in P&L percentage calculation
- **Handling:** Catch exception, display "ERROR" in cell, log to system health
- **User Impact:** Single cell shows error, rest of dashboard continues

### Layout Overflow
- **Scenario:** Terminal width < 80 columns
- **Handling:** Truncate columns, use abbreviations, stack sections vertically
- **User Impact:** Some columns hidden, but critical data remains visible

### Database Connection Loss
- **Scenario:** Metrics database unavailable
- **Handling:** Display "DB: ❌" in health section, continue with in-memory data
- **User Impact:** Historical metrics unavailable, real-time data continues

## Testing Strategy

### Unit Testing
- Test P&L calculation functions with known inputs/outputs
- Test color coding logic with boundary values
- Test timestamp parsing and formatting
- Test data model validation
- Test layout calculations for different terminal widths

### Property-Based Testing
- **Property 1:** Generate random positions and verify portfolio value consistency
- **Property 2:** Generate random price changes and verify P&L calculations
- **Property 3:** Generate random trades and verify all fields present
- **Property 4:** Generate random signals and verify confidence bounds
- **Property 5:** Generate random worker votes and verify average consistency
- **Property 6:** Generate random performance data and verify metric validity
- **Property 7:** Generate random trade sequences and verify timestamp ordering
- **Property 8:** Generate random P&L values and verify color coding
- **Property 9:** Generate random timestamps and verify staleness detection
- **Property 10:** Generate random terminal widths and verify layout fit
- **Property 11:** Simulate refresh cycles and verify update frequencies
- **Property 12:** Simulate component failures and verify status accuracy

### Integration Testing
- Test dashboard with live Binance testnet connection
- Test with real ADAN signal generation
- Test with actual portfolio manager data
- Test with metrics database queries
- Test full refresh cycle with all components

### Performance Testing
- Verify dashboard renders in <100ms
- Verify refresh cycle completes in <500ms for price updates
- Verify memory usage stays <500MB
- Verify CPU usage stays <50%

