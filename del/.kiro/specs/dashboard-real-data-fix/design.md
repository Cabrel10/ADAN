# Dashboard Real Data Integration - Design

## Overview

The dashboard currently uses MockDataCollector which generates fake data. We need to switch to RealDataCollector which fetches live data from Binance testnet and the ADAN trading system. We also need to add network latency metrics to monitor system health.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (Terminal UI)                   │
│  - Header (Portfolio, Positions, Win Rate)                  │
│  - Decision Matrix (Signal, Confidence, Workers)            │
│  - Active Positions (Real-time P&L)                         │
│  - Closed Trades (Trade History)                            │
│  - System Health (Network, CPU, Memory)                     │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Refresh every N seconds
                            │
┌─────────────────────────────────────────────────────────────┐
│              RealDataCollector (Data Source)                 │
│  - Connects to state file (paper_trading_state.json)        │
│  - Fetches portfolio state                                  │
│  - Fetches market context (price, RSI, ADX, etc)           │
│  - Fetches trading signal                                   │
│  - Fetches system health                                    │
│  - Measures network latency                                 │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
   │ State File │   │ Binance API  │   │ WebSocket    │
   │ (JSON)     │   │ (REST)       │   │ (Live Data)  │
   └────────────┘   └──────────────┘   └──────────────┘
```

## Components and Interfaces

### 1. RealDataCollector
- **Purpose**: Fetch real market and portfolio data
- **Methods**:
  - `connect()`: Connect to data sources
  - `disconnect()`: Close connections
  - `get_portfolio_state()`: Get current positions and capital
  - `get_market_context()`: Get current price, RSI, ADX, volatility
  - `get_current_signal()`: Get trading signal from ensemble
  - `get_system_health()`: Get API latency, feed status, resource usage

### 2. NetworkMetricsCollector
- **Purpose**: Measure network latency and connectivity
- **Methods**:
  - `measure_api_latency()`: Measure REST API response time
  - `measure_feed_lag()`: Measure WebSocket feed lag
  - `get_connection_status()`: Check API and feed connectivity

### 3. Dashboard Renderer
- **Purpose**: Display data in terminal
- **Updates**: Refresh every N seconds with fresh data
- **Sections**: Header, Decision Matrix, Positions, Trades, System Health

## Data Models

### PortfolioState
```python
@dataclass
class PortfolioState:
    total_value_usd: float
    available_capital_usd: float
    open_positions: List[Position]
    closed_trades: List[ClosedTrade]
    current_signal: Signal
    market_context: MarketContext
    system_health: dict
    timestamp: datetime
```

### MarketContext
```python
@dataclass
class MarketContext:
    price: float
    volatility_atr: float
    rsi: float
    adx: float
    trend_strength: str
    market_regime: str
    volume_change: float
    timestamp: datetime
```

### SystemHealth
```python
@dataclass
class SystemHealth:
    api_status: bool
    api_latency_ms: float
    feed_status: bool
    feed_lag_ms: float
    model_status: bool
    model_latency_ms: float
    db_status: bool
    cpu_percent: float
    memory_gb: float
    memory_total_gb: float
    threads: int
    uptime_percent: float
    alerts: List[Alert]
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Real Data Connection
*For any* dashboard instance, when RealDataCollector is used, the system SHALL successfully connect to the state file and fetch portfolio data.
**Validates: Requirements 1.1, 1.2**

### Property 2: Market Data Freshness
*For any* refresh cycle, the market data displayed SHALL be fresher than the previous refresh (timestamp increases).
**Validates: Requirements 1.4**

### Property 3: Network Metrics Accuracy
*For any* system health check, the API latency and feed lag values SHALL be non-negative and within reasonable bounds (< 10000ms).
**Validates: Requirements 2.1, 2.2**

### Property 4: Signal Consistency
*For any* trading signal, the confidence value SHALL be between 0 and 100, and the signal direction SHALL be one of {BUY, SELL, HOLD}.
**Validates: Requirements 3.1, 3.2**

### Property 5: Portfolio State Validity
*For any* portfolio state, the total value SHALL equal available capital plus sum of position values.
**Validates: Requirements 4.1, 4.2**

### Property 6: System Health Bounds
*For any* system health metrics, CPU usage SHALL be 0-100%, memory usage SHALL be non-negative, and uptime SHALL be 0-100%.
**Validates: Requirements 5.1, 5.2, 5.4**

## Error Handling

1. **State File Not Found**: Display "Waiting for state file..." and retry every 5 seconds
2. **API Connection Failed**: Display "API Error" with red indicator, retry with exponential backoff
3. **Invalid Data**: Log error, use last valid data, display warning
4. **Network Timeout**: Display latency warning, continue with cached data
5. **Graceful Shutdown**: Close connections on Ctrl+C, save state if needed

## Testing Strategy

### Unit Tests
- Test RealDataCollector connection logic
- Test data parsing from state file
- Test network metrics calculation
- Test error handling for missing files

### Property-Based Tests
- Property 1: Real data connection works
- Property 2: Market data freshness increases
- Property 3: Network metrics in valid ranges
- Property 4: Signal values valid
- Property 5: Portfolio state arithmetic correct
- Property 6: System health metrics in bounds

### Integration Tests
- Test full dashboard refresh cycle
- Test data flow from state file to display
- Test network latency measurement
- Test graceful shutdown

### Testing Framework
- **Unit Tests**: pytest
- **Property Tests**: hypothesis
- **Integration Tests**: pytest with fixtures

## Implementation Notes

1. **State File Location**: `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
2. **Refresh Rate**: Default 60 seconds, configurable via CLI
3. **Network Metrics**: Measure on each refresh cycle
4. **Error Recovery**: Exponential backoff with max 30 second retry
5. **Display Update**: Use Rich library for terminal rendering
