# ADAN BTC/USDT Terminal Dashboard - Specification

## Overview

This specification defines the complete requirements, design, and implementation plan for the ADAN BTC/USDT Terminal Dashboard—a professional real-time monitoring interface for the ADAN trading bot.

## Specification Files

### 1. **requirements.md**
Comprehensive requirements document with 12 major requirements and 100+ acceptance criteria covering:
- Global portfolio state display
- Active positions details
- Trade history
- ADAN decision matrix
- Performance analytics
- System health monitoring
- Market context
- Color-coded visual indicators
- Real-time data updates
- Layout and organization
- Professional aesthetics
- Data persistence and logging

### 2. **design.md**
Detailed design document with:
- System architecture (6-layer design)
- Component specifications (6 major sections)
- Data models (Position, ClosedTrade, Signal, MarketContext)
- 12 correctness properties for validation
- Error handling strategies
- Testing strategy (unit, property-based, integration, performance)

### 3. **tasks.md**
Implementation plan with 50+ actionable coding tasks organized in 11 phases:
- Phase 1: Foundation & Data Models
- Phase 2: Color Coding & Formatting
- Phase 3: Table Rendering Components
- Phase 4: Layout & Live Display
- Phase 5: Data Integration
- Phase 6: Correctness Properties & Testing
- Phase 7: Logging & Data Persistence
- Phase 8: Checkpoint & Validation
- Phase 9: Performance & Optimization
- Phase 10: Documentation & Polish
- Phase 11: Final Integration & Deployment

## Key Features

### Dashboard Sections

1. **Header** - Global portfolio state (value, capital, positions, P&L, win rate, runtime)
2. **Decision Matrix** - ADAN's current signal, confidence, worker votes, market context
3. **Active Positions** - Detailed per-position information (entry, current, SL, TP, P&L)
4. **Closed Trades** - Last 5 trades with outcomes and analysis
5. **Performance Analytics** - Comprehensive trading statistics (win rate, Sharpe, drawdown, etc.)
6. **System Health** - API/feed/model/DB status, CPU/memory, alerts

### Professional Features

- **Real-time Updates**: Price data every 500ms, metrics every 5s, performance every 30s
- **Color Coding**: Professional trading palette with conditional colors for P&L, confidence, risk
- **Data Validation**: 12 correctness properties ensuring data consistency
- **Comprehensive Testing**: Unit tests, property-based tests, integration tests, performance tests
- **Data Persistence**: Structured logging of all trades, signals, and system events
- **Expert-Ready**: Professional terminology, precise formatting, actionable insights

## Architecture

```
Exchange API → ADAN Engine → Portfolio Manager → Metrics DB
                                    ↓
                            Data Collection Layer
                                    ↓
                            Data Processing Layer
                                    ↓
                            Rendering Layer (Rich)
                                    ↓
                            Terminal Output
```

## Data Models

### Position
- Pair, side, size (BTC), entry price, current price
- SL/TP prices and distances
- Unrealized P&L (USD + %)
- Open time, duration
- Market context at entry

### ClosedTrade
- Pair, side, size (BTC), entry/exit prices
- Realized P&L (USD + %)
- Open/close times, duration
- Close reason (TP/SL/Manual/Time)
- Entry confidence

### Signal
- Direction (BUY/SELL/HOLD)
- Confidence (0.0-1.0)
- Horizon (5m/1h/4h/1d)
- Worker votes [W1, W2, W3, W4]
- Decision driver (Trend/MeanReversion/Breakout)

### MarketContext
- Price, volatility (ATR %), RSI, ADX
- Trend strength, market regime
- Volume change, timestamp

## Correctness Properties

1. **Portfolio Value Consistency** - Sum of positions + capital = total value
2. **Position P&L Accuracy** - P&L calculation matches formula
3. **Trade History Completeness** - All required fields present and non-null
4. **Signal Confidence Bounds** - Confidence between 0.0 and 1.0
5. **Worker Vote Consistency** - Average votes within 0.05 of signal confidence
6. **Performance Metrics Validity** - Metrics are mathematically valid
7. **Timestamp Ordering** - Trades in reverse chronological order
8. **Color Coding Correctness** - Colors match P&L/confidence/risk ranges
9. **Data Staleness Detection** - Stale indicator appears when age > 90s
10. **Layout Responsiveness** - All sections fit on 80+ column terminal
11. **Real-Time Update Frequency** - Correct refresh intervals for each data type
12. **System Health Status Accuracy** - Status symbols reflect actual component state

## Testing Strategy

### Unit Tests
- Data model calculations
- Color coding logic
- Formatting functions
- Section renderers

### Property-Based Tests
- All 12 correctness properties
- Random data generation
- Boundary value testing
- Edge case handling

### Integration Tests
- Layout rendering
- Data collection
- Real data integration
- Live Binance testnet connection

### Performance Tests
- Render time (<100ms)
- Refresh cycle (<500ms)
- Memory usage (<500MB)
- CPU usage (<50%)

## Implementation Status

**Status**: Specification Complete ✅

- [x] Requirements document (12 requirements, 100+ criteria)
- [x] Design document (architecture, components, properties)
- [x] Implementation plan (50+ tasks, 11 phases)
- [ ] Implementation (in progress)

## Next Steps

1. Review and approve specification
2. Begin Phase 1: Foundation & Data Models
3. Execute tasks incrementally
4. Run tests at each checkpoint
5. Deploy to production

## Contact & Questions

For questions about this specification, refer to:
- **Requirements**: See `requirements.md` for detailed acceptance criteria
- **Design**: See `design.md` for architecture and data models
- **Implementation**: See `tasks.md` for coding tasks and execution plan

---

**Specification Version**: 1.0  
**Last Updated**: 2024-12-12  
**Status**: Ready for Implementation
