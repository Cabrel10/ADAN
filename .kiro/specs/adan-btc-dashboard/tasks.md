# ADAN BTC/USDT Terminal Dashboard - Implementation Plan

## Overview

This implementation plan converts the feature design into a series of actionable coding tasks. Each task builds incrementally on previous tasks, with no orphaned code. The plan focuses on writing, modifying, and testing code to bring the dashboard to life.

---

## Phase 1: Foundation & Data Models

- [x] 1. Set up project structure and dependencies
  - Create `scripts/adan_btc_dashboard.py` as main entry point
  - Create `src/adan_trading_bot/dashboard/` package
  - Install Rich library: `pip install rich`
  - Create `__init__.py` files for package structure
  - _Requirements: 10.1_

- [x] 1.1 Create data model classes
  - Implement `Position` dataclass with all fields and calculated properties
  - Implement `ClosedTrade` dataclass with all fields and calculated properties
  - Implement `Signal` dataclass with all fields
  - Implement `MarketContext` dataclass with all fields
  - Create `src/adan_trading_bot/dashboard/models.py`
  - _Requirements: 2.1, 3.1, 4.1, 7.1_

- [x] 1.2 Write unit tests for data models
  - Test Position P&L calculations with known values
  - Test ClosedTrade P&L calculations with known values
  - Test Signal confidence bounds (0.0-1.0)
  - Test timestamp parsing and formatting
  - Create `tests/test_dashboard_models.py`
  - _Requirements: 2.9, 3.7, 4.2_

- [x] 1.3 Create data collection interface
  - Define abstract `DataCollector` class with methods for fetching positions, trades, signals, market context
  - Create `src/adan_trading_bot/dashboard/data_collector.py`
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

- [x] 1.4 Implement mock data collector for testing
  - Create `MockDataCollector` that generates realistic test data
  - Implement methods to generate random positions, trades, signals
  - Create `src/adan_trading_bot/dashboard/mock_collector.py`
  - _Requirements: 1.1, 2.1, 3.1, 4.1_

---

## Phase 2: Color Coding & Formatting

- [x] 2. Create color palette and formatting utilities
  - Define professional trading color palette (not neon)
  - Create color mapping functions for P&L ranges
  - Create color mapping functions for confidence ranges
  - Create color mapping functions for risk ranges
  - Create `src/adan_trading_bot/dashboard/colors.py`
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14_

- [x] 2.1 Create formatting utilities
  - Implement `format_usd(value)` → "$X,XXX.XX"
  - Implement `format_btc(value)` → "0.XXXX"
  - Implement `format_percentage(value)` → "X.XX%"
  - Implement `format_time(duration)` → "HH:MM:SS"
  - Implement `format_confidence(value)` → "0.XX"
  - Create `src/adan_trading_bot/dashboard/formatters.py`
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11_

- [x] 2.2 Write property tests for color coding
  - **Property 8: Color Coding Correctness**
  - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14**
  - Generate random P&L values and verify correct color applied
  - Generate random confidence values and verify correct color applied
  - Generate random risk values and verify correct color applied
  - Create `tests/test_dashboard_colors.py`

- [x] 2.3 Write unit tests for formatting
  - Test USD formatting with various values (0, 1000, 1000000, negative)
  - Test BTC formatting with 4 decimal places
  - Test percentage formatting with 1-2 decimal places
  - Test time formatting with various durations
  - Create `tests/test_dashboard_formatters.py`

---

## Phase 3: Table Rendering Components

- [x] 3. Create header section renderer
  - Implement `render_header(portfolio_data)` function
  - Display portfolio value, available capital, position count, P&L, win rate, runtime
  - Use Rich Panel and Text for styling
  - Apply professional colors
  - Create `src/adan_trading_bot/dashboard/sections/header.py`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 3.1 Create decision matrix section renderer
  - Implement `render_decision_matrix(signal_data, market_context)` function
  - Display signal, confidence, horizon, worker votes, decision driver
  - Display volatility, RSI, ADX, latency, slippage
  - Use Rich Table with proper alignment
  - Apply color coding for confidence and signal
  - Create `src/adan_trading_bot/dashboard/sections/decision_matrix.py`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10_

- [x] 3.2 Create active positions section renderer
  - Implement `render_positions(positions_list)` function
  - Display one row per position with all required columns
  - Calculate and display P&L, duration, SL/TP distances
  - Handle empty positions case ("No active positions")
  - Use Rich Table with proper alignment and colors
  - Create `src/adan_trading_bot/dashboard/sections/positions.py`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12_

- [x] 3.3 Create closed trades section renderer
  - Implement `render_closed_trades(trades_list)` function
  - Display last 5 trades in reverse chronological order
  - Show outcome symbol (✅/❌/⚠️), duration, size, prices, P&L, reason, confidence
  - Handle empty trades case ("No closed trades")
  - Use Rich Table with proper alignment and colors
  - Create `src/adan_trading_bot/dashboard/sections/closed_trades.py`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 3.4 Create performance analytics section renderer
  - Implement `render_performance(trades_list, metrics)` function
  - Display win rate, profit factor, trade count, Sharpe, Sortino, max drawdown
  - Display best/worst trades, average holding time, most profitable timeframe
  - Use Rich Table with proper alignment
  - Create `src/adan_trading_bot/dashboard/sections/performance.py`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

- [x] 3.5 Create system health section renderer
  - Implement `render_system_health(health_data)` function
  - Display API, feed, model, database status with symbols and metrics
  - Display CPU, memory, threads, uptime
  - Display alerts list with severity levels
  - Handle empty alerts case ("No alerts")
  - Use Rich Table with proper alignment and colors
  - Create `src/adan_trading_bot/dashboard/sections/system_health.py`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10_

- [x] 3.6 Write unit tests for section renderers
  - Test header rendering with various portfolio states
  - Test decision matrix rendering with different signals
  - Test positions rendering with multiple positions
  - Test closed trades rendering with various outcomes
  - Test performance rendering with different metrics
  - Test system health rendering with various statuses
  - Create `tests/test_dashboard_sections.py`

---

## Phase 4: Layout & Live Display

- [x] 4. Create layout manager
  - Implement `create_layout()` function using Rich Layout
  - Define hierarchical layout: header → main (left/right) → footer
  - Left side: decision matrix, positions, performance
  - Right side: closed trades, system health
  - Create `src/adan_trading_bot/dashboard/layout.py`
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10_

- [x] 4.1 Create dashboard generator
  - Implement `generate_dashboard(data)` function
  - Combine all section renderers into complete layout
  - Apply consistent spacing and alignment
  - Ensure no horizontal scrolling on 80+ column terminal
  - Create `src/adan_trading_bot/dashboard/generator.py`
  - _Requirements: 10.1, 10.7, 10.8, 10.9, 10.10, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8_

- [x] 4.2 Create main dashboard application
  - Implement `AdanBtcDashboard` class with Live display
  - Initialize Rich Console and Live
  - Implement refresh loop with appropriate intervals
  - Handle keyboard interrupt gracefully
  - Create `src/adan_trading_bot/dashboard/app.py`
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 4.3 Create main entry point
  - Implement `scripts/adan_btc_dashboard.py`
  - Parse command-line arguments (data source, refresh rate, etc.)
  - Initialize data collector
  - Start dashboard application
  - Handle errors and logging
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 4.4 Write integration tests for layout
  - Test layout renders without errors
  - Test all sections are present in layout
  - Test layout adapts to different terminal widths
  - Test no horizontal scrolling on 80+ columns
  - Create `tests/test_dashboard_layout.py`

---

## Phase 5: Data Integration

- [x] 5. Create real data collector
  - Implement `RealDataCollector` class
  - Integrate with existing ADAN components (portfolio manager, metrics DB)
  - Fetch current positions from portfolio manager
  - Fetch closed trades from metrics database
  - Fetch current signal from ADAN engine
  - Fetch market context from exchange API
  - Create `src/adan_trading_bot/dashboard/real_collector.py`
  - _Requirements: 1.1, 2.1, 3.1, 4.1, 6.1, 6.2, 6.3, 6.4_

- [x] 5.1 Create data cache layer
  - Implement `DataCache` class to store latest data
  - Implement TTL-based staleness detection
  - Implement fallback to last known value when data unavailable
  - Create `src/adan_trading_bot/dashboard/cache.py`
  - _Requirements: 9.5, 9.6_

- [x] 5.2 Create data aggregator
  - Implement `DataAggregator` class
  - Aggregate position data with market context
  - Calculate performance metrics from trade history
  - Combine all data sources into single dashboard data structure
  - Create `src/adan_trading_bot/dashboard/aggregator.py`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [x] 5.3 Write integration tests with real data
  - Test data collector fetches all required data
  - Test data cache handles staleness correctly
  - Test data aggregator combines data correctly
  - Test with live Binance testnet connection
  - Create `tests/test_dashboard_integration.py`

---

## Phase 6: Correctness Properties & Testing

- [x] 6. Implement property tests for core properties
  - **Property 1: Portfolio Value Consistency**
  - **Validates: Requirements 1.1, 1.2, 1.3**
  - Generate random portfolio states
  - Verify sum of positions + capital = total value
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.1 Implement property test for P&L accuracy
  - **Property 2: Position P&L Calculation Accuracy**
  - **Validates: Requirements 2.9**
  - Generate random positions with various prices
  - Verify P&L calculation matches formula
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.2 Implement property test for trade history
  - **Property 3: Trade History Completeness**
  - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**
  - Generate random trades
  - Verify all required fields are present and non-null
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.3 Implement property test for signal bounds
  - **Property 4: Signal Confidence Bounds**
  - **Validates: Requirements 4.2**
  - Generate random signals
  - Verify confidence is between 0.0 and 1.0
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.4 Implement property test for worker votes
  - **Property 5: Worker Vote Consistency**
  - **Validates: Requirements 4.4**
  - Generate random worker votes
  - Verify average of votes is within 0.05 of signal confidence
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.5 Implement property test for metrics validity
  - **Property 6: Performance Metrics Non-Negativity**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
  - Generate random performance data
  - Verify metrics are mathematically valid
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.6 Implement property test for timestamp ordering
  - **Property 7: Timestamp Ordering**
  - **Validates: Requirements 3.1**
  - Generate random trade sequences
  - Verify trades are in reverse chronological order
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.7 Implement property test for color coding
  - **Property 8: Color Coding Correctness**
  - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14**
  - Generate random P&L, confidence, risk values
  - Verify correct color is applied for each range
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.8 Implement property test for staleness detection
  - **Property 9: Data Staleness Detection**
  - **Validates: Requirements 9.6**
  - Generate data with various timestamps
  - Verify stale indicator appears when age > 90 seconds
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.9 Implement property test for layout responsiveness
  - **Property 10: Layout Responsiveness**
  - **Validates: Requirements 10.8**
  - Test layout with various terminal widths (80, 100, 120, 160 columns)
  - Verify all sections fit without horizontal scrolling
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.10 Implement property test for update frequency
  - **Property 11: Real-Time Update Frequency**
  - **Validates: Requirements 9.1, 9.2, 9.3**
  - Measure refresh intervals for different data types
  - Verify price updates every ~500ms, metrics every ~5s, performance every ~30s
  - Create test in `tests/test_dashboard_properties.py`

- [ ] 6.11 Implement property test for health status
  - **Property 12: System Health Status Accuracy**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
  - Mock various component states
  - Verify status symbols (✅/❌) accurately reflect state
  - Create test in `tests/test_dashboard_properties.py`

---

## Phase 7: Logging & Data Persistence

- [x] 7. Create logging system
  - Implement `DashboardLogger` class
  - Log trade entries with timestamp, price, size, signal strength
  - Log trade exits with timestamp, price, P&L, close reason
  - Log alerts with severity and timestamp
  - Log system startup/shutdown with configuration
  - Create `src/adan_trading_bot/dashboard/logger.py`
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 7.1 Create structured log storage
  - Implement JSON-based log storage
  - Create methods to query historical trade data
  - Create methods to query historical performance metrics
  - Create `src/adan_trading_bot/dashboard/log_storage.py`
  - _Requirements: 12.6, 12.7, 12.8_

- [x] 7.2 Write unit tests for logging
  - Test trade entry logging with all required fields
  - Test trade exit logging with all required fields
  - Test alert logging with severity
  - Test log storage and retrieval
  - Create `tests/test_dashboard_logging.py`

---

## Phase 8: Checkpoint & Validation

- [ ] 8. Checkpoint - Ensure all tests pass
  - Run all unit tests: `pytest tests/test_dashboard_*.py -v`
  - Run all property tests: `pytest tests/test_dashboard_properties.py -v --hypothesis-seed=0`
  - Verify no test failures
  - Ensure all tests pass, ask the user if questions arise.

---

## Phase 9: Performance & Optimization

- [ ] 9. Performance testing and optimization
  - Measure dashboard render time (target: <100ms)
  - Measure refresh cycle time (target: <500ms for price updates)
  - Profile memory usage (target: <500MB)
  - Profile CPU usage (target: <50%)
  - Optimize hot paths if needed
  - Create `tests/test_dashboard_performance.py`

- [ ] 9.1 Optimize rendering
  - Cache static elements
  - Minimize table recalculations
  - Use efficient string formatting
  - Profile and optimize bottlenecks

---

## Phase 10: Documentation & Polish

- [ ] 10. Create user documentation
  - Write README for dashboard usage
  - Document command-line arguments
  - Document configuration options
  - Document keyboard shortcuts (if any)
  - Create `docs/DASHBOARD_GUIDE.md`

- [ ] 10.1 Create developer documentation
  - Document architecture and design decisions
  - Document how to add new sections
  - Document how to customize colors
  - Document how to integrate new data sources
  - Create `docs/DASHBOARD_DEVELOPMENT.md`

- [ ] 10.2 Add inline code documentation
  - Add docstrings to all public functions
  - Add type hints to all functions
  - Add comments for complex logic
  - Ensure code is self-documenting

---

## Phase 11: Final Integration & Deployment

- [ ] 11. Final integration with ADAN system
  - Integrate dashboard with paper trading monitor
  - Ensure dashboard can run alongside trading bot
  - Test with live paper trading data
  - Verify no performance impact on trading bot
  - _Requirements: All_

- [ ] 11.1 Create deployment script
  - Create `scripts/deploy_dashboard.sh`
  - Automate installation of dependencies
  - Automate configuration setup
  - Create systemd service file (optional)

- [ ] 11.2 Final checkpoint - Ensure all tests pass
  - Run complete test suite: `pytest tests/ -v`
  - Verify all property tests pass
  - Verify all integration tests pass
  - Ensure the system is production-ready
  - Ensure all tests pass, ask the user if questions arise.

---

## Summary

This implementation plan provides a structured, incremental approach to building the ADAN BTC/USDT Terminal Dashboard. Each phase builds on previous phases with no orphaned code. The plan includes:

- **Phase 1-2:** Foundation, data models, colors, formatting
- **Phase 3-4:** Table rendering, layout, live display
- **Phase 5:** Real data integration
- **Phase 6:** Correctness properties and comprehensive testing
- **Phase 7:** Logging and data persistence
- **Phase 8-11:** Validation, optimization, documentation, deployment

Total estimated tasks: 50+ coding tasks with comprehensive testing coverage.

