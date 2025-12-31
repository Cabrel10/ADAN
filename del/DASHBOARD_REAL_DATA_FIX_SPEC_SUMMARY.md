# Dashboard Real Data Fix - Spec Summary

**Date:** December 16, 2025  
**Status:** SPEC COMPLETE - Ready for Implementation  
**Feature:** dashboard-real-data-fix

## Problem Statement

The ADAN dashboard currently displays static mock data instead of real market data from Binance testnet. This prevents the model from trading because it receives no real market signals. Additionally, network latency metrics are missing, making it impossible to monitor system health.

## Solution Overview

Switch the dashboard from MockDataCollector to RealDataCollector, which fetches live data from the paper trading state file. Add network latency metrics to monitor API and WebSocket performance.

## Spec Documents Created

1. **requirements.md** - 6 requirements with 30 acceptance criteria
   - Real market data display
   - Network latency metrics
   - Trading signal display
   - Portfolio state display
   - System health monitoring
   - Data refresh mechanism

2. **design.md** - Complete system design
   - Architecture diagram
   - Component interfaces
   - Data models
   - 6 correctness properties
   - Error handling strategy
   - Testing strategy

3. **tasks.md** - 11 implementation tasks
   - All tasks marked as required (comprehensive approach)
   - Property-based tests for each property
   - Integration tests for end-to-end validation
   - Checkpoint tasks to verify progress

## Correctness Properties

1. **Real Data Connection**: Dashboard connects to state file and fetches portfolio data
2. **Market Data Freshness**: Market data timestamp increases on each refresh
3. **Network Metrics Accuracy**: API latency and feed lag are non-negative and < 10000ms
4. **Signal Consistency**: Signal confidence 0-100%, direction in {BUY, SELL, HOLD}
5. **Portfolio State Validity**: Total value = available capital + position values
6. **System Health Bounds**: CPU 0-100%, memory non-negative, uptime 0-100%

## Key Implementation Details

- **State File Location**: `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
- **Default Refresh Rate**: 60 seconds (configurable)
- **Error Recovery**: Exponential backoff with max 30 second retry
- **Network Metrics**: Measured on each refresh cycle
- **Testing Framework**: pytest + hypothesis for property-based tests

## Expected Outcomes

After implementation:
- ✅ Dashboard displays real BTC price, RSI, ADX, volatility
- ✅ Dashboard shows real portfolio positions and P&L
- ✅ Dashboard displays trading signals with worker votes
- ✅ Dashboard shows network latency metrics (API, WebSocket)
- ✅ Dashboard updates every 60 seconds with fresh data
- ✅ Model receives real observations and can trade
- ✅ All 6 correctness properties verified by tests

## Next Steps

1. Review requirements.md - Confirm all acceptance criteria are correct
2. Review design.md - Confirm architecture and properties are sound
3. Review tasks.md - Confirm implementation plan is complete
4. Begin implementation starting with Task 1

## Files Created

- `.kiro/specs/dashboard-real-data-fix/requirements.md`
- `.kiro/specs/dashboard-real-data-fix/design.md`
- `.kiro/specs/dashboard-real-data-fix/tasks.md`
- `DASHBOARD_REAL_DATA_FIX_SPEC_SUMMARY.md` (this file)
