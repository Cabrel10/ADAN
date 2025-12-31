# Dashboard Real Data Integration - Implementation Plan

## Overview

This implementation plan converts the dashboard from using mock data to real market data from Binance testnet. It also adds network latency metrics to monitor system health.

## Tasks

- [ ] 1. Enhance RealDataCollector with Network Metrics
  - Add network latency measurement methods
  - Measure API response times
  - Measure WebSocket feed lag
  - Track connection status
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 1.1 Write property test for network metrics
  - **Property 3: Network Metrics Accuracy**
  - **Validates: Requirements 2.1, 2.2**

- [ ] 2. Fix RealDataCollector State File Reading
  - Verify state file path is correct
  - Add error handling for missing files
  - Implement retry logic with exponential backoff
  - Add logging for debugging
  - _Requirements: 1.1, 1.2, 1.5_

- [ ] 2.1 Write property test for state file reading
  - **Property 1: Real Data Connection**
  - **Validates: Requirements 1.1, 1.2**

- [ ] 3. Implement Market Data Freshness Tracking
  - Add timestamp comparison logic
  - Ensure market data updates on each refresh
  - Validate data freshness before display
  - _Requirements: 1.4_

- [ ] 3.1 Write property test for market data freshness
  - **Property 2: Market Data Freshness**
  - **Validates: Requirements 1.4**

- [ ] 4. Enhance Signal Display with Worker Votes
  - Parse worker votes from state file
  - Display individual worker confidence
  - Show decision driver (which worker made decision)
  - Format signal with color coding (BUY=green, SELL=red, HOLD=yellow)
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 4.1 Write property test for signal validity
  - **Property 4: Signal Consistency**
  - **Validates: Requirements 3.1, 3.2**

- [ ] 5. Implement Portfolio State Validation
  - Verify total value = available capital + position values
  - Validate position P&L calculations
  - Check trade history consistency
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ] 5.1 Write property test for portfolio state
  - **Property 5: Portfolio State Validity**
  - **Validates: Requirements 4.1, 4.2**

- [ ] 6. Add System Health Metrics Validation
  - Validate CPU usage (0-100%)
  - Validate memory usage (non-negative)
  - Validate uptime (0-100%)
  - Add alert thresholds
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 6.1 Write property test for system health bounds
  - **Property 6: System Health Bounds**
  - **Validates: Requirements 5.1, 5.2, 5.4**

- [ ] 7. Implement Graceful Error Handling
  - Handle missing state file gracefully
  - Implement retry logic with exponential backoff
  - Display clear error messages
  - Continue with cached data on failure
  - _Requirements: 1.5_

- [ ] 8. Test Dashboard with Real Data
  - Run dashboard with RealDataCollector
  - Verify market data displays correctly
  - Verify network metrics display
  - Verify signal updates
  - Verify portfolio state updates
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1, 4.1, 5.1_

- [ ] 8.1 Write integration test for complete dashboard
  - Test full refresh cycle
  - Test data flow from state file to display
  - Test error recovery
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 3.1, 4.1, 5.1_

- [ ] 9. Checkpoint - Verify All Tests Pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Update Dashboard Documentation
  - Document real data collector usage
  - Document network metrics
  - Document troubleshooting guide
  - _Requirements: All_

- [ ] 11. Final Checkpoint - Dashboard Ready for Production
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Optional tasks (marked with *) include property-based tests and integration tests
- Core implementation tasks must be completed for dashboard to work
- All network metrics should be measured on each refresh cycle
- State file location: `/mnt/new_data/t10_training/phase2_results/paper_trading_state.json`
- Default refresh rate: 60 seconds (configurable via CLI)
- Error recovery: Exponential backoff with max 30 second retry
