# Data Flow Fix - Implementation Tasks

## Overview

Fix 5 critical issues preventing data flow from WebSocket to trading model. Each task is focused and builds on previous ones.

---

## Task List

- [x] 1. Fix Worker ID Normalization
  - Add `clean_worker_id()` function to multi_asset_chunked_env.py
  - Handle string formats ('W0', 'w1'), integers, and None
  - Default to 0 for invalid inputs
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 1.1 Write property test for worker ID normalization
  - **Property 2: Worker ID Normalization**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [x] 2. Fix Dashboard Data Collection
  - Update diagnostic script to use correct RealDataCollector methods
  - Use `get_portfolio_state()` instead of `get_latest_data()`
  - Use `get_market_context()` for market data
  - Use `get_system_health()` for system status
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 2.1 Write property test for dashboard data retrieval
  - **Property 3: Dashboard Data Retrieval**
  - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [ ] 3. Fix Indicator Calculator Usage
  - Update diagnostic to use static methods correctly
  - Call `IndicatorCalculator.calculate_rsi()` not `calc.calculate_rsi()`
  - Call `IndicatorCalculator.calculate_atr()` not `calc.calculate_atr()`
  - Call `IndicatorCalculator.calculate_adx()` not `calc.calculate_adx()`
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 3.1 Write property test for indicator calculations
  - **Property 4: Indicator Calculation**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [ ] 4. Fix Observation Builder Usage
  - Update diagnostic to initialize without config parameter
  - Call `ObservationBuilder()` not `ObservationBuilder(config)`
  - Verify build() method returns numpy array
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 4.1 Write property test for observation building
  - **Property 5: Observation Building**
  - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [ ] 5. Fix API Key Validation
  - Verify connector.py checks for BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET_KEY
  - Add clear error messages if keys are missing
  - Document required environment variables
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 5.1 Write property test for API key validation
  - **Property 1: API Keys Validation**
  - **Validates: Requirements 1.1, 1.2**

- [ ] 6. Run Complete Data Flow Diagnostic
  - Execute updated diagnostic script
  - Verify all 6 tests pass
  - Document results
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6.1 Write property test for complete data flow
  - **Property 6: Complete Data Flow**
  - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

- [ ] 7. Checkpoint - Verify All Tests Pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Test Paper Trading with Real Data
  - Launch paper trading with fixed data flow
  - Verify dashboard shows real market data
  - Verify model receives observations
  - Verify trades are executed
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 9. Final Verification
  - Run complete diagnostic again
  - Verify all 6 data flow tests pass
  - Document final status
  - Create summary report
