# Implementation Plan: Data Integrity Fix for Market Indicators

- [x] 1. Set up indicator calculation module with correct formulas
  - Create `src/adan_trading_bot/indicators/calculator.py` with RSI, ADX, ATR implementations
  - Implement RSI using Wilder's smoothing method (period=14)
  - Implement ADX using directional movement method (period=14)
  - Implement ATR using true range smoothing (period=14)
  - Add input validation (sufficient data history, no NaN values)
  - Add comprehensive logging of intermediate calculation steps
  - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3_

- [x] 1.1 Write property-based tests for RSI calculation
  - **Feature: data-integrity-fix, Property 1: RSI Calculation Correctness**
  - **Validates: Requirements 1.1, 3.1**

- [x] 1.2 Write property-based tests for ADX calculation
  - **Feature: data-integrity-fix, Property 2: ADX Calculation Correctness**
  - **Validates: Requirements 1.2, 3.2**

- [x] 1.3 Write property-based tests for ATR calculation
  - **Feature: data-integrity-fix, Property 3: ATR Calculation Correctness**
  - **Validates: Requirements 1.3, 3.3**

- [x] 1.4 Write unit tests for indicator calculator
  - Test RSI with known price sequences
  - Test ADX with trending and ranging data
  - Test ATR with high and low volatility data
  - Test boundary conditions and edge cases
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Create data validator module for corruption detection
  - Create `src/adan_trading_bot/validation/data_validator.py`
  - Implement Binance API client to fetch reference indicator values
  - Implement deviation calculation (5% warning, 10% halt thresholds)
  - Implement data freshness check (reject data >5 minutes old)
  - Implement mock data detection (verify real API credentials)
  - Add comprehensive logging of validation results
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 5.1, 5.2_

- [x] 2.1 Write property-based tests for data freshness validation
  - **Feature: data-integrity-fix, Property 4: Data Freshness Validation**
  - **Validates: Requirements 1.4, 2.1**

- [x] 2.2 Write property-based tests for deviation detection
  - **Feature: data-integrity-fix, Property 5: Deviation Detection**
  - **Validates: Requirements 2.2, 2.3**

- [x] 2.3 Write property-based tests for mock data detection
  - **Feature: data-integrity-fix, Property 7: Mock Data Detection**
  - **Validates: Requirements 5.1, 5.2, 5.5**

- [x] 2.4 Write unit tests for data validator
  - Test deviation detection at 5% threshold
  - Test deviation detection at 10% threshold
  - Test data freshness validation
  - Test mock data detection
  - _Requirements: 2.1, 2.2, 2.3, 5.1, 5.2_

- [x] 3. Create observation builder module with validated indicators
  - Create `src/adan_trading_bot/observation/builder.py`
  - Implement feature vector construction from validated indicators
  - Implement normalization using current market statistics (not stale training data)
  - Implement market regime classification (bullish, bearish, ranging)
  - Add timestamp and metadata for traceability
  - Ensure observation reflects current market conditions
  - _Requirements: 4.1, 4.2, 4.4, 4.5_

- [x] 3.1 Write property-based tests for observation accuracy
  - **Feature: data-integrity-fix, Property 8: Observation Accuracy**
  - **Validates: Requirements 4.1, 4.4, 4.5**

- [x] 3.2 Write property-based tests for normalization consistency
  - **Feature: data-integrity-fix, Property 9: Normalization Consistency**
  - **Validates: Requirements 4.2**

- [x] 3.3 Write unit tests for observation builder
  - Test feature vector construction
  - Test normalization with current statistics
  - Test market regime classification
  - Test timestamp and metadata inclusion
  - _Requirements: 4.1, 4.2, 4.4, 4.5_

- [x] 4. Integrate indicator calculator into paper trading monitor
  - Modify `scripts/paper_trading_monitor.py` to use new IndicatorCalculator
  - Replace any hardcoded or cached indicator values with calculated values
  - Ensure monitor uses real Binance API data, not mock data
  - Add logging of raw price data and calculated indicators
  - _Requirements: 1.1, 1.2, 1.3, 5.1, 5.2, 6.1_

- [x] 5. Integrate data validator into paper trading monitor
  - Modify `scripts/paper_trading_monitor.py` to call DataValidator after each calculation
  - Implement warning logging for 5% deviation
  - Implement trading halt for 10% deviation
  - Export diagnostic reports on corruption detection
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.1_

- [x] 6. Integrate observation builder into paper trading monitor
  - Modify `scripts/paper_trading_monitor.py` to use ObservationBuilder
  - Ensure observations use validated indicators
  - Ensure normalization uses current market statistics
  - Verify market regime classification is accurate
  - _Requirements: 4.1, 4.2, 4.4, 4.5_

- [x] 7. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7.1 Write property-based tests for calculation reproducibility
  - **Feature: data-integrity-fix, Property 10: Calculation Reproducibility**
  - **Validates: Requirements 3.4**

- [x] 7.2 Write property-based tests for corruption audit trail
  - **Feature: data-integrity-fix, Property 6: Corruption Audit Trail**
  - **Validates: Requirements 2.4, 6.1**

- [x] 8. Create diagnostic and monitoring utilities
  - Create `scripts/diagnose_data_integrity.py` for root cause analysis
  - Create `scripts/validate_indicators_against_binance.py` for manual validation
  - Create `scripts/test_indicator_reproducibility.py` for reproducibility verification
  - Add utilities to export diagnostic reports
  - _Requirements: 2.4, 6.1, 6.2, 6.3_

- [x] 9. Create comprehensive logging and audit trail
  - Implement structured logging for all indicator calculations
  - Log raw OHLCV data, intermediate values, and final indicators
  - Log validation results (calculated vs reference, deviation %)
  - Log any detected anomalies with sufficient detail for reconstruction
  - Implement automatic export of diagnostic files on corruption detection
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 10. Verify data integrity in production
  - Run diagnostic scripts to verify indicators match Binance data
  - Verify no mock or test data is being used
  - Verify observation vectors accurately reflect market conditions
  - Verify trading models receive correct market context
  - Document baseline indicator values for comparison
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 5.1, 5.2_

- [x] 11. Final Checkpoint - Make sure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

