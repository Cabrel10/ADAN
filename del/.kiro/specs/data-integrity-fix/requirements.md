# Requirements Document: Data Integrity Fix for Market Indicators

## Introduction

The ADAN trading system is currently experiencing a critical data integrity failure where calculated market indicators (ADX, RSI, ATR) do not match real market conditions from Binance. This mismatch causes the ensemble of trading models to make incorrect decisions based on corrupted input signals. The system reports ADX=100, RSI=44, and ATR=0.88% while actual market data shows ADX≈29.77, RSI≈37.59, and ATR≈0.18%. This 3-5x discrepancy in key indicators results in models unanimously voting BUY in a bearish market, leading to systematic trading losses. The fix requires establishing a reliable indicator calculation pipeline with real-time validation against authoritative market data sources.

## Glossary

- **Indicator**: A technical analysis metric (RSI, ADX, ATR) calculated from OHLCV price data
- **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and magnitude of price changes (0-100 scale)
- **ADX (Average Directional Index)**: Trend strength indicator (0-100 scale, >25 indicates strong trend)
- **ATR (Average True Range)**: Volatility indicator measuring average price movement range
- **OHLCV**: Open, High, Low, Close, Volume price data
- **Binance**: Authoritative cryptocurrency exchange providing real market data
- **Paper Trading Monitor**: System component that calculates indicators and generates trading signals
- **Observation**: Normalized feature vector fed to trading models
- **Market Regime**: Classification of current market conditions (trending, ranging, bullish, bearish)
- **Data Corruption**: Mismatch between calculated and real market indicator values
- **Covariate Shift**: Distribution mismatch between training and inference data

## Requirements

### Requirement 1

**User Story:** As a system operator, I want the indicator calculation pipeline to produce accurate market indicators, so that trading models receive correct market context and make sound trading decisions.

#### Acceptance Criteria

1. WHEN the paper trading monitor calculates RSI from recent price data THEN the calculated RSI value SHALL match the RSI calculated from the same price data using the standard Wilder's smoothing method with period=14
2. WHEN the paper trading monitor calculates ADX from recent price data THEN the calculated ADX value SHALL match the ADX calculated from the same price data using the standard directional movement method with period=14
3. WHEN the paper trading monitor calculates ATR from recent price data THEN the calculated ATR value SHALL match the ATR calculated from the same price data using the standard true range method with period=14
4. WHEN the paper trading monitor retrieves OHLCV data from Binance THEN the retrieved data SHALL be from the current market timeframe and SHALL not be older than 5 minutes
5. WHEN the paper trading monitor calculates indicators THEN the system SHALL log the raw price data, intermediate calculation steps, and final indicator values for audit purposes

### Requirement 2

**User Story:** As a system architect, I want to validate that calculated indicators match real market conditions, so that I can detect data corruption before it affects trading decisions.

#### Acceptance Criteria

1. WHEN the paper trading monitor completes an indicator calculation cycle THEN the system SHALL compare calculated indicators against reference values from Binance API
2. WHEN calculated indicators deviate from reference values by more than 5% THEN the system SHALL log a warning and flag the data as potentially corrupted
3. WHEN calculated indicators deviate from reference values by more than 10% THEN the system SHALL halt trading and alert the operator
4. WHEN the system detects data corruption THEN it SHALL record the timestamp, calculated values, reference values, and deviation percentage for root cause analysis
5. WHEN the system detects data corruption THEN it SHALL provide a diagnostic report showing which indicator(s) are affected and the magnitude of the error

### Requirement 3

**User Story:** As a data engineer, I want to ensure the indicator calculation pipeline uses correct formulas and parameters, so that calculated values are mathematically sound and reproducible.

#### Acceptance Criteria

1. WHEN the RSI calculation function is invoked THEN it SHALL use Wilder's smoothing method with the formula: RS = (average of up closes) / (average of down closes), RSI = 100 - (100 / (1 + RS))
2. WHEN the ADX calculation function is invoked THEN it SHALL calculate directional movements (+DM, -DM), true range (TR), and smooth them over period=14 before computing DX and ADX
3. WHEN the ATR calculation function is invoked THEN it SHALL calculate true range as max(high-low, abs(high-prev_close), abs(low-prev_close)) and smooth over period=14
4. WHEN indicator calculation functions are tested THEN they SHALL produce identical results when given the same input data across multiple runs
5. WHEN indicator calculation functions are tested with historical data THEN they SHALL produce results that match established technical analysis libraries (e.g., TA-Lib, pandas-ta)

### Requirement 4

**User Story:** As a trading model consumer, I want the observation vector fed to models to accurately represent current market conditions, so that models can learn and make decisions based on true market signals.

#### Acceptance Criteria

1. WHEN the observation vector is constructed THEN it SHALL include correctly calculated RSI, ADX, and ATR values
2. WHEN the observation vector is normalized THEN the normalization parameters (mean, std) SHALL be computed from the current market data, not from stale training data
3. WHEN the observation vector is constructed THEN it SHALL include a timestamp indicating when the underlying market data was collected
4. WHEN the observation vector is passed to trading models THEN the models SHALL receive data that reflects current market regime (bullish, bearish, ranging) accurately
5. WHEN market conditions change significantly THEN the observation vector SHALL reflect the new conditions within one calculation cycle (≤5 minutes)

### Requirement 5

**User Story:** As a system maintainer, I want to detect and prevent the use of mock or test data in production, so that trading decisions are always based on real market data.

#### Acceptance Criteria

1. WHEN the paper trading monitor starts THEN it SHALL verify that it is configured to use real Binance API credentials, not test/mock credentials
2. WHEN the paper trading monitor retrieves price data THEN it SHALL verify that the data source is Binance production API, not a mock or test data provider
3. WHEN the paper trading monitor calculates indicators THEN it SHALL verify that no hardcoded or cached indicator values are being used
4. WHEN the paper trading monitor detects that it is using mock or test data THEN it SHALL immediately halt trading and log a critical error
5. WHEN the system is deployed to production THEN it SHALL have automated checks that prevent any mock data from being used in trading decisions

### Requirement 6

**User Story:** As a system debugger, I want comprehensive logging of the indicator calculation pipeline, so that I can trace data corruption to its source and verify fixes.

#### Acceptance Criteria

1. WHEN the paper trading monitor calculates indicators THEN it SHALL log: timestamp, raw OHLCV data, intermediate calculation values, final indicator values, and market regime classification
2. WHEN the paper trading monitor compares calculated indicators to reference values THEN it SHALL log: calculated value, reference value, deviation percentage, and pass/fail status
3. WHEN the paper trading monitor detects an anomaly THEN it SHALL log sufficient detail to reconstruct the calculation and identify the root cause
4. WHEN the system is in debug mode THEN it SHALL log every step of the indicator calculation with intermediate values
5. WHEN the system detects data corruption THEN the logs SHALL be automatically exported to a diagnostic file for analysis

