# Data Flow Fix - Requirements

## Introduction

The ADAN trading system has a critical data flow break preventing real-time trading. WebSocket connects successfully but data doesn't reach the model, dashboard, or trading engine. This spec addresses the 5 critical issues blocking the complete data pipeline.

## Glossary

- **WebSocket Manager**: Real-time market data streaming from Binance
- **LiveDataManager**: Manages historical + real-time data integration
- **Environment**: Trading environment that processes observations
- **Dashboard**: Real-time monitoring interface
- **Indicators**: Technical analysis calculations (RSI, ADX, ATR)
- **Observations**: Feature vectors fed to the trading model

## Requirements

### Requirement 1: Fix Missing API Keys

**User Story:** As a trader, I want the system to use Binance testnet API keys, so that I can fetch historical data and initialize the trading environment.

#### Acceptance Criteria

1. WHEN the system initializes LiveDataManager THEN it SHALL successfully retrieve API keys from environment variables
2. WHEN API keys are missing THEN the system SHALL provide clear error messages indicating which variables are needed
3. WHEN the system connects to Binance testnet THEN it SHALL successfully fetch historical OHLCV data for configured assets

### Requirement 2: Fix Missing clean_worker_id Function

**User Story:** As a developer, I want the environment to properly normalize worker IDs, so that the trading system can track multiple workers correctly.

#### Acceptance Criteria

1. WHEN the environment initializes with a worker_id parameter THEN it SHALL normalize it to an integer (e.g., 'W0' → 0, 'W1' → 1)
2. WHEN worker_id is None THEN the system SHALL default to worker_id 0
3. WHEN worker_id is already an integer THEN the system SHALL use it as-is
4. WHEN worker_id is a string like 'W0' or 'w1' THEN the system SHALL extract the numeric part

### Requirement 3: Fix Dashboard Data Collection

**User Story:** As a trader, I want the dashboard to collect real trading data, so that I can monitor live performance.

#### Acceptance Criteria

1. WHEN the dashboard collector is initialized THEN it SHALL successfully connect to the ADAN system
2. WHEN the dashboard requests portfolio state THEN it SHALL return current positions and capital
3. WHEN the dashboard requests market context THEN it SHALL return current price and market data
4. WHEN the dashboard requests system health THEN it SHALL return system status and metrics

### Requirement 4: Fix Indicator Calculator Interface

**User Story:** As a developer, I want the indicator calculator to work with the observation builder, so that technical indicators are calculated correctly.

#### Acceptance Criteria

1. WHEN the observation builder calls IndicatorCalculator THEN it SHALL use static methods correctly
2. WHEN calculating RSI THEN the system SHALL return a value between 0-100
3. WHEN calculating ADX THEN the system SHALL return a value between 0-100
4. WHEN calculating ATR THEN the system SHALL return a positive value in price units

### Requirement 5: Fix Observation Builder Interface

**User Story:** As a developer, I want the observation builder to construct feature vectors correctly, so that the model receives valid input.

#### Acceptance Criteria

1. WHEN the observation builder is initialized THEN it SHALL not require a config parameter
2. WHEN building observations from market data THEN it SHALL return a numpy array of features
3. WHEN market data is missing THEN the system SHALL handle gracefully and return default observations
4. WHEN observations are built THEN they SHALL include all required indicators (RSI, ADX, ATR)

### Requirement 6: Verify Complete Data Flow

**User Story:** As a trader, I want to verify that data flows correctly from WebSocket to model, so that I can trust the trading system.

#### Acceptance Criteria

1. WHEN the system starts THEN WebSocket SHALL connect to Binance testnet
2. WHEN WebSocket receives market data THEN LiveDataManager SHALL process it
3. WHEN LiveDataManager has data THEN Environment SHALL build observations
4. WHEN observations are built THEN Dashboard SHALL display real-time data
5. WHEN all components work THEN Model SHALL execute trades based on real data
