# Dashboard Real Data Integration - Requirements

## Introduction

The ADAN dashboard currently displays static mock data instead of real market data from Binance testnet. The model cannot trade because it receives no real market signals. This spec fixes the dashboard to display real-time market data, network metrics, and trading signals.

## Glossary

- **Dashboard**: Terminal UI displaying real-time trading data
- **RealDataCollector**: Component fetching live data from Binance and trading system
- **MockDataCollector**: Component generating fake data for testing
- **Market Data**: Real-time price, volume, and indicator data from Binance
- **Network Metrics**: API latency, WebSocket lag, feed status
- **Trading Signal**: BUY/SELL/HOLD decision from ensemble model
- **Portfolio State**: Current positions, capital, P&L
- **System Health**: API status, feed status, model status, resource usage

## Requirements

### Requirement 1: Real Market Data Display

**User Story:** As a trader, I want the dashboard to display real market data from Binance testnet, so that I can see actual market conditions and make informed trading decisions.

#### Acceptance Criteria

1. WHEN the dashboard starts THEN the system SHALL connect to RealDataCollector instead of MockDataCollector
2. WHEN the dashboard refreshes THEN the system SHALL fetch fresh market data from Binance testnet via the state file
3. WHEN market data is fetched THEN the system SHALL display current BTC price, volatility, RSI, ADX, and trend
4. WHEN market conditions change THEN the system SHALL update the dashboard display within the refresh interval
5. WHEN the state file is unavailable THEN the system SHALL display a clear error message and retry connection

### Requirement 2: Network Latency Metrics

**User Story:** As a system operator, I want to see network latency metrics on the dashboard, so that I can monitor system performance and detect connectivity issues.

#### Acceptance Criteria

1. WHEN the dashboard displays system health THEN the system SHALL show API latency in milliseconds
2. WHEN the dashboard displays system health THEN the system SHALL show WebSocket feed lag in milliseconds
3. WHEN the dashboard displays system health THEN the system SHALL show API connection status (OK/ERROR)
4. WHEN the dashboard displays system health THEN the system SHALL show feed connection status (OK/ERROR)
5. WHEN network latency exceeds threshold THEN the system SHALL display warning indicator

### Requirement 3: Trading Signal Display

**User Story:** As a trader, I want to see the current trading signal with confidence and worker votes, so that I can understand the model's trading decision.

#### Acceptance Criteria

1. WHEN the dashboard displays decision matrix THEN the system SHALL show current signal (BUY/SELL/HOLD)
2. WHEN the dashboard displays decision matrix THEN the system SHALL show signal confidence (0-100%)
3. WHEN the dashboard displays decision matrix THEN the system SHALL show individual worker votes
4. WHEN the dashboard displays decision matrix THEN the system SHALL show decision driver (which worker/ensemble)
5. WHEN signal changes THEN the system SHALL update display immediately on next refresh

### Requirement 4: Portfolio State Display

**User Story:** As a trader, I want to see real portfolio state including positions and trades, so that I can track my trading performance.

#### Acceptance Criteria

1. WHEN the dashboard displays portfolio THEN the system SHALL show total portfolio value in USD
2. WHEN the dashboard displays portfolio THEN the system SHALL show available capital
3. WHEN the dashboard displays positions THEN the system SHALL show all open positions with entry price, current price, P&L
4. WHEN the dashboard displays trades THEN the system SHALL show last 5 closed trades with entry, exit, duration, P&L
5. WHEN portfolio state changes THEN the system SHALL update display within refresh interval

### Requirement 5: System Health Monitoring

**User Story:** As a system operator, I want to see comprehensive system health metrics, so that I can ensure the trading system is operating correctly.

#### Acceptance Criteria

1. WHEN the dashboard displays system health THEN the system SHALL show CPU usage percentage
2. WHEN the dashboard displays system health THEN the system SHALL show memory usage in GB
3. WHEN the dashboard displays system health THEN the system SHALL show number of active threads
4. WHEN the dashboard displays system health THEN the system SHALL show system uptime percentage
5. WHEN any component fails THEN the system SHALL display alert with component name and error

### Requirement 6: Data Refresh Mechanism

**User Story:** As a trader, I want the dashboard to refresh data at configurable intervals, so that I can balance real-time updates with system performance.

#### Acceptance Criteria

1. WHEN the dashboard starts THEN the system SHALL use default refresh rate of 60 seconds
2. WHEN user specifies refresh rate THEN the system SHALL use specified rate (minimum 1 second)
3. WHEN refresh interval elapses THEN the system SHALL fetch fresh data from all sources
4. WHEN data fetch fails THEN the system SHALL retry with exponential backoff
5. WHEN user presses Ctrl+C THEN the system SHALL gracefully shutdown and close connections
