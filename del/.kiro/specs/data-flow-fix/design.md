# Data Flow Fix - Design

## Overview

This design addresses 5 critical issues preventing data flow from WebSocket to trading model. The fixes are minimal and focused on restoring the complete pipeline without architectural changes.

## Architecture

```
WebSocket (✅ Working)
    ↓
LiveDataManager (❌ Missing API keys)
    ↓
Environment (❌ Missing clean_worker_id)
    ↓
Observations (❌ Wrong constructor)
    ↓
Dashboard (❌ Wrong method names)
    ↓
Model (❌ No data)
```

## Components and Interfaces

### 1. API Key Management

**Current Issue:** LiveDataManager fails because Binance testnet API keys are not in environment

**Solution:**
- Ensure `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_SECRET_KEY` are set
- Add validation in connector.py to check for keys before creating exchange client
- Provide clear error messages if keys are missing

**Interface:**
```python
def get_exchange_client(config, use_testnet=True):
    # Validates API keys exist
    # Returns exchange client or raises ValueError with clear message
```

### 2. Worker ID Normalization

**Current Issue:** `clean_worker_id()` function is called but not defined

**Solution:**
- Add `clean_worker_id()` function to normalize worker IDs
- Handle formats: 'W0', 'w1', 0, 1, None
- Default to 0 if invalid

**Interface:**
```python
def clean_worker_id(worker_id):
    """Normalize worker_id to integer"""
    # 'W0' → 0, 'W1' → 1, None → 0, 0 → 0
```

### 3. Dashboard Data Collection

**Current Issue:** RealDataCollector has wrong method names

**Solution:**
- Use existing methods: `get_portfolio_state()`, `get_market_context()`, `get_system_health()`
- Update diagnostic to call correct methods
- Verify data flows from file-based state

**Interface:**
```python
collector = RealDataCollector()
portfolio = collector.get_portfolio_state()  # ✅ Correct
market = collector.get_market_context()      # ✅ Correct
health = collector.get_system_health()       # ✅ Correct
```

### 4. Indicator Calculator

**Current Issue:** IndicatorCalculator uses static methods, not instance methods

**Solution:**
- Use static methods: `IndicatorCalculator.calculate_rsi()`, etc.
- No instantiation needed
- Pass data directly to static methods

**Interface:**
```python
rsi = IndicatorCalculator.calculate_rsi(close_prices)  # ✅ Static method
atr = IndicatorCalculator.calculate_atr(high, low, close)  # ✅ Static method
```

### 5. Observation Builder

**Current Issue:** ObservationBuilder constructor doesn't take config parameter

**Solution:**
- Initialize without config: `builder = ObservationBuilder()`
- Use static methods from IndicatorCalculator
- Build observations from market data directly

**Interface:**
```python
builder = ObservationBuilder()  # ✅ No config needed
obs = builder.build(market_data)  # ✅ Returns numpy array
```

## Data Models

### Market Data Structure
```python
market_data = {
    'BTCUSDT': {
        '5m': pd.DataFrame({
            'open', 'high', 'low', 'close', 'volume'
        }),
        '1h': pd.DataFrame(...),
        '4h': pd.DataFrame(...)
    }
}
```

### Observation Structure
```python
observation = {
    'features': np.array([...]),  # Feature vector
    'rsi': float,                  # 0-100
    'adx': float,                  # 0-100
    'atr': float,                  # Price units
    'regime': str,                 # 'bullish', 'bearish', 'ranging'
    'timestamp': datetime
}
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: API Keys Validation
*For any* system initialization, if Binance testnet is configured, the system SHALL have valid API keys available before attempting to create an exchange client.
**Validates: Requirements 1.1, 1.2**

### Property 2: Worker ID Normalization
*For any* worker_id input (string, int, or None), the clean_worker_id function SHALL return a non-negative integer.
**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

### Property 3: Dashboard Data Retrieval
*For any* RealDataCollector instance, calling get_portfolio_state() SHALL return a PortfolioState object with valid structure.
**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

### Property 4: Indicator Calculation
*For any* valid OHLCV data, IndicatorCalculator static methods SHALL return numeric values within expected ranges (RSI: 0-100, ADX: 0-100, ATR: > 0).
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 5: Observation Building
*For any* market data dictionary, ObservationBuilder.build() SHALL return a numpy array with consistent shape and valid indicator values.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4**

### Property 6: Complete Data Flow
*For any* trading session, data SHALL flow successfully from WebSocket → LiveDataManager → Environment → Observations → Dashboard without errors.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

## Error Handling

1. **Missing API Keys**: Raise ValueError with clear message indicating which environment variables are needed
2. **Invalid Worker ID**: Default to 0 and log warning
3. **Missing Market Data**: Return empty DataFrame and log warning
4. **Indicator Calculation Failure**: Return default values (RSI=50, ADX=20, ATR=0)
5. **Dashboard Connection Failure**: Log error but don't crash, allow retry

## Testing Strategy

### Unit Tests
- Test clean_worker_id with various inputs
- Test IndicatorCalculator static methods with sample data
- Test ObservationBuilder initialization and build method
- Test RealDataCollector method calls

### Property-Based Tests
- Property 1: API keys validation across different configurations
- Property 2: Worker ID normalization for all input types
- Property 3: Dashboard data retrieval consistency
- Property 4: Indicator calculations within valid ranges
- Property 5: Observation building produces valid arrays
- Property 6: Complete data flow without errors

### Integration Tests
- Test full data flow from WebSocket to model
- Test dashboard receives real-time data
- Test model executes trades based on observations
