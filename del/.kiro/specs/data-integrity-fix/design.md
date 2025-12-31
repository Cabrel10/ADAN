# Design Document: Data Integrity Fix for Market Indicators

## Overview

The data integrity fix establishes a reliable indicator calculation pipeline with real-time validation against authoritative market data. The solution consists of three main components:

1. **Indicator Calculator** - Implements correct RSI, ADX, ATR formulas using standard technical analysis methods
2. **Data Validator** - Compares calculated indicators against Binance reference data and detects corruption
3. **Observation Builder** - Constructs normalized feature vectors with validated indicators and current market regime

The design ensures that trading models always receive accurate market context by validating every calculation against real market data and halting trading if corruption is detected.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Paper Trading Monitor                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Binance API     │         │  Data Logger     │          │
│  │  (Real Data)     │         │  (Audit Trail)   │          │
│  └────────┬─────────┘         └────────▲─────────┘          │
│           │                            │                     │
│           ▼                            │                     │
│  ┌──────────────────────────────────────────────┐           │
│  │   Indicator Calculator                       │           │
│  │  ┌─────────────┐ ┌─────────────┐ ┌────────┐ │           │
│  │  │ RSI Calc    │ │ ADX Calc    │ │ ATR    │ │           │
│  │  │ (Wilder's)  │ │ (DM Method) │ │ Calc   │ │           │
│  │  └─────────────┘ └─────────────┘ └────────┘ │           │
│  └────────┬─────────────────────────────────────┘           │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐           │
│  │   Data Validator                             │           │
│  │  ┌──────────────────────────────────────┐   │           │
│  │  │ Compare vs Binance Reference Values  │   │           │
│  │  │ Detect Deviation > 5% (warning)      │   │           │
│  │  │ Detect Deviation > 10% (halt)        │   │           │
│  │  └──────────────────────────────────────┘   │           │
│  └────────┬─────────────────────────────────────┘           │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐           │
│  │   Observation Builder                        │           │
│  │  ┌──────────────────────────────────────┐   │           │
│  │  │ Construct Feature Vector             │   │           │
│  │  │ Normalize with Current Statistics    │   │           │
│  │  │ Classify Market Regime               │   │           │
│  │  │ Add Timestamp                        │   │           │
│  │  └──────────────────────────────────────┘   │           │
│  └────────┬─────────────────────────────────────┘           │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────┐           │
│  │   Trading Models (Ensemble)                  │           │
│  │  [W1] [W2] [W3] [W4]                         │           │
│  └──────────────────────────────────────────────┘           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Indicator Calculator

**Purpose**: Calculate RSI, ADX, ATR using standard technical analysis formulas

**Interface**:
```python
class IndicatorCalculator:
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float
    def calculate_all(ohlcv: pd.DataFrame) -> Dict[str, float]
```

**Key Methods**:
- `calculate_rsi()`: Implements Wilder's smoothing method
- `calculate_adx()`: Implements directional movement method
- `calculate_atr()`: Implements true range smoothing
- `calculate_all()`: Orchestrates all calculations with logging

**Responsibilities**:
- Validate input data (sufficient history, no NaN values)
- Implement mathematically correct formulas
- Log intermediate calculation steps
- Return calculated indicator values

### 2. Data Validator

**Purpose**: Validate calculated indicators against Binance reference data

**Interface**:
```python
class DataValidator:
    def get_reference_indicators(symbol: str, timeframe: str) -> Dict[str, float]
    def validate_indicators(calculated: Dict[str, float], reference: Dict[str, float]) -> ValidationResult
    def check_data_freshness(timestamp: datetime) -> bool
    def check_mock_data_usage() -> bool
```

**Key Methods**:
- `get_reference_indicators()`: Fetch real indicators from Binance
- `validate_indicators()`: Compare calculated vs reference with deviation thresholds
- `check_data_freshness()`: Ensure data is not older than 5 minutes
- `check_mock_data_usage()`: Verify real API credentials are in use

**Responsibilities**:
- Fetch reference data from Binance API
- Calculate deviation percentages
- Determine pass/fail/halt status
- Log validation results
- Trigger alerts on corruption detection

### 3. Observation Builder

**Purpose**: Construct normalized feature vectors with validated indicators

**Interface**:
```python
class ObservationBuilder:
    def build_observation(indicators: Dict[str, float], market_data: Dict) -> np.ndarray
    def normalize_observation(obs: np.ndarray, stats: Dict) -> np.ndarray
    def classify_market_regime(indicators: Dict[str, float]) -> str
    def add_timestamp(obs: np.ndarray, ts: datetime) -> np.ndarray
```

**Key Methods**:
- `build_observation()`: Construct feature vector from validated indicators
- `normalize_observation()`: Apply normalization using current market statistics
- `classify_market_regime()`: Determine bullish/bearish/ranging based on indicators
- `add_timestamp()`: Include data collection timestamp

**Responsibilities**:
- Combine validated indicators into feature vector
- Normalize using current market statistics (not stale training data)
- Classify market regime accurately
- Include metadata for traceability
- Ensure observation reflects current market conditions

## Data Models

### IndicatorValues
```python
@dataclass
class IndicatorValues:
    rsi: float              # 0-100
    adx: float              # 0-100
    atr: float              # in price units
    atr_percent: float      # as percentage of price
    timestamp: datetime
    data_freshness_seconds: int
```

### ValidationResult
```python
@dataclass
class ValidationResult:
    status: str             # "pass", "warning", "halt"
    rsi_deviation: float    # percentage
    adx_deviation: float    # percentage
    atr_deviation: float    # percentage
    reference_values: Dict[str, float]
    calculated_values: Dict[str, float]
    timestamp: datetime
    message: str
```

### MarketRegime
```python
@dataclass
class MarketRegime:
    regime: str             # "bullish", "bearish", "ranging"
    trend_strength: str     # "strong", "moderate", "weak"
    volatility_level: str   # "high", "normal", "low"
    confidence: float       # 0-1
```

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: RSI Calculation Correctness

*For any* sequence of prices with sufficient history (≥14 periods), calculating RSI using Wilder's smoothing method SHALL produce a value between 0 and 100 that matches the RSI calculated by established technical analysis libraries (TA-Lib, pandas-ta) within 0.1% tolerance.

**Validates: Requirements 1.1, 3.1**

### Property 2: ADX Calculation Correctness

*For any* sequence of high, low, close prices with sufficient history (≥28 periods for smoothing), calculating ADX using the directional movement method SHALL produce a value between 0 and 100 that matches the ADX calculated by established technical analysis libraries within 0.1% tolerance.

**Validates: Requirements 1.2, 3.2**

### Property 3: ATR Calculation Correctness

*For any* sequence of high, low, close prices with sufficient history (≥14 periods), calculating ATR using true range smoothing SHALL produce a value that matches the ATR calculated by established technical analysis libraries within 0.1% tolerance.

**Validates: Requirements 1.3, 3.3**

### Property 4: Data Freshness Validation

*For any* market data retrieved from Binance, the timestamp of the data SHALL be within 5 minutes of the current time, and the system SHALL reject data older than 5 minutes.

**Validates: Requirements 1.4, 2.1**

### Property 5: Deviation Detection

*For any* calculated indicator value, if the deviation from the Binance reference value exceeds 5%, the system SHALL log a warning; if deviation exceeds 10%, the system SHALL halt trading and alert the operator.

**Validates: Requirements 2.2, 2.3**

### Property 6: Corruption Audit Trail

*For any* detected data corruption event, the system SHALL record and persist: timestamp, calculated values, reference values, deviation percentages, and root cause indicators for later analysis.

**Validates: Requirements 2.4, 6.1**

### Property 7: Mock Data Detection

*For any* system startup, the system SHALL verify that real Binance API credentials are configured and that no mock or test data providers are active before allowing trading to commence.

**Validates: Requirements 5.1, 5.2, 5.5**

### Property 8: Observation Accuracy

*For any* set of validated indicators, the observation vector constructed from them SHALL accurately represent the current market regime (bullish, bearish, ranging) and SHALL reflect market condition changes within one calculation cycle (≤5 minutes).

**Validates: Requirements 4.1, 4.4, 4.5**

### Property 9: Normalization Consistency

*For any* observation vector, normalization parameters (mean, standard deviation) SHALL be computed from current market data, not from stale training data, ensuring that normalized values accurately reflect current market conditions.

**Validates: Requirements 4.2**

### Property 10: Calculation Reproducibility

*For any* set of OHLCV data, calculating indicators multiple times SHALL produce identical results, demonstrating that the calculation pipeline is deterministic and free of state-dependent bugs.

**Validates: Requirements 3.4**

## Error Handling

### Data Corruption Detection
- **Trigger**: Calculated indicator deviates >5% from reference
- **Action**: Log warning, continue trading with caution
- **Escalation**: If deviation >10%, halt trading immediately

### Stale Data Detection
- **Trigger**: Market data older than 5 minutes
- **Action**: Reject data, request fresh data from Binance
- **Escalation**: If fresh data unavailable for >10 minutes, halt trading

### Mock Data Detection
- **Trigger**: System detects mock/test credentials or data source
- **Action**: Log critical error, halt trading immediately
- **Escalation**: Require manual intervention to restart

### Calculation Errors
- **Trigger**: Insufficient data history, NaN values, division by zero
- **Action**: Log error, skip calculation cycle, retry next cycle
- **Escalation**: If errors persist for >5 cycles, halt trading

## Testing Strategy

### Unit Testing

Unit tests verify specific examples and edge cases:

1. **RSI Calculation Tests**
   - Test with known price sequences that produce known RSI values
   - Test boundary conditions (all up, all down, flat prices)
   - Test with insufficient data (< 14 periods)

2. **ADX Calculation Tests**
   - Test with trending data (strong uptrend, strong downtrend)
   - Test with ranging data (no clear trend)
   - Test with insufficient data (< 28 periods)

3. **ATR Calculation Tests**
   - Test with high volatility data
   - Test with low volatility data
   - Test with gap scenarios

4. **Validation Tests**
   - Test deviation detection at 5% threshold
   - Test deviation detection at 10% threshold
   - Test data freshness validation

5. **Observation Builder Tests**
   - Test feature vector construction
   - Test normalization with current statistics
   - Test market regime classification

### Property-Based Testing

Property-based tests verify universal properties across many inputs using a testing framework like Hypothesis (Python):

1. **RSI Property Test**
   - Generate random price sequences
   - Calculate RSI using our implementation
   - Verify result is between 0-100
   - Verify result matches reference library within tolerance

2. **ADX Property Test**
   - Generate random OHLC sequences
   - Calculate ADX using our implementation
   - Verify result is between 0-100
   - Verify result matches reference library within tolerance

3. **ATR Property Test**
   - Generate random OHLC sequences
   - Calculate ATR using our implementation
   - Verify result is positive
   - Verify result matches reference library within tolerance

4. **Data Freshness Property Test**
   - Generate timestamps at various ages
   - Verify freshness check correctly identifies stale data
   - Verify 5-minute threshold is enforced

5. **Deviation Detection Property Test**
   - Generate calculated and reference values with known deviations
   - Verify 5% threshold triggers warning
   - Verify 10% threshold triggers halt

6. **Reproducibility Property Test**
   - Generate random OHLCV data
   - Calculate indicators multiple times
   - Verify results are identical across runs

**Configuration**: Each property-based test SHALL run minimum 100 iterations with randomly generated inputs to ensure broad coverage.

