# ADAN BTC/USDT Terminal Dashboard - Developer Guide

## Architecture Overview

The dashboard follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard Application                    │
│                    (app.py)                                 │
├─────────────────────────────────────────────────────────────┤
│                    Dashboard Generator                      │
│                    (generator.py)                           │
├─────────────────────────────────────────────────────────────┤
│  Layout Manager  │  Section Renderers  │  Data Aggregator  │
│  (layout.py)     │  (sections/)        │  (aggregator.py)  │
├─────────────────────────────────────────────────────────────┤
│  Data Collector  │  Data Cache  │  Formatters  │  Colors   │
│  (collectors/)   │  (cache.py)  │  (formatters)│ (colors.py)│
├─────────────────────────────────────────────────────────────┤
│                    Data Models                              │
│                    (models.py)                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules

#### `models.py`
Defines data structures:
- `Position`: Open trading position
- `ClosedTrade`: Completed trade with outcome
- `Signal`: ADAN trading signal
- `MarketContext`: Market indicators and context
- `PortfolioState`: Overall portfolio state
- `Alert`: System alert
- `SystemHealth`: System health metrics

#### `data_collector.py`
Abstract base class for data collection:
```python
class DataCollector(ABC):
    @abstractmethod
    def connect(self): pass
    
    @abstractmethod
    def get_portfolio_state(self) -> PortfolioState: pass
    
    @abstractmethod
    def get_current_signal(self) -> Signal: pass
    
    # ... other methods
```

#### `mock_collector.py`
Mock data collector for testing:
- Generates realistic test data
- No external dependencies
- Useful for development and testing

#### `real_collector.py`
Real data collector:
- Integrates with ADAN components
- Fetches live portfolio data
- Connects to exchange APIs

### Formatting & Colors

#### `formatters.py`
Formatting functions:
- `format_usd(value)` → "$X,XXX.XX"
- `format_btc(value)` → "0.XXXX"
- `format_percentage(value)` → "X.XX%"
- `format_time(duration)` → "HH:MM:SS"
- `format_confidence(value)` → "0.XX"
- `format_price(value)` → "$X,XXX.XX"
- `format_ratio(value)` → "X.XX"

#### `colors.py`
Color mapping functions:
- `get_pnl_color(pnl_pct)` → Color based on P&L
- `get_confidence_color(confidence)` → Color based on confidence
- `get_risk_color(risk_pct)` → Color based on risk
- `get_signal_color(direction)` → Color based on signal
- `get_status_color(status)` → Color based on status

### Section Renderers

Each section is a separate module in `sections/`:

#### `header.py`
```python
def render_header(portfolio: PortfolioState) -> Panel:
    """Render portfolio overview header"""
```

#### `decision_matrix.py`
```python
def render_decision_matrix(signal: Signal, market: MarketContext) -> Panel:
    """Render trading signal and market context"""
```

#### `positions.py`
```python
def render_positions(positions: List[Position]) -> Panel:
    """Render active positions table"""
```

#### `closed_trades.py`
```python
def render_closed_trades(trades: List[ClosedTrade]) -> Panel:
    """Render closed trades table"""
```

#### `performance.py`
```python
def render_performance(trades: List[ClosedTrade]) -> Panel:
    """Render performance analytics"""
```

#### `system_health.py`
```python
def render_system_health(health_data: Dict) -> Panel:
    """Render system health status"""
```

### Layout & Generation

#### `layout.py`
```python
def create_layout() -> Layout:
    """Create main dashboard layout"""

def get_optimal_layout(console: Console) -> Layout:
    """Get layout optimized for terminal size"""

def update_layout_content(layout: Layout, sections: Dict) -> None:
    """Update layout with rendered sections"""
```

#### `generator.py`
```python
def generate_dashboard(collector: DataCollector, console: Console) -> Layout:
    """Generate complete dashboard with all sections"""
```

### Data Management

#### `cache.py`
```python
class DataCache:
    """Cache with TTL and staleness detection"""
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Store value with TTL"""
    
    def get(self, key: str) -> Tuple[Any, bool]:
        """Get value and staleness flag"""
    
    def is_stale(self, key: str) -> bool:
        """Check if data is stale"""
```

#### `aggregator.py`
```python
class DataAggregator:
    """Aggregate data from multiple sources"""
    
    def aggregate(self, collector: DataCollector) -> Dict:
        """Aggregate all data into dashboard format"""
```

### Application

#### `app.py`
```python
class AdanBtcDashboard:
    """Main dashboard application"""
    
    def __init__(self, collector: DataCollector, refresh_rate_ms: int = 500):
        pass
    
    def run(self) -> None:
        """Run dashboard with live updates"""
    
    def run_once(self) -> None:
        """Generate dashboard once and exit"""
```

## Adding New Sections

To add a new dashboard section:

1. Create a new file in `src/adan_trading_bot/dashboard/sections/`:

```python
# src/adan_trading_bot/dashboard/sections/custom_section.py
from rich.panel import Panel
from rich.table import Table

def render_custom_section(data: Dict) -> Panel:
    """Render custom section"""
    table = Table(show_header=True)
    table.add_column("Column 1")
    table.add_column("Column 2")
    
    for item in data.get("items", []):
        table.add_row(item["col1"], item["col2"])
    
    return Panel(table, title="Custom Section")
```

2. Update `generator.py` to include the new section:

```python
from .sections.custom_section import render_custom_section

def generate_dashboard(collector: DataCollector, console: Console) -> Layout:
    # ... existing code ...
    
    # Add custom section
    custom_data = collector.get_custom_data()
    sections["custom"] = render_custom_section(custom_data)
    
    # ... rest of code ...
```

3. Update `layout.py` to add the section to the layout:

```python
def create_layout() -> Layout:
    layout = Layout()
    
    # ... existing layout code ...
    
    layout["custom_area"] = Layout(name="custom")
    
    return layout
```

4. Write tests in `tests/test_dashboard_custom_section.py`

## Customizing Colors

To customize color schemes, edit `colors.py`:

```python
def get_pnl_color(pnl_pct: float) -> str:
    """Customize P&L color mapping"""
    if pnl_pct > 5.0:
        return "bold green"  # Large profit
    elif pnl_pct > 2.0:
        return "green"       # Profit
    elif pnl_pct > 0.0:
        return "yellow"      # Small profit
    # ... etc
```

## Integrating New Data Sources

To add a new data source:

1. Create a new collector in `src/adan_trading_bot/dashboard/collectors/`:

```python
# src/adan_trading_bot/dashboard/collectors/custom_collector.py
from ..data_collector import DataCollector
from ..models import PortfolioState, Signal, MarketContext

class CustomDataCollector(DataCollector):
    def connect(self) -> None:
        """Connect to data source"""
        pass
    
    def get_portfolio_state(self) -> PortfolioState:
        """Fetch portfolio state"""
        pass
    
    def get_current_signal(self) -> Signal:
        """Fetch current signal"""
        pass
    
    def get_market_context(self) -> MarketContext:
        """Fetch market context"""
        pass
    
    # ... implement other methods
```

2. Register the collector in `app.py`:

```python
COLLECTORS = {
    "mock": MockDataCollector,
    "real": RealDataCollector,
    "custom": CustomDataCollector,
}
```

## Testing

### Unit Tests

Test individual components:

```bash
pytest tests/test_dashboard_models.py -v
pytest tests/test_dashboard_formatters.py -v
pytest tests/test_dashboard_colors.py -v
pytest tests/test_dashboard_sections.py -v
```

### Property-Based Tests

Test correctness properties:

```bash
pytest tests/test_dashboard_properties.py -v
```

### Performance Tests

Test performance targets:

```bash
pytest tests/test_dashboard_performance.py -v
```

### Integration Tests

Test full dashboard:

```bash
pytest tests/test_dashboard_integration.py -v
pytest tests/test_dashboard_layout.py -v
```

### Running All Tests

```bash
pytest tests/test_dashboard_*.py -v
```

## Performance Optimization

### Rendering Optimization

1. **Cache static elements**: Don't re-render unchanged sections
2. **Minimize table recalculations**: Pre-calculate metrics
3. **Use efficient string formatting**: Avoid repeated conversions

### Memory Optimization

1. **Limit historical data**: Keep only recent trades
2. **Use generators**: For large data sets
3. **Clear caches periodically**: Prevent memory leaks

### CPU Optimization

1. **Adjust refresh rate**: Balance responsiveness vs CPU
2. **Batch updates**: Update multiple sections together
3. **Profile hot paths**: Use cProfile to identify bottlenecks

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Print Debug Information

```python
from rich.console import Console
console = Console()
console.print(f"[DEBUG] {variable}", style="dim")
```

### Use Python Debugger

```python
import pdb; pdb.set_trace()
```

## Code Style

Follow PEP 8 with these guidelines:

- Use type hints for all functions
- Write docstrings for all public functions
- Keep functions under 50 lines
- Use descriptive variable names
- Add comments for complex logic

## Documentation

### Docstring Format

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something is invalid
    """
```

### Inline Comments

```python
# Use comments for WHY, not WHAT
# Good: Multiply by 100 to convert to percentage
pct = value * 100

# Bad: Set x to 5
x = 5
```

## Version Control

### Commit Messages

```
[FEATURE] Add custom section renderer
[FIX] Correct P&L calculation for short positions
[PERF] Optimize dashboard rendering
[TEST] Add property tests for color coding
[DOCS] Update developer guide
```

### Branch Naming

```
feature/custom-section
fix/pnl-calculation
perf/rendering-optimization
test/property-tests
```

## Deployment

### Building

```bash
python setup.py build
```

### Testing Before Deployment

```bash
pytest tests/ -v
python -m pytest tests/ --cov=src/adan_trading_bot/dashboard
```

### Deployment Checklist

- [ ] All tests passing
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Performance targets met
- [ ] No breaking changes
- [ ] Version bumped

## Troubleshooting Development

### Import Errors

```bash
# Ensure package is installed in development mode
pip install -e .
```

### Test Failures

```bash
# Run with verbose output
pytest tests/ -vv

# Run specific test
pytest tests/test_dashboard_models.py::TestPosition::test_pnl_calculation -vv
```

### Performance Issues

```bash
# Profile code
python -m cProfile -s cumtime scripts/adan_btc_dashboard.py

# Use memory profiler
pip install memory-profiler
python -m memory_profiler scripts/adan_btc_dashboard.py
```

## Resources

- [Rich Documentation](https://rich.readthedocs.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Pytest Documentation](https://docs.pytest.org/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Write tests
5. Update documentation
6. Submit a pull request

## Support

For development questions or issues:
1. Check existing documentation
2. Review test examples
3. Check git history for similar changes
4. Contact the development team
