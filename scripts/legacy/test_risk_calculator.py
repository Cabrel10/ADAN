#!/usr/bin/env python3
"""
Test script for the RiskCalculator component.

This script tests:
1. VaR and CVaR calculations with different methods
2. Maximum drawdown and recovery metrics
3. Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
4. Volatility analysis
5. Portfolio risk assessment
6. Comprehensive risk reporting
"""

import sys
import os
import logging
from typing import Dict, Any, List
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adan_trading_bot.risk_management.risk_calculator import RiskCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config() -> Dict[str, Any]:
    """Create a test configuration for the RiskCalculator."""
    return {
        'risk_calculation': {
            'confidence_levels': [0.95, 0.99],
            'lookback_periods': [30, 60, 252],
            'risk_free_rate': 0.02,
            'max_drawdown_threshold': 0.2,
            'var_threshold': 0.05,
            'volatility_threshold': 0.3
        }
    }

def generate_test_returns(n_points: int = 252, volatility: float = 0.2, 
                         drift: float = 0.1, seed: int = 42) -> List[float]:
    """Generate synthetic return data for testing."""
    np.random.seed(seed)
    
    # Generate daily returns with specified volatility and drift
    daily_vol = volatility / np.sqrt(252)
    daily_drift = drift / 252
    
    returns = np.random.normal(daily_drift, daily_vol, n_points)
    return returns.tolist()

def generate_equity_curve(returns: List[float], initial_value: float = 10000.0) -> List[float]:
    """Generate equity curve from returns."""
    equity = [initial_value]
    
    for ret in returns:
        new_value = equity[-1] * (1 + ret)
        equity.append(new_value)
    
    return equity

def test_var_calculations():
    """Test VaR calculations with different methods."""
    logger.info("Testing VaR calculations...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Generate test data
    returns = generate_test_returns(n_points=1000, volatility=0.2)
    
    # Test historical VaR
    var_hist = risk_calc.calculate_var(returns, confidence_level=0.95, method='historical')
    assert var_hist >= 0, f"VaR should be non-negative, got {var_hist}"
    assert var_hist < 1.0, f"VaR should be reasonable, got {var_hist}"
    
    # Test parametric VaR
    var_param = risk_calc.calculate_var(returns, confidence_level=0.95, method='parametric')
    assert var_param >= 0, f"Parametric VaR should be non-negative, got {var_param}"
    
    # Test Monte Carlo VaR
    var_mc = risk_calc.calculate_var(returns, confidence_level=0.95, method='monte_carlo')
    assert var_mc >= 0, f"Monte Carlo VaR should be non-negative, got {var_mc}"
    
    # VaR should increase with higher confidence level
    var_99 = risk_calc.calculate_var(returns, confidence_level=0.99, method='historical')
    assert var_99 >= var_hist, f"99% VaR should be >= 95% VaR, got {var_99} vs {var_hist}"
    
    # Test with insufficient data
    var_empty = risk_calc.calculate_var([], confidence_level=0.95)
    assert var_empty == 0.0, f"VaR with no data should be 0, got {var_empty}"
    
    logger.info("✅ VaR calculations test passed")

def test_cvar_calculations():
    """Test CVaR calculations."""
    logger.info("Testing CVaR calculations...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Generate test data with some extreme losses
    returns = generate_test_returns(n_points=1000, volatility=0.3)
    
    # Calculate CVaR
    cvar_95 = risk_calc.calculate_cvar(returns, confidence_level=0.95)
    var_95 = risk_calc.calculate_var(returns, confidence_level=0.95)
    
    # CVaR should be >= VaR
    assert cvar_95 >= var_95, f"CVaR should be >= VaR, got {cvar_95} vs {var_95}"
    assert cvar_95 >= 0, f"CVaR should be non-negative, got {cvar_95}"
    
    # Test with insufficient data
    cvar_empty = risk_calc.calculate_cvar([])
    assert cvar_empty == 0.0, f"CVaR with no data should be 0, got {cvar_empty}"
    
    logger.info("✅ CVaR calculations test passed")

def test_drawdown_calculations():
    """Test maximum drawdown calculations."""
    logger.info("Testing drawdown calculations...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Create a simple equity curve with known drawdown
    equity_curve = [1000, 1100, 1200, 1000, 800, 900, 1300, 1400]
    
    dd_metrics = risk_calc.calculate_maximum_drawdown(equity_curve)
    
    # Check that all expected keys are present
    expected_keys = ['max_drawdown', 'max_drawdown_pct', 'drawdown_duration', 
                    'recovery_time', 'current_drawdown']
    for key in expected_keys:
        assert key in dd_metrics, f"Missing key {key} in drawdown metrics"
    
    # Maximum drawdown should be from 1200 to 800 = 400
    assert abs(dd_metrics['max_drawdown'] - 400) < 1, \
        f"Expected max drawdown ~400, got {dd_metrics['max_drawdown']}"
    
    # Maximum drawdown percentage should be 400/1200 = 33.33%
    expected_pct = 400 / 1200
    assert abs(dd_metrics['max_drawdown_pct'] - expected_pct) < 0.01, \
        f"Expected max drawdown pct ~{expected_pct:.2%}, got {dd_metrics['max_drawdown_pct']:.2%}"
    
    # Test with insufficient data
    dd_empty = risk_calc.calculate_maximum_drawdown([])
    assert dd_empty['max_drawdown'] == 0.0, "Empty data should return 0 drawdown"
    
    logger.info("✅ Drawdown calculations test passed")

def test_risk_adjusted_returns():
    """Test risk-adjusted return metrics."""
    logger.info("Testing risk-adjusted return metrics...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Generate returns with positive drift
    returns = generate_test_returns(n_points=252, volatility=0.2, drift=0.1)
    equity_curve = generate_equity_curve(returns)
    
    # Test Sharpe ratio
    sharpe = risk_calc.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float), f"Sharpe ratio should be float, got {type(sharpe)}"
    # With positive drift and reasonable volatility, Sharpe should be positive
    assert sharpe > -2.0, f"Sharpe ratio seems unreasonable: {sharpe}"
    
    # Test Sortino ratio
    sortino = risk_calc.calculate_sortino_ratio(returns)
    assert isinstance(sortino, float), f"Sortino ratio should be float, got {type(sortino)}"
    # Sortino should generally be higher than Sharpe (less penalty for upside volatility)
    
    # Test Calmar ratio
    calmar = risk_calc.calculate_calmar_ratio(returns, equity_curve)
    assert isinstance(calmar, float), f"Calmar ratio should be float, got {type(calmar)}"
    
    # Test with insufficient data
    sharpe_empty = risk_calc.calculate_sharpe_ratio([])
    assert sharpe_empty == 0.0, "Empty data should return 0 Sharpe ratio"
    
    logger.info("✅ Risk-adjusted returns test passed")

def test_volatility_metrics():
    """Test volatility analysis."""
    logger.info("Testing volatility metrics...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Generate returns with known volatility
    returns = generate_test_returns(n_points=252, volatility=0.2)
    
    vol_metrics = risk_calc.calculate_volatility_metrics(returns)
    
    # Check that all expected keys are present
    expected_keys = ['volatility', 'annualized_volatility', 'upside_volatility', 
                    'downside_volatility', 'volatility_skew']
    for key in expected_keys:
        assert key in vol_metrics, f"Missing key {key} in volatility metrics"
    
    # Annualized volatility should be roughly sqrt(252) times daily volatility
    daily_vol = vol_metrics['volatility']
    annual_vol = vol_metrics['annualized_volatility']
    expected_annual = daily_vol * np.sqrt(252)
    
    assert abs(annual_vol - expected_annual) < 0.01, \
        f"Annualized volatility calculation error: {annual_vol} vs {expected_annual}"
    
    # All volatilities should be non-negative
    for key in ['volatility', 'annualized_volatility', 'upside_volatility', 'downside_volatility']:
        assert vol_metrics[key] >= 0, f"{key} should be non-negative, got {vol_metrics[key]}"
    
    logger.info("✅ Volatility metrics test passed")

def test_portfolio_risk():
    """Test portfolio risk calculations."""
    logger.info("Testing portfolio risk...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Create mock portfolio positions
    positions = {
        'BTC': {'weight': 0.6, 'var': 0.05},
        'ETH': {'weight': 0.3, 'var': 0.04},
        'ADA': {'weight': 0.1, 'var': 0.06}
    }
    
    # Test without correlations (conservative estimate)
    portfolio_risk = risk_calc.calculate_portfolio_risk(positions)
    
    expected_keys = ['portfolio_var', 'portfolio_volatility', 'concentration_risk']
    for key in expected_keys:
        assert key in portfolio_risk, f"Missing key {key} in portfolio risk"
    
    # Portfolio VaR should be reasonable
    assert portfolio_risk['portfolio_var'] >= 0, "Portfolio VaR should be non-negative"
    assert portfolio_risk['portfolio_var'] <= 0.1, "Portfolio VaR should be reasonable"
    
    # Concentration risk should be sum of squared weights
    expected_concentration = 0.6**2 + 0.3**2 + 0.1**2
    assert abs(portfolio_risk['concentration_risk'] - expected_concentration) < 0.01, \
        f"Concentration risk calculation error"
    
    # Test with correlations
    correlations = {
        'BTC': {'BTC': 1.0, 'ETH': 0.7, 'ADA': 0.5},
        'ETH': {'BTC': 0.7, 'ETH': 1.0, 'ADA': 0.6},
        'ADA': {'BTC': 0.5, 'ETH': 0.6, 'ADA': 1.0}
    }
    
    portfolio_risk_corr = risk_calc.calculate_portfolio_risk(positions, correlations)
    
    # With positive correlations, diversified VaR should be lower than sum
    total_individual_var = sum(pos['weight'] * pos['var'] for pos in positions.values())
    assert portfolio_risk_corr['portfolio_var'] <= total_individual_var, \
        "Diversified portfolio VaR should be lower than sum of individual VaRs"
    
    logger.info("✅ Portfolio risk test passed")

def test_risk_assessment():
    """Test overall risk level assessment."""
    logger.info("Testing risk assessment...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Test low risk scenario
    low_risk_metrics = {
        'max_drawdown_pct': 0.05,
        'var_95': 0.01,
        'annualized_volatility': 0.1,
        'sharpe_ratio': 1.5
    }
    
    risk_level = risk_calc.assess_risk_level(low_risk_metrics)
    assert risk_level == 'LOW', f"Expected LOW risk, got {risk_level}"
    
    # Test high risk scenario
    high_risk_metrics = {
        'max_drawdown_pct': 0.4,
        'var_95': 0.15,
        'annualized_volatility': 0.6,
        'sharpe_ratio': -0.5
    }
    
    risk_level = risk_calc.assess_risk_level(high_risk_metrics)
    assert risk_level in ['HIGH', 'EXTREME'], f"Expected HIGH/EXTREME risk, got {risk_level}"
    
    logger.info("✅ Risk assessment test passed")

def test_comprehensive_risk_report():
    """Test comprehensive risk report generation."""
    logger.info("Testing comprehensive risk report...")
    
    config = create_test_config()
    risk_calc = RiskCalculator(config)
    
    # Generate test data
    returns = generate_test_returns(n_points=252, volatility=0.25, drift=0.08)
    equity_curve = generate_equity_curve(returns)
    
    # Generate comprehensive report
    report = risk_calc.calculate_comprehensive_risk_report(returns, equity_curve)
    
    # Check that report contains expected sections
    expected_keys = ['timestamp', 'data_points', 'var_95', 'cvar_95', 'max_drawdown_pct',
                    'sharpe_ratio', 'sortino_ratio', 'volatility', 'risk_level', 'warnings']
    
    for key in expected_keys:
        assert key in report, f"Missing key {key} in comprehensive report"
    
    # Check data consistency
    assert report['data_points'] == len(returns), "Data points count mismatch"
    assert report['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'], \
        f"Invalid risk level: {report['risk_level']}"
    
    # Test with insufficient data
    empty_report = risk_calc.calculate_comprehensive_risk_report([], [])
    assert 'error' in empty_report, "Empty data should return error in report"
    
    logger.info("✅ Comprehensive risk report test passed")

def run_all_tests():
    """Run all risk calculator tests."""
    logger.info("🚀 Starting risk calculator tests...")
    
    try:
        test_var_calculations()
        test_cvar_calculations()
        test_drawdown_calculations()
        test_risk_adjusted_returns()
        test_volatility_metrics()
        test_portfolio_risk()
        test_risk_assessment()
        test_comprehensive_risk_report()
        
        logger.info("🎉 All risk calculator tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)