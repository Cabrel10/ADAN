import logging
import os
import sys
import yaml

# Add project root to path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.adan_trading_bot.risk_management.risk_assessor import RiskAssessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_risk_metrics_config():
    """Tests if the risk metrics are configured correctly."""
    logging.info("Testing risk metrics configuration...")
    with open("/home/morningstar/Documents/trading/ADAN/config/risk_metrics_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    risk_assessor = RiskAssessor(config)

    # Check if the metrics are enabled
    assert risk_assessor.config.get("risk_metrics", {}).get("enabled") is True

    # Check if the log interval is set correctly
    assert risk_assessor.config.get("risk_metrics", {}).get("log_interval") == 100

    # Check if the metrics are set correctly
    expected_metrics = [
        "var",
        "cvar",
        "drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "volatility",
    ]
    assert risk_assessor.config.get("risk_metrics", {}).get("metrics") == expected_metrics

    logging.info("✅ Risk metrics configuration test passed")

def run_all_tests():
    """Runs all tests for the risk metrics configuration."""
    logging.info("--- Starting Risk Metrics Configuration tests ---")
    try:
        test_risk_metrics_config()
        logging.info("--- All Risk Metrics Configuration tests passed! ---")
    except Exception as e:
        logging.error(f"❌ A test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()