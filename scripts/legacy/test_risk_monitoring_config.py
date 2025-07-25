
import logging
import os
import sys
import yaml

# Add project root to path to allow module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_risk_monitoring_config():
    """Tests if the risk monitoring is configured correctly."""
    logging.info("Testing risk monitoring configuration...")
    with open("/home/morningstar/Documents/trading/ADAN/config/risk_monitoring_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Check if the monitoring is enabled
    assert config.get("risk_monitoring", {}).get("enabled") is True

    # Check if the log level is set correctly
    assert config.get("risk_monitoring", {}).get("log_level") == "INFO"

    # Check if the log file is set correctly
    assert config.get("risk_monitoring", {}).get("log_file") == "logs/risk_monitoring.log"

    logging.info("✅ Risk monitoring configuration test passed")

def run_all_tests():
    """Runs all tests for the risk monitoring configuration."""
    logging.info("--- Starting Risk Monitoring Configuration tests ---")
    try:
        test_risk_monitoring_config()
        logging.info("--- All Risk Monitoring Configuration tests passed! ---")
    except Exception as e:
        logging.error(f"❌ A test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
