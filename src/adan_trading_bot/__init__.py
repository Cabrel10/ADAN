"""ADAN Trading Bot Package."""

# Import core components for easier access
from .data_processing import feature_engineer
from .environment import MultiAssetChunkedEnv
from .trading import OrderManager
from .portfolio import PortfolioManager

__version__ = '0.1.0'

# Initialize logging
import logging

# Configure root logger if not already configured
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Create package logger
logger = logging.getLogger(__name__)
logger.info(f"ADAN Trading Bot v{__version__} initialized")

# Export public API
__all__ = [
    'feature_engineer',
    'MultiAssetChunkedEnv',
    'OrderManager',
    'PortfolioManager',
    'logger'
]
