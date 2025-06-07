"""
Custom logger setup for the ADAN trading bot.
"""
import os
import logging
import yaml
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# Define a custom theme for rich
custom_theme = Theme({
    "info": "green",
    "warning": "yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "trade_buy": "bold blue",
    "trade_sell": "bold purple",
    "profit": "bold green",
    "loss": "bold red",
})

console = Console(theme=custom_theme)

def setup_logging(config_path=None, default_level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to the logging configuration file.
        default_level: Default logging level if config is not found.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("adan_trading_bot")
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Get log level from config
            log_level_str = config.get('level', 'INFO')
            log_level = getattr(logging, log_level_str.upper())
            
            # Get log file path if specified
            log_file = config.get('log_file', None)
            
            # Get rich traceback option
            rich_tracebacks = config.get('rich_tracebacks', True)
        except Exception as e:
            print(f"Error loading logging config: {e}")
            log_level = default_level
            log_file = None
            rich_tracebacks = True
    else:
        log_level = default_level
        log_file = None
        rich_tracebacks = True
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=rich_tracebacks,
                console=console,
                markup=True
            )
        ]
    )
    
    # Configure our module logger
    logger.setLevel(log_level)
    
    # Add file handler if log file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger():
    """
    Get the configured logger instance.
    
    Returns:
        logging.Logger: Logger instance.
    """
    return logging.getLogger("adan_trading_bot")

# Convenience functions for logging with rich formatting
def log_info(message):
    """Log an info message with rich formatting."""
    get_logger().info(f"[info]{message}[/info]")

def log_warning(message):
    """Log a warning message with rich formatting."""
    get_logger().warning(f"[warning]{message}[/warning]")

def log_error(message):
    """Log an error message with rich formatting."""
    get_logger().error(f"[error]{message}[/error]")

def log_critical(message):
    """Log a critical message with rich formatting."""
    get_logger().critical(f"[critical]{message}[/critical]")

def log_success(message):
    """Log a success message with rich formatting."""
    get_logger().info(f"[success]{message}[/success]")

def log_trade_buy(message):
    """Log a buy trade with rich formatting."""
    get_logger().info(f"[trade_buy]{message}[/trade_buy]")

def log_trade_sell(message):
    """Log a sell trade with rich formatting."""
    get_logger().info(f"[trade_sell]{message}[/trade_sell]")

def log_profit(message):
    """Log a profit message with rich formatting."""
    get_logger().info(f"[profit]{message}[/profit]")

def log_loss(message):
    """Log a loss message with rich formatting."""
    get_logger().info(f"[loss]{message}[/loss]")
