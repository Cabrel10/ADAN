import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_asset_data(asset_name: str, start_date_str: str, end_date_str: str, base_price: float, volatility: float = 0.001, volume_mean: int = 10000) -> pd.DataFrame:
    """
    Generates synthetic OHLCV data for a single asset.
    """
    logger.info(f"Generating synthetic data for {asset_name} from {start_date_str} to {end_date_str} with base price {base_price:.2f}")
    timestamps = pd.date_range(start=start_date_str, end=end_date_str, freq='1min', tz='UTC')
    n_periods = len(timestamps)

    returns = np.random.normal(loc=0, scale=volatility, size=n_periods)
    prices = np.zeros(n_periods)
    prices[0] = base_price
    for t in range(1, n_periods):
        prices[t] = prices[t-1] * (1 + returns[t])
        if prices[t] <= 0: # Ensure price is positive
            prices[t] = prices[t-1] * 0.98

    df = pd.DataFrame(index=timestamps)
    df['open'] = prices
    df['close'] = np.roll(prices, -1)
    df.iloc[-1, df.columns.get_loc('close')] = df.iloc[-1, df.columns.get_loc('open')] * (1 + np.random.normal(loc=0, scale=volatility/2))

    min_val = df[['open', 'close']].min(axis=1)
    max_val = df[['open', 'close']].max(axis=1)

    df['low'] = min_val * (1 - np.random.uniform(0, volatility * 5, size=n_periods))
    df['high'] = max_val * (1 + np.random.uniform(0, volatility * 5, size=n_periods))

    df['low'] = np.minimum(df['low'], min_val)
    df['high'] = np.maximum(df['high'], max_val)

    df['volume'] = np.random.poisson(volume_mean, size=n_periods).astype(float)

    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df.index.name = 'timestamp'

    logger.info(f"Finished generating data for {asset_name}. Shape: {df.shape}")
    return df

def main():
    logger.info("Starting synthetic data generation process...")
    assets_to_generate = {
        "ADAUSDT": {"base_price": 0.4, "volatility": 0.002, "volume_mean": 50000},
        "BNBUSDT": {"base_price": 300, "volatility": 0.0015, "volume_mean": 2000},
        "BTCUSDT": {"base_price": 60000, "volatility": 0.001, "volume_mean": 100},
        "ETHUSDT": {"base_price": 3000, "volatility": 0.0012, "volume_mean": 500},
        "XRPUSDT": {"base_price": 0.5, "volatility": 0.0025, "volume_mean": 40000}
    }
    start_date = "2025-05-01 00:00:00"
    end_date = "2025-05-31 23:59:00"

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory set to: {output_dir.resolve()}")

    for asset, params in assets_to_generate.items():
        df_asset = generate_asset_data(asset, start_date, end_date, **params)
        output_path = output_dir / f"{asset}_1m_raw.parquet"
        try:
            df_asset.to_parquet(output_path)
            logger.info(f"Successfully generated synthetic data for {asset} to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save parquet file for {asset} at {output_path}: {e}")

    logger.info("Synthetic data generation process completed.")

if __name__ == "__main__":
    main()
