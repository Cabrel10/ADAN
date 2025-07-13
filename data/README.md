# Data Directory

This directory stores all market data used by the ADAN trading bot, including raw, processed, and merged datasets, as well as scalers and encoders.

## Important Subdirectories:

- `raw/`: Contains raw OHLCV (Open, High, Low, Close, Volume) market data, typically in 1-minute intervals.
- `processed/`: Stores data after initial processing, including calculated technical indicators. This directory is further organized by `lot` (e.g., `lot1`, `lot2`) and then by `unified` timeframe and asset.
  - `processed/unified/`: Contains processed data for each asset and timeframe, ready for merging.
  - `processed/merged/`: Contains the final merged datasets, split into `train`, `validation`, and `test` sets, used for training and evaluation.
- `scalers_encoders/`: Stores fitted data scalers (e.g., `StandardScaler`) and encoders used for normalizing and transforming features. These are crucial for consistent data preprocessing during training and inference.

## Usage:

Data in this directory is managed by scripts in the `scripts/` directory, particularly `convert_real_data.py` for initial processing and `merge_processed_data.py` for combining data across assets and timeframes.

## Related Documentation:

- [Main README.md](../../README.md)
- [Data Processing Pipeline (src/adan_trading_bot/data_processing/)](../src/adan_trading_bot/data_processing/README.md)
- [Scripts (scripts/)](../scripts/README.md)