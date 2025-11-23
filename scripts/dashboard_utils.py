import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import shutil
import tempfile

def run_evaluation(asset, capital, checkpoint_path, start_date=None, end_date=None, timeframes=["5m", "1h", "4h"], max_steps=20000):
    """
    Runs a single evaluation episode for the given asset and capital.
    If start_date and end_date are provided, filters data to that range.
    """
    print(f"Starting evaluation for {asset} with ${capital}...")
    
    # 1. Load Config
    config_loader = ConfigLoader()
    config = config_loader.load_config("config/config.yaml")

    # 2. Override Config
    if 'environment' in config:
        config['environment']['assets'] = [asset]
    else:
        config['assets'] = [asset]
    
    config['initial_capital'] = float(capital)

    if 'workers' in config:
        for w_id in config['workers']:
            if isinstance(config['workers'][w_id], dict):
                config['workers'][w_id]['assets'] = [asset]

    # Ensure data dirs are correct
    original_train_dir = "data/processed/indicators/train"
    if 'data' in config:
        config['data']['data_dirs']['train'] = original_train_dir
        config['data']['data_dirs']['test'] = "data/processed/indicators/test"

    # Handle Date Filtering
    temp_dir = None
    if start_date and end_date:
        print(f"Filtering data from {start_date} to {end_date}...")
        temp_dir = tempfile.mkdtemp()
        asset_dir = os.path.join(temp_dir, asset)
        os.makedirs(asset_dir, exist_ok=True)
        
        for tf in timeframes:
            # Try loading original data
            original_path = f"{original_train_dir}/{asset}/{tf}.parquet"
            if not os.path.exists(original_path):
                 # Try chunk 0
                 original_path = f"{original_train_dir}/{asset}/{tf}/0.parquet"
            
            if os.path.exists(original_path):
                try:
                    df = pd.read_parquet(original_path)
                    # Filter
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df_filtered = df.loc[mask]
                    
                    if df_filtered.empty:
                        print(f"Warning: No data found for {tf} in range {start_date}-{end_date}")
                    else:
                        # Save to temp dir
                        save_path = os.path.join(asset_dir, f"{tf}.parquet")
                        df_filtered.to_parquet(save_path)
                        print(f"Saved {len(df_filtered)} rows for {tf} to {save_path}")
                except Exception as e:
                    print(f"Error filtering {tf}: {e}")
        
        # Point config to temp dir
        config['data']['data_dirs']['train'] = temp_dir
        # Also update chunk_size to ensure we load enough data if it's small
        # But usually chunk_size is large enough or handled by the loader
    
    # Force deterministic start
    if 'environment' not in config:
        config['environment'] = {}
    config['environment']['random_start'] = False
    
    # 3. Initialize Env
    try:
        env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="INFO")
    except Exception as e:
        if temp_dir: shutil.rmtree(temp_dir)
        return {"error": f"Failed to init environment: {str(e)}"}

    # 4. Load Model
    try:
        model = PPO.load(checkpoint_path, env=env)
    except Exception as e:
        if temp_dir: shutil.rmtree(temp_dir)
        return {"error": f"Failed to load model from {checkpoint_path}: {str(e)}"}

    # 5. Run Episode
    obs, _ = env.reset()
    done = False
    portfolio_values = []
    timestamps = []
    step_count = 0
    
    # Pre-load data for plotting (from the filtered source if applicable)
    data_frames = {}
    data_source_dir = temp_dir if temp_dir else original_train_dir
    
    for tf in timeframes:
        path = f"{data_source_dir}/{asset}/{tf}.parquet"
        if not os.path.exists(path):
             path_chunk = f"{data_source_dir}/{asset}/{tf}/0.parquet"
             if os.path.exists(path_chunk):
                 path = path_chunk
        
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df = df.sort_index()
                data_frames[tf] = df
            except:
                data_frames[tf] = pd.DataFrame()
        else:
            data_frames[tf] = pd.DataFrame()

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        current_ts = None
        if hasattr(env, 'current_timestamp') and env.current_timestamp is not None:
            current_ts = env.current_timestamp
        
        # Fallback: Try to get from current_data
        if current_ts is None and hasattr(env, 'current_data') and hasattr(env, 'step_in_chunk'):
            try:
                # Try 5m timeframe for asset
                if asset in env.current_data and '5m' in env.current_data[asset]:
                    df = env.current_data[asset]['5m']
                    if not df.empty and env.step_in_chunk < len(df):
                        current_ts = df.index[env.step_in_chunk]
            except:
                pass
        
        if current_ts is None:
             current_ts = pd.Timestamp.now()
        
        # Strict Output Filtering
        if start_date and end_date:
            ts_str = str(current_ts)
            if ts_str >= start_date and ts_str <= end_date:
                portfolio_values.append(env.portfolio_manager.equity)
                timestamps.append(current_ts)
        else:
            portfolio_values.append(env.portfolio_manager.equity)
            timestamps.append(current_ts)
    
    # Cleanup temp dir
    if temp_dir:
        shutil.rmtree(temp_dir)

    # 6. Extract Trades
    trades = []
    
    # Helper to find data at timestamp
    def get_context_at_time(ts, data_frames):
        """Safely extract context (indicators) at a given timestamp"""
        context = {}
        if ts is None:
            return context
        
        try:
            ts_dt = pd.to_datetime(ts)
        except:
            return context
        
        # Try 5m first as it's most granular
        df = data_frames.get('5m')
        if df is not None and not df.empty:
            try:
                # Use asof to find nearest prior timestamp
                idx = df.index.asof(ts_dt)
                if idx is not None and idx in df.index:
                    row = df.loc[idx]
                    context['rsi'] = float(row.get('rsi_14', np.nan)) if pd.notna(row.get('rsi_14')) else None
                    context['macd'] = float(row.get('macd_12_26_9', np.nan)) if pd.notna(row.get('macd_12_26_9')) else None
                    bb_upper = row.get('bb_upper')
                    bb_lower = row.get('bb_lower')
                    close = row.get('close')
                    if bb_upper and bb_lower and close and pd.notna(bb_upper) and pd.notna(bb_lower) and pd.notna(close):
                        context['bb_width'] = float((bb_upper - bb_lower) / close)
                    context['close'] = float(close) if pd.notna(close) else None
            except Exception as e:
                pass
        return context

    if hasattr(env.portfolio_manager, 'trade_log') and env.portfolio_manager.trade_log:
        for t in env.portfolio_manager.trade_log:
            if t.get('asset') != asset:
                continue
            
            # 1. Extract CLOSE event details
            ts_close = t.get('timestamp') or t.get('closed_at')
            price_close = t.get('price') or t.get('exit_price')
            size = t.get('size')
            pnl = t.get('pnl', 0)
            
            # 2. Extract OPEN event details
            ts_open = t.get('opened_at')
            price_open = t.get('entry_price')
            
            # Validate OPEN
            if ts_open and price_open and size:
                include_open = True
                if start_date and end_date:
                    ts_str = str(ts_open)
                    if not (ts_str >= start_date and ts_str <= end_date):
                        include_open = False
                
                if include_open:
                    ctx_open = get_context_at_time(ts_open, data_frames)
                    trades.append({
                        "time": ts_open,
                        "type": "OPEN",
                        "price": float(price_open),
                        "size": float(size),
                        "pnl": 0.0,
                        "sl": float(t.get('sl')) if t.get('sl') else None,
                        "tp": float(t.get('tp')) if t.get('tp') else None,
                        "timeframe": t.get('timeframe', '5m'),
                        "context": ctx_open
                    })

            # Validate CLOSE
            if ts_close and price_close and size:
                include_close = True
                if start_date and end_date:
                    ts_str = str(ts_close)
                    if not (ts_str >= start_date and ts_str <= end_date):
                        include_close = False
                
                if include_close:
                    ctx_close = get_context_at_time(ts_close, data_frames)
                    trades.append({
                        "time": ts_close,
                        "type": "CLOSE",
                        "price": float(price_close),
                        "size": float(size),
                        "pnl": float(pnl),
                        "sl": float(t.get('sl')) if t.get('sl') else None,
                        "tp": float(t.get('tp')) if t.get('tp') else None,
                        "timeframe": t.get('timeframe', '5m'),
                        "context": ctx_close
                    })
    
    # Fallback: if no trades found, try metrics
    if not trades and hasattr(env.portfolio_manager, 'metrics') and hasattr(env.portfolio_manager.metrics, 'closed_positions'):
        try:
            for p in env.portfolio_manager.metrics.closed_positions:
                if p.get('asset') != asset:
                    continue
                
                ts_entry = p.get('entry_time')
                ts_exit = p.get('exit_time')
                
                # Entry
                if ts_entry:
                    include_entry = True
                    if start_date and end_date:
                        if not (str(ts_entry) >= start_date and str(ts_entry) <= end_date):
                            include_entry = False
                    
                    if include_entry:
                        ctx_entry = get_context_at_time(ts_entry, data_frames)
                        trades.append({
                            "time": ts_entry,
                            "type": "OPEN",
                            "price": float(p.get('entry_price', 0)),
                            "size": float(p.get('size', 0)),
                            "pnl": 0.0,
                            "sl": float(p.get('sl')) if p.get('sl') else None,
                            "tp": float(p.get('tp')) if p.get('tp') else None,
                            "timeframe": p.get('timeframe', '5m'),
                            "context": ctx_entry
                        })
                
                # Exit
                if ts_exit:
                    include_exit = True
                    if start_date and end_date:
                        if not (str(ts_exit) >= start_date and str(ts_exit) <= end_date):
                            include_exit = False
                    
                    if include_exit:
                        ctx_exit = get_context_at_time(ts_exit, data_frames)
                        trades.append({
                            "time": ts_exit,
                            "type": "CLOSE",
                            "price": float(p.get('exit_price', 0)),
                            "size": float(p.get('size', 0)),
                            "pnl": float(p.get('realized_pnl', 0)),
                            "sl": float(p.get('sl')) if p.get('sl') else None,
                            "tp": float(p.get('tp')) if p.get('tp') else None,
                            "timeframe": p.get('timeframe', '5m'),
                            "context": ctx_exit
                        })
        except Exception as e:
            print(f"Error extracting trades from metrics: {e}")

    # 7. Calculate Metrics
    # Sort and Deduplicate by timestamp
    if timestamps and portfolio_values:
        df_results = pd.DataFrame({'ts': timestamps, 'val': portfolio_values})
        df_results['ts'] = pd.to_datetime(df_results['ts'])
        df_results = df_results.sort_values('ts')
        df_results = df_results.drop_duplicates(subset='ts', keep='last')
        
        timestamps = df_results['ts'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        portfolio_values = df_results['val'].tolist()
    
    final_equity = portfolio_values[-1] if portfolio_values else capital
    total_return = (final_equity - capital) / capital * 100 if capital > 0 else 0
    
    max_dd = 0.0
    drawdowns = []
    if portfolio_values:
        peak = portfolio_values[0]
        for val in portfolio_values:
            peak = max(peak, val)
            dd = (val - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(dd)
        max_dd = min(drawdowns) if drawdowns else 0.0

    # Calculate additional metrics (only CLOSE trades count for PnL)
    close_trades = [t for t in trades if t.get('type') == 'CLOSE']
    winning_trades = [t for t in close_trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in close_trades if t.get('pnl', 0) < 0]
    
    total_close_trades = len(close_trades)
    win_rate = (len(winning_trades) / total_close_trades * 100
                if total_close_trades > 0 else 0)
    
    gross_profit = sum([t.get('pnl', 0) for t in winning_trades])
    gross_loss = abs(sum([t.get('pnl', 0) for t in losing_trades]))
    profit_factor = (gross_profit / gross_loss if gross_loss > 0
                     else float('inf'))

    metrics = {
        "initial_capital": capital,
        "final_equity": final_equity,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "total_trades": total_close_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "steps": step_count
    }

    return {
        "trades": trades,
        "portfolio_values": portfolio_values,
        "drawdowns": drawdowns,
        "timestamps": timestamps,
        "data": data_frames,
        "metrics": metrics
    }
