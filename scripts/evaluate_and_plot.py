# scripts/evaluate_and_plot.py
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# ------------------------------
# CONFIGURATION
# ------------------------------
CHECKPOINT_PATH = "checkpoints_final/adan_model_checkpoint_640000_steps.zip"
DATA_DIR_BASE = "data/processed/indicators/train"
TIMEFRAMES = ["5m", "1h", "4h"]
ASSET = "XRPUSDT"
INITIAL_CAPITAL = 20.0

# ------------------------------
# Chargement des données (pour le plot)
# ------------------------------
def load_data(tf):
    # Flat structure: data/processed/indicators/train/XRPUSDT/5m.parquet
    path = f"{DATA_DIR_BASE}/{ASSET}/{tf}.parquet"
    if not os.path.exists(path):
        # Try with 0.parquet if flat structure fails
        path_chunk = f"{DATA_DIR_BASE}/{ASSET}/{tf}/0.parquet"
        if os.path.exists(path_chunk):
            path = path_chunk
        else:
            print(f"File not found: {path} or {path_chunk}")
            return pd.DataFrame()
    
    try:
        df = pd.read_parquet(path)
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

data = {tf: load_data(tf) for tf in TIMEFRAMES}

# ------------------------------
# Chargement du modèle + exécution
# ------------------------------
from stable_baselines3 import PPO
from adan_trading_bot.environment.multi_asset_chunked_env import MultiAssetChunkedEnv
from adan_trading_bot.common.config_loader import ConfigLoader

print("Loading config...")
config_loader = ConfigLoader()
config = config_loader.load_config("config/config.yaml")

# OVERRIDE CONFIG FOR EVALUATION
if 'environment' in config:
    config['environment']['assets'] = [ASSET]
else:
    # Fallback if structure is different
    config['assets'] = [ASSET]

config['initial_capital'] = INITIAL_CAPITAL

# OVERRIDE WORKER CONFIG (Critical because worker config takes precedence)
if 'workers' in config:
    # Override for all workers just in case
    for w_id in config['workers']:
        if isinstance(config['workers'][w_id], dict):
            config['workers'][w_id]['assets'] = [ASSET]

# Ensure data dirs are correct relative to where we run the script
if 'data' in config:
    config['data']['data_dirs']['train'] = "data/processed/indicators/train"
    config['data']['data_dirs']['test'] = "data/processed/indicators/test"

print(f"Initializing environment for {ASSET} with ${INITIAL_CAPITAL}...")
env = MultiAssetChunkedEnv(config=config, worker_id=0, log_level="INFO")

print(f"Loading model from {CHECKPOINT_PATH}...")
model = PPO.load(CHECKPOINT_PATH, env=env)

print("Starting evaluation...")
obs, _ = env.reset()
done = False

portfolio_values = []
timestamps = []

# Run for a limited number of steps or until done
max_steps = 2000 # Limit to avoid infinite loops if any
step_count = 0

while not done and step_count < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    step_count += 1

    portfolio_values.append(env.portfolio_manager.equity)
    # Use environment timestamp if available, else estimate
    current_ts = env.current_timestamp if hasattr(env, 'current_timestamp') else pd.Timestamp.now()
    timestamps.append(current_ts)

print(f"Evaluation finished. Steps: {step_count}")

# Extract trades from PortfolioManager trade_log
trades = []
if hasattr(env.portfolio_manager, 'trade_log'):
    # trade_log contains dicts with trade info
    for t in env.portfolio_manager.trade_log:
        # Check if trade is for our asset (should be)
        if t.get('asset') == ASSET:
            trades.append({
                "time": t.get('timestamp') or t.get('exit_time') or t.get('entry_time'), # Use whatever timestamp is available
                "type": t.get('type', 'unknown'), # 'buy' or 'sell' or 'close_long' etc.
                "price": t.get('price') or t.get('exit_price') or t.get('entry_price'),
                "size": t.get('size'),
                "pnl": t.get('pnl', 0),
                "sl": t.get('sl'),
                "tp": t.get('tp'),
                "timeframe": t.get('timeframe', '5m')
            })
    
    # Also check closed_positions in metrics if trade_log is empty or different format
    if not trades and hasattr(env.portfolio_manager, 'metrics'):
        for p in env.portfolio_manager.metrics.closed_positions:
             if p.get('asset') == ASSET:
                # Add entry
                trades.append({
                    "time": p.get('entry_time'),
                    "type": "buy", # Assuming long only for now
                    "price": p.get('entry_price'),
                    "size": p.get('size'),
                    "pnl": 0,
                    "timeframe": p.get('timeframe')
                })
                # Add exit
                trades.append({
                    "time": p.get('exit_time'),
                    "type": "sell",
                    "price": p.get('exit_price'),
                    "size": p.get('size'),
                    "pnl": p.get('realized_pnl'),
                    "timeframe": p.get('timeframe')
                })

print(f"Extracted {len(trades)} trades/actions.")

# ------------------------------
# Création du graphique Plotly
# ------------------------------
print("Generating plot...")
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(f"Prix & Trades (5m) - {ASSET}", "Capital Curve", "Drawdown", "RSI"),
    row_heights=[0.5, 0.2, 0.15, 0.15]
)

# 1. Bougies 5m + trades
df5 = data["5m"]
# Filter df5 to match the simulation period if possible, or just plot all
# To make it readable, let's try to align with timestamps
if timestamps:
    start_time = min(timestamps)
    end_time = max(timestamps)
    # Add some buffer
    mask = (df5.index >= start_time) & (df5.index <= end_time)
    df5_plot = df5.loc[mask]
    if df5_plot.empty:
        df5_plot = df5.iloc[:step_count] # Fallback
else:
    df5_plot = df5.iloc[:1000]

fig.add_trace(go.Candlestick(
    x=df5_plot.index,
    open=df5_plot['open'],
    high=df5_plot['high'],
    low=df5_plot['low'],
    close=df5_plot['close'],
    name=f"{ASSET} 5m"
), row=1, col=1)

# Points d'entrée / sortie
buys = [t for t in trades if t["type"] == "buy"] # Filter by timeframe if needed
sells = [t for t in trades if t["type"] == "sell"]

fig.add_trace(go.Scatter(
    x=[t["time"] for t in buys],
    y=[t["price"] for t in buys],
    mode="markers",
    name="BUY",
    marker=dict(symbol="triangle-up", size=12, color="lime", line=dict(width=2, color='darkgreen'))
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[t["time"] for t in sells],
    y=[t["price"] for t in sells],
    mode="markers",
    name="SELL",
    marker=dict(symbol="triangle-down", size=12, color="red", line=dict(width=2, color='darkred'))
), row=1, col=1)

# 2. Capital curve
fig.add_trace(go.Scatter(
    x=timestamps,
    y=portfolio_values,
    name="Équité",
    line=dict(color="gold", width=2)
), row=2, col=1)

# 3. Drawdown
if portfolio_values:
    max_p = portfolio_values[0]
    drawdowns = []
    for val in portfolio_values:
        max_p = max(max_p, val)
        dd = (val - max_p) / max_p * 100
        drawdowns.append(dd)
else:
    drawdowns = []

fig.add_trace(go.Scatter(
    x=timestamps,
    y=drawdowns,
    name="Drawdown %",
    fill='tozeroy',
    fillcolor="rgba(255,0,0,0.3)",
    line=dict(color="red")
), row=3, col=1)

# 4. RSI (exemple indicateur)
if 'rsi_14' in df5_plot.columns:
    fig.add_trace(go.Scatter(x=df5_plot.index, y=df5_plot['rsi_14'], name="RSI 14", line=dict(color="purple")), row=4, col=1)
    # Add RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=4, col=1)

# Mise en forme
final_equity = portfolio_values[-1] if portfolio_values else INITIAL_CAPITAL
max_dd = min(drawdowns) if drawdowns else 0.0

fig.update_layout(
    title=f"ADAN Evaluation - {ASSET} - Start: ${INITIAL_CAPITAL} -> End: ${final_equity:.2f} ({len(trades)} trades, Max DD: {max_dd:.2f}%)",
    height=1200,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

output_file = "ADAN_evaluation_XRP.html"
fig.write_html(output_file)
print(f"✅ Graphique complet généré : {output_file}")
print(f"   → {len(trades)} trades | Capital final: {final_equity:.2f} USDT | Max DD: {max_dd:.2f}%")
