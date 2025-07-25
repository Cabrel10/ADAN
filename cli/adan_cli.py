import sys
import argparse
import yaml
from pathlib import Path
import subprocess
import socket
import json
import time
import random

def list_pairs():
    print("Listing supported pairs...")
    # Placeholder for actual implementation

def run_backtest():
    print("Running backtest via scripts/endurance_test.py...")
    script_path = Path(__file__).parent.parent / 'scripts' / 'endurance_test.py'
    try:
        subprocess.run([str(script_path)], check=True)
        print("Backtest completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
    except FileNotFoundError:
        print(f"Error: Backtest script not found at {script_path}")

def plot_live(args):
    print("Starting live simulation stream and sending metrics to UI...")
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"Connected to UI at {HOST}:{PORT}")
            for i in range(10):
                dummy_data = {
                    "timestamp": time.time(),
                    "portfolio_value": round(random.uniform(10000, 15000), 2),
                    "drawdown": round(random.uniform(0.01, 0.10), 4),
                    "risk_mode": random.choice(["Low", "Medium", "High"]),
                    "log_message": f"Live update {i+1}"
                }
                message = json.dumps(dummy_data) + '\n' # Add newline as delimiter
                s.sendall(message.encode('utf-8'))
                print(f"Sent: {message.strip()}")
                time.sleep(1) # Simulate real-time updates
    except ConnectionRefusedError:
        print(f"Error: Connection to UI refused. Is the UI listening on {HOST}:{PORT}?")
    except Exception as e:
        print(f"Error in plot_live: {e}")



def process_data(args):
    print("Processing raw data into datasets via scripts/process_data.py...")
    python_executable = sys.executable
    script_path = Path(__file__).parent.parent / 'scripts' / 'process_data.py'
    command = [python_executable, str(script_path)]
    if args.timeframe:
        command.extend(["--timeframe", args.timeframe])
    try:
        subprocess.run(command, check=True)
        print("Data processing completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error processing data: {e}")
    except FileNotFoundError:
        print(f"Error: Process data script not found at {script_path}")

def merge_data(args):
    print("Merging processed data via scripts/merge_processed_data.py...")
    python_executable = sys.executable
    script_path = Path(__file__).parent.parent / 'scripts' / 'merge_processed_data.py'
    try:
        subprocess.run([python_executable, str(script_path)], check=True)
        print("Data merging completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error merging data: {e}")
    except FileNotFoundError:
        print(f"Error: Merge data script not found at {script_path}")

def download_data(symbol, timeframe, exchange):
    print(f"Downloading historical data for {symbol} ({timeframe}) from {exchange}...")
    python_executable = sys.executable
    script_path = Path(__file__).parent.parent / 'scripts' / 'download_historical_data.py'
    try:
        subprocess.run([python_executable, str(script_path), "--symbol", symbol, "--timeframe", timeframe, "--exchange", exchange], check=True)
        print("Data download completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
    except FileNotFoundError:
        print(f"Error: Download script not found at {script_path}")

def update_config(config_type, section, key, value):
    print(f"Updating {config_type} config: section={section}, key={key}, value={value}")
    
    config_dir = Path(__file__).parent.parent / 'config'
    
    if config_type == 'dbe':
        config_path = config_dir / 'dbe_config.yaml'
    elif config_type == 'environment':
        config_path = config_dir / 'environment_config.yaml'
    elif config_type == 'main':
        config_path = config_dir / 'main_config.yaml'
    else:
        print(f"Error: Unknown config type '{config_type}'. Supported types are 'dbe', 'environment', 'main'.")
        return

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Navigate to the section and update the key
        current_section = config
        for part in section.split('.'):
            current_section = current_section.setdefault(part, {})
        
        # Attempt to convert value to appropriate type
        try:
            if isinstance(current_section.get(key), int):
                value = int(value)
            elif isinstance(current_section.get(key), float):
                value = float(value)
            elif isinstance(current_section.get(key), bool):
                value = value.lower() == 'true'
        except ValueError:
            pass # Keep as string if conversion fails

        current_section[key] = value

        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        print(f"Successfully updated {config_type} config at {config_path}")

    except Exception as e:
        print(f"Error updating config file {config_path}: {e}")

def export_report():
    print("Generating and opening report (JSON/HTML)...")
    # Placeholder for actual implementation

def main():
    parser = argparse.ArgumentParser(description="ADAN Trading Bot CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list-pairs command
    list_pairs_parser = subparsers.add_parser("list-pairs", help="List supported trading pairs")
    list_pairs_parser.set_defaults(func=list_pairs)

    # run-backtest command
    run_backtest_parser = subparsers.add_parser("run-backtest", help="Run a backtest via scripts/endurance_test.py")
    run_backtest_parser.set_defaults(func=run_backtest)

    # plot-live command
    plot_live_parser = subparsers.add_parser("plot-live", help="Start a live simulation stream and send metrics to UI")
    plot_live_parser.set_defaults(func=plot_live)

    # update-config command
    update_config_parser = subparsers.add_parser("update-config", help="Modify configuration YAML files")
    update_config_parser.add_argument("--type", required=True, choices=['dbe', 'environment', 'main', 'data', 'agent'], help="Type of config file (dbe, environment, main, data, agent)")
    update_config_parser.add_argument("--section", required=True, help="Configuration section (e.g., risk_parameters, reward, learning)")
    update_config_parser.add_argument("--key", required=True, help="New value for the configuration key")
    update_config_parser.add_argument("--value", required=True, help="New value for the configuration key")
    update_config_parser.set_defaults(func=lambda args: update_config(args.type, args.section, args.key, args.value))

    # export-report command
    export_report_parser = subparsers.add_parser("export-report", help="Generate and open analysis report (JSON/HTML)")
    export_report_parser.set_defaults(func=export_report)

    # download-data command
    download_data_parser = subparsers.add_parser("download-data", help="Download historical data")
    download_data_parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTC/USDT)")
    download_data_parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., 1h, 4h, 1d)")
    download_data_parser.add_argument("--exchange", required=True, help="Exchange (e.g., binance, bitget)")
    download_data_parser.set_defaults(func=lambda args: download_data(args.symbol, args.timeframe, args.exchange))

    # process-data command
    process_data_parser = subparsers.add_parser("process-data", help="Process raw data into datasets")
    process_data_parser.add_argument("--timeframe", help="Timeframe to process (e.g., 1h, 4h, 5m)")
    process_data_parser.set_defaults(func=process_data)

    # merge-data command
    merge_data_parser = subparsers.add_parser("merge-data", help="Merge processed data")
    merge_data_parser.set_defaults(func=merge_data)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args) # Pass the args object to the lambda for all commands
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
