#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from datetime import timedelta
from collections import deque

def readable_eta(seconds: float) -> str:
    if seconds is None or seconds <= 0 or seconds == float('inf'):
        return "--:--:--"
    return str(timedelta(seconds=int(seconds)))

def format_bar(pct: float, width: int = 40) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100.0)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def parse_timesteps(line: str):
    # Try multiple common patterns
    patterns = [
        r"num_timesteps\s*[:=]\s*(\d+)",
        r"timesteps\s*[:=]\s*(\d+)",
        r"steps\s*[:=]\s*(\d+)",
        r"total timesteps\s*[:=]\s*(\d+)",
        r"\b(\d+)\s*steps\b",
        r"Saving model .*_(\d+)\.zip",
    ]
    for pat in patterns:
        m = re.search(pat, line, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def tail_f(path: str):
    with open(path, 'r', errors='ignore') as f:
        # Seek to end initially
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line

def main():
    parser = argparse.ArgumentParser(description="Monitor ADAN training progress with ETA")
    parser.add_argument('--log-file', required=True, help='Path to training log file')
    parser.add_argument('--total-timesteps', type=int, default=500000, help='Expected total timesteps')
    parser.add_argument('--refresh', type=float, default=1.0, help='Refresh interval seconds')
    parser.add_argument('--quiet', action='store_true', help='Less verbose output')
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    last_ts = 0
    start_time = time.time()
    last_update = start_time
    # Keep a small window of (timestamp, timesteps) to smooth ETA
    window = deque(maxlen=30)

    print("Monitoring log:", args.log_file)
    print("Target timesteps:", args.total_timesteps)

    try:
        for line in tail_f(args.log_file):
            now = time.time()
            ts = parse_timesteps(line)
            if ts is None:
                # Opportunistically catch common SB3 progress patterns
                if 'Evaluating' in line or 'Saving' in line or 'learning rate' in line:
                    pass
            else:
                if ts >= last_ts:
                    last_ts = ts
                    window.append((now, ts))

            # Refresh output at interval
            if now - last_update >= args.refresh:
                last_update = now
                # Compute rate
                eta = None
                rate = None
                if len(window) >= 2:
                    (t0, s0), (t1, s1) = window[0], window[-1]
                    dt = max(1e-6, (t1 - t0))
                    ds = max(0, s1 - s0)
                    rate = ds / dt  # steps per second
                    if rate and args.total_timesteps > 0:
                        remaining = max(0, args.total_timesteps - last_ts)
                        eta = remaining / rate if rate > 0 else None
                pct = (last_ts / args.total_timesteps * 100.0) if args.total_timesteps else 0.0
                bar = format_bar(pct)
                rate_str = f"{rate:,.0f} stp/s" if rate else "-- stp/s"
                eta_str = readable_eta(eta)
                msg = f"{bar} {pct:6.2f}%  steps={last_ts:,}  rate={rate_str}  ETA={eta_str}"
                print("\r" + msg, end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
