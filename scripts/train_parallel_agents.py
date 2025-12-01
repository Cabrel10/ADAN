import argparse
import os
import copy
import time
from typing import Optional
import json
from datetime import datetime
import logging
import signal

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from adan_trading_bot.common.config_loader import ConfigLoader
from adan_trading_bot.common.custom_logger import setup_logging
from adan_trading_bot.data_processing.data_loader import ChunkedDataLoader
from adan_trading_bot.environment.realistic_trading_env import (
    RealisticTradingEnv
)
from adan_trading_bot.model.model_ensemble import ModelEnsemble
from adan_trading_bot.utils.seed_manager import SeedManager

def linear_schedule(start_val, end_val, progress):
    return start_val + (end_val - start_val) * progress

def get_adaptive_risk(step: int, total_steps: int,
                      start_cfg: dict, target_cfg: dict,
                      current_drawdown: float = 0.0) -> dict:
    progress = min(1.0, max(0.0, step / max(1, total_steps)))
    pos_size = linear_schedule(start_cfg['position_size_pct'], target_cfg['position_size_pct'], progress)
    sl = linear_schedule(start_cfg['stop_loss_pct'], target_cfg['stop_loss_pct'], progress)
    tp = linear_schedule(start_cfg['take_profit_pct'], target_cfg['take_profit_pct'], progress)

    if current_drawdown >= 0.25:
        safety_mult = 0.4
    elif current_drawdown >= 0.15:
        safety_mult = 0.65
    else:
        safety_mult = 1.0

    adapted = {
        'max_position_size_pct': pos_size * safety_mult,
        'stop_loss_pct': max(0.01, sl),
        'take_profit_pct': tp
    }
    return adapted

class AdaptiveRiskCallback(BaseCallback):
    def __init__(self, total_fine_tune_steps: int, start_cfg: dict, target_cfg: dict, verbose=0):
        super(AdaptiveRiskCallback, self).__init__(verbose)
        self.total_fine_tune_steps = total_fine_tune_steps
        self.start_cfg = start_cfg
        self.target_cfg = target_cfg
        self.risk_log_samples = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        for i in range(len(infos)):
            info = infos[i]
            current_drawdown = info.get("portfolio", {}).get("max_dd", 0.0)

            risk_params = get_adaptive_risk(
                step=self.num_timesteps,
                total_steps=self.total_fine_tune_steps,
                start_cfg=self.start_cfg,
                target_cfg=self.target_cfg,
                current_drawdown=current_drawdown
            )
            self.training_env.env_method('set_global_risk', indices=[i], **risk_params)

            # Log a sample of the risk params
            if self.num_timesteps % 1000 == 0:
                log_sample = f"Step {self.num_timesteps}: {risk_params}"
                if len(self.risk_log_samples) < 10:
                    self.risk_log_samples.append(log_sample)

        return True


class TimeoutHandler:
    def __init__(self, seconds, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)


class CapitalTierTracker:
    """Tracks capital tier progression for each worker."""

    TIERS = {
        "Micro": {"min": 0, "max": 100},
        "Small": {"min": 100, "max": 1000},
        "Medium": {"min": 1000, "max": 10000},
        "High": {"min": 10000, "max": 100000},
        "Enterprise": {"min": 100000, "max": float("inf")},
    }

    def __init__(self, initial_balance=20):
        self.initial_balance = initial_balance
        self.current_tier = "Micro"
        self.tier_history = [("Micro", 0, initial_balance)]
        self.progression_log = []

    def get_tier_from_balance(self, balance):
        """Determine tier based on current balance."""
        for tier_name, limits in self.TIERS.items():
            if limits["min"] <= balance < limits["max"]:
                return tier_name
        return "Enterprise"  # Fallback for very high balances

    def update(self, step, balance, pnl=0.0):
        """Update tier tracking."""
        new_tier = self.get_tier_from_balance(balance)

        if new_tier != self.current_tier:
            # Tier upgrade/downgrade detected
            self.tier_history.append((new_tier, step, balance))
            self.progression_log.append(
                {
                    "step": step,
                    "from_tier": self.current_tier,
                    "to_tier": new_tier,
                    "balance": balance,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.current_tier = new_tier

    def get_progression_summary(self):
        """Get summary of tier progression."""
        return {
            "current_tier": self.current_tier,
            "tier_history": self.tier_history,
            "total_progressions": len(self.progression_log),
            "progression_log": self.progression_log,
            "reached_enterprise": self.current_tier == "Enterprise",
        }


class MetricsMonitor(BaseCallback):
    """
    Enhanced callback to monitor each worker's performance and capital tier progression.
    Generates real-time portfolio curves and tracks tier advancement.
    """

    def __init__(self, config, num_workers=4, log_interval=1000):
        super().__init__()
        self.config = config
        self.num_workers = num_workers
        self.log_interval = log_interval
        self.worker_metrics = {}
        self.portfolio_curves = {i: [] for i in range(num_workers)}
        self.tier_trackers = {
            i: CapitalTierTracker(config["portfolio"]["initial_balance"])
            for i in range(num_workers)
        }
        self.step_count = 0
        self.start_time = time.time()

        # Minimal daily tracker implementation for per-day stats
        class DailyTracker:
            def __init__(self):
                self.current_day = None
                self.daily = {}

            def update(self, day_id, balance, trade_info, return_value):
                d = self.daily.setdefault(day_id, {
                    "balances": [],
                    "returns": [],
                    "trades_closed": 0,
                    "wins": 0,
                    "losses": 0,
                })
                d["balances"].append(float(balance))
                d["returns"].append(float(return_value))
                if trade_info:
                    if trade_info.get("trade_closed"):
                        d["trades_closed"] += 1
                        pnl = float(trade_info.get("trade_pnl", 0.0))
                        if pnl > 0:
                            d["wins"] += 1
                        elif pnl < 0:
                            d["losses"] += 1

            def get_current_day_summary(self):
                if not self.daily:
                    return {}
                day_id = sorted(self.daily.keys())[-1]
                d = self.daily[day_id]
                avg_ret = np.mean(d["returns"]) if d["returns"] else 0.0
                win_rate = (d["wins"] / max(1, d["trades_closed"])) * 100.0
                gross_profit = sum(r for r in d["returns"] if r > 0)
                gross_loss = abs(sum(r for r in d["returns"] if r < 0))
                pf = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
                return {
                    "daily_pnl": (d["balances"][-1] - d["balances"][0]) if d["balances"] else 0.0,
                    "daily_return_pct": avg_ret * 100.0,
                    "trades_closed": d["trades_closed"],
                    "win_rate": win_rate,
                    "profit_factor": pf,
                }

            def finalize_current_day(self):
                # No-op for minimal tracker
                pass

            def get_average_daily_performance(self):
                if not self.daily:
                    return {}
                daily_returns = []
                trades = []
                win_rates = []
                profit_factors = []
                sharpes = []
                for day_id, d in self.daily.items():
                    avg_ret = np.mean(d["returns"]) if d["returns"] else 0.0
                    daily_returns.append(avg_ret * 100.0)
                    trades.append(d["trades_closed"])
                    wr = (d["wins"] / max(1, d["trades_closed"])) * 100.0
                    win_rates.append(wr)
                    gp = sum(r for r in d["returns"] if r > 0)
                    gl = abs(sum(r for r in d["returns"] if r < 0))
                    profit_factors.append((gp / gl) if gl > 0 else 0.0)
                    if len(d["returns"]) > 1:
                        vol = np.std(d["returns"]) * np.sqrt(252)
                        mu = np.mean(d["returns"]) * 252
                        sharpes.append((mu / vol) if vol > 0 else 0.0)
                    else:
                        sharpes.append(0.0)
                return {
                    "avg_daily_return_pct": float(np.mean(daily_returns)) if daily_returns else 0.0,
                    "avg_trades_per_day": float(np.mean(trades)) if trades else 0.0,
                    "avg_win_rate": float(np.mean(win_rates)) if win_rates else 0.0,
                    "avg_profit_factor": float(np.mean(profit_factors)) if profit_factors else 0.0,
                    "avg_daily_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
                }

        # Create daily trackers per worker
        self.daily_trackers = {i: DailyTracker() for i in range(num_workers)}

        # Initialize worker-specific tracking
        for i in range(num_workers):
            self.worker_metrics[i] = {
                "total_steps": 0,
                "total_rewards": [],
                "portfolio_values": [],
                "realized_pnls": [],
                "sharpe_ratios": [],
                "drawdowns": [],
                "trade_counts": [],
                "win_rates": [],
                "daily_performance": [],
                "tier_progressions": [],
            }

    def _on_step(self) -> bool:
        """Called at each step of training."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            self._collect_worker_metrics()

        return True

    def _collect_worker_metrics(self):
        """Collect metrics from all workers with daily tracking."""
        try:
            # Get portfolio managers and environments
            portfolio_managers = self.training_env.get_attr("portfolio_manager")
            environments = self.training_env.get_attr("data")

            for worker_id, pm in enumerate(portfolio_managers):
                if pm is None:
                    continue

                # Get current metrics
                metrics = pm.metrics.get_metrics_summary()
                # Use PortfolioManager API to fetch current portfolio value
                try:
                    current_balance = float(pm.get_portfolio_value())
                except Exception:
                    current_balance = float(self.config["portfolio"]["initial_balance"])
                current_pnl = (
                    metrics.get("total_return", 0.0)
                    * self.config["portfolio"]["initial_balance"]
                    / 100.0
                )

                # Get current day from environment data
                current_day = 0
                trade_info = {}
                if worker_id < len(environments) and environments[worker_id]:
                    env_data = environments[worker_id]
                    if "TIMESTAMP" in env_data and len(env_data["TIMESTAMP"]) > 0:
                        # Calculate day from timestamp (assuming milliseconds)
                        timestamp = env_data["TIMESTAMP"].iloc[-1] if hasattr(env_data["TIMESTAMP"], 'iloc') else env_data["TIMESTAMP"][-1]
                        current_day = int(timestamp // (24 * 60 * 60 * 1000))

                    # Check for trade information in recent metrics
                    recent_trades = metrics.get("recent_trades", [])
                    if recent_trades:
                        last_trade = recent_trades[-1] if isinstance(recent_trades, list) else recent_trades
                        if isinstance(last_trade, dict):
                            trade_info = {
                                "trade_closed": last_trade.get("closed", False),
                                "trade_opened": last_trade.get("opened", False),
                                "trade_pnl": last_trade.get("pnl", 0.0)
                            }

                # Update tier tracker
                self.tier_trackers[worker_id].update(
                    self.step_count, current_balance, current_pnl
                )

                # Update daily tracker
                return_value = metrics.get("last_return", 0.0)
                self.daily_trackers[worker_id].update(
                    current_day, current_balance, trade_info, return_value
                )

                # Get daily performance summary
                daily_summary = self.daily_trackers[worker_id].get_current_day_summary()

                # Store worker metrics
                worker_data = {
                    "step": self.step_count,
                    "balance": current_balance,
                    "pnl": current_pnl,
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "drawdown": metrics.get("max_drawdown", 0.0),
                    "trade_count": metrics.get("executed_trades_closed", 0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "tier": self.tier_trackers[worker_id].current_tier,
                    "timestamp": time.time() - self.start_time,
                    "current_day": current_day,
                    "daily_performance": daily_summary,
                }

                self.portfolio_curves[worker_id].append(worker_data)

                # Update aggregated metrics
                # Update aggregated metrics
                self.worker_metrics[worker_id]["total_steps"] = self.step_count
                self.worker_metrics[worker_id]["portfolio_values"].append(
                    current_balance
                )
                self.worker_metrics[worker_id]["realized_pnls"].append(current_pnl)
                self.worker_metrics[worker_id]["sharpe_ratios"].append(
                    metrics.get("sharpe_ratio", 0.0)
                )
                self.worker_metrics[worker_id]["drawdowns"].append(
                    metrics.get("max_drawdown", 0.0)
                )
                self.worker_metrics[worker_id]["trade_counts"].append(
                    metrics.get("executed_trades_closed", 0)
                )
                self.worker_metrics[worker_id]["win_rates"].append(
                    metrics.get("win_rate", 0.0)
                )
                self.worker_metrics[worker_id]["daily_performance"].append(daily_summary)

                # Log worker progress including daily metrics
                if worker_id == 0 or self.step_count % (self.log_interval * 5) == 0:
                    self.logger.record(f"worker_{worker_id}/balance", current_balance)
                    self.logger.record(f"worker_{worker_id}/pnl", current_pnl)
                    self.logger.record(
                        f"worker_{worker_id}/tier",
                        self.tier_trackers[worker_id].current_tier,
                    )
                    self.logger.record(
                        f"worker_{worker_id}/sharpe", metrics.get("sharpe_ratio", 0.0)
                    )

                    # Log daily metrics
                    if daily_summary:
                        self.logger.record(f"worker_{worker_id}/daily_pnl", daily_summary.get("daily_pnl", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_return_pct", daily_summary.get("daily_return_pct", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_trades", daily_summary.get("trades_closed", 0))
                        self.logger.record(f"worker_{worker_id}/daily_win_rate", daily_summary.get("win_rate", 0.0))
                        self.logger.record(f"worker_{worker_id}/daily_profit_factor", min(daily_summary.get("profit_factor", 0.0), 10.0))  # Cap for logging

        except Exception as e:
            logging.getLogger(__name__).error(f"Error collecting worker metrics: {e}", exc_info=True)
            raise

    def get_final_summary(self):
        """Generate comprehensive training summary with daily metrics."""
        # Finalize daily tracking for all workers
        for worker_id in range(self.num_workers):
            self.daily_trackers[worker_id].finalize_current_day()

        summary = {
            "training_duration": time.time() - self.start_time,
            "total_steps": self.step_count,
            "workers": {},
            "overall_daily_performance": {},
        }

        all_avg_daily_returns = []
        all_avg_trades_per_day = []
        all_avg_win_rates = []
        all_avg_profit_factors = []
        all_avg_sharpe_ratios = []

        for worker_id in range(self.num_workers):
            if not self.worker_metrics[worker_id]["portfolio_values"]:
                continue

            final_balance = self.worker_metrics[worker_id]["portfolio_values"][-1]
            total_return = (
                (final_balance - self.config["portfolio"]["initial_balance"])
                / self.config["portfolio"]["initial_balance"]
                * 100
            )

            # Tier progression summary
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()

            # Daily performance summary
            daily_avg_performance = self.daily_trackers[worker_id].get_average_daily_performance()

            worker_summary = {
                "final_balance": final_balance,
                "total_return_pct": total_return,
                "final_tier": tier_summary["current_tier"],
                "tier_progressions": tier_summary["total_progressions"],
                "final_sharpe": (
                    self.worker_metrics[worker_id]["sharpe_ratios"][-1]
                    if self.worker_metrics[worker_id]["sharpe_ratios"]
                    else 0.0
                ),
                "max_drawdown": (
                    max(self.worker_metrics[worker_id]["drawdowns"])
                    if self.worker_metrics[worker_id]["drawdowns"]
                    else 0.0
                ),
                "total_trades": (
                    self.worker_metrics[worker_id]["trade_counts"][-1]
                    if self.worker_metrics[worker_id]["trade_counts"]
                    else 0
                ),
                "final_win_rate": (
                    self.worker_metrics[worker_id]["win_rates"][-1]
                    if self.worker_metrics[worker_id]["win_rates"]
                    else 0.0
                ),
                "reached_enterprise": tier_summary["reached_enterprise"],
                # Daily performance metrics
                "daily_performance": daily_avg_performance,
            }

            summary["workers"][worker_id] = worker_summary

            # Collect for overall averages
            if daily_avg_performance:
                all_avg_daily_returns.append(daily_avg_performance.get("avg_daily_return_pct", 0))
                all_avg_trades_per_day.append(daily_avg_performance.get("avg_trades_per_day", 0))
                all_avg_win_rates.append(daily_avg_performance.get("avg_win_rate", 0))
                profit_factor = daily_avg_performance.get("avg_profit_factor", 0)
                if profit_factor != float("inf") and profit_factor > 0:
                    all_avg_profit_factors.append(profit_factor)
                all_avg_sharpe_ratios.append(daily_avg_performance.get("avg_daily_sharpe", 0))

        # Calculate overall daily performance across all workers
        if all_avg_daily_returns:
            summary["overall_daily_performance"] = {
                "avg_daily_return_pct": np.mean(all_avg_daily_returns),
                "avg_trades_per_day": np.mean(all_avg_trades_per_day),
                "avg_win_rate": np.mean(all_avg_win_rates),
                "avg_profit_factor": np.mean(all_avg_profit_factors) if all_avg_profit_factors else 0,
                "avg_daily_sharpe": np.mean(all_avg_sharpe_ratios),
                "best_daily_return_pct": max(all_avg_daily_returns),
                "worst_daily_return_pct": min(all_avg_daily_returns),
                "consistency_score": 1.0 - (np.std(all_avg_daily_returns) / max(abs(np.mean(all_avg_daily_returns)), 0.01)),
            }

        return summary

    def generate_portfolio_curves(self, output_dir):
        """Generate portfolio progression curves for each worker."""
        if go is None:
            logging.warning("Plotly not available, skipping curves")
            return

        os.makedirs(output_dir, exist_ok=True)

        for worker_id in range(self.num_workers):
            if not self.portfolio_curves[worker_id]:
                continue

            df = pd.DataFrame(self.portfolio_curves[worker_id])
            worker_name = f"w{worker_id + 1}"

            # Create portfolio progression chart
            fig = go.Figure()

            # Portfolio balance line
            fig.add_trace(
                go.Scatter(
                    x=df["step"],
                    y=df["balance"],
                    mode="lines",
                    name=f"{worker_name} Portfolio Balance",
                    line=dict(color="blue", width=2),
                )
            )

            # Add tier progression markers
            tier_changes = self.tier_trackers[worker_id].progression_log
            if tier_changes:
                tier_steps = [tc["step"] for tc in tier_changes]
                tier_balances = [tc["balance"] for tc in tier_changes]
                tier_labels = [
                    f"{tc['from_tier']} → {tc['to_tier']}" for tc in tier_changes
                ]

                fig.add_trace(
                    go.Scatter(
                        x=tier_steps,
                        y=tier_balances,
                        mode="markers+text",
                        name=f"{worker_name} Tier Upgrades",
                        text=tier_labels,
                        textposition="top center",
                        marker=dict(color="red", size=10, symbol="diamond"),
                    )
                )

            # Add tier zones as background
            tier_colors = {
                "Micro": "lightgray",
                "Small": "lightblue",
                "Medium": "lightgreen",
                "High": "lightyellow",
                "Enterprise": "lightcoral",
            }

            for tier_name, limits in CapitalTierTracker.TIERS.items():
                if limits["max"] != float("inf"):
                    fig.add_hrect(
                        y0=limits["min"],
                        y1=limits["max"],
                        fillcolor=tier_colors.get(tier_name, "lightgray"),
                        opacity=0.2,
                        line_width=0,
                        annotation_text=tier_name,
                        annotation_position="top left",
                    )

            fig.update_layout(
                title=f"Portfolio Progression - {worker_name.upper()} (Capital Tier Advancement)",
                xaxis_title="Training Steps",
                yaxis_title="Portfolio Balance ($)",
                yaxis_type="log",
                showlegend=True,
            )

            # Save chart
            chart_path = os.path.join(
                output_dir, f"portfolio_progression_{worker_name}.html"
            )
            fig.write_html(chart_path)
            print(f"✅ Generated portfolio chart: {chart_path}")

    def get_final_summary(self):
        """Get final training summary with tier progression."""
        summary = {
            "training_duration_minutes": (time.time() - self.start_time) / 60,
            "total_steps": self.step_count,
            "workers": {},
        }

        for worker_id in range(self.num_workers):
            worker_name = f"w{worker_id + 1}"
            tier_summary = self.tier_trackers[worker_id].get_progression_summary()

            if self.portfolio_curves[worker_id]:
                final_data = self.portfolio_curves[worker_id][-1]
                initial_balance = self.config["portfolio"]["initial_balance"]

                summary["workers"][worker_name] = {
                    "initial_balance": initial_balance,
                    "final_balance": final_data["balance"],
                    "total_return_pct": (
                        (final_data["balance"] - initial_balance) / initial_balance
                    )
                    * 100,
                    "final_pnl": final_data["pnl"],
                    "final_sharpe": final_data["sharpe_ratio"],
                    "max_drawdown": max(self.worker_metrics[worker_id]["drawdowns"])
                    if self.worker_metrics[worker_id]["drawdowns"]
                    else 0,
                    "total_trades": final_data["trade_count"],
                    "tier_progression": tier_summary,
                    "reached_enterprise": tier_summary["reached_enterprise"],
                }

        return summary


def train_worker(worker_id: str, worker_idx: int, config: dict, resume: bool, checkpoint_dir: str, final_export_dir: str):
    """
    Independent training function for a single worker process.
    """
    # Re-setup logging for this process
    setup_logging(config)
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.info(f"🚀 STARTING WORKER PROCESS: {worker_id} (PID: {os.getpid()})")
    
    try:
        # Initialize SeedManager for this process with unique seed offset
        seed = config.get("general", {}).get("random_seed", 42) + worker_idx
        SeedManager.initialize(seed)
        logger.info(f"🎲 Initialized SeedManager with seed={seed}")

        # Get worker-specific config
        worker_config = config["workers"][worker_id]
        agent_config = worker_config.get("agent_config", {})
        
        logger.info(f"📋 Worker Config for {worker_id}:")
        logger.info(f"   Learning Rate: {agent_config.get('learning_rate')}")
        logger.info(f"   N Steps: {agent_config.get('n_steps')}")
        
        # Create worker-specific environment
        def make_worker_env(env_idx):
            """Create environment for this worker"""
            def _init():
                wc = copy.deepcopy(worker_config)
                data_loader = ChunkedDataLoader(
                    config=config, worker_config=wc, worker_id=env_idx
                )
                data = data_loader.load_chunk(0)
                
                env_worker_config = copy.deepcopy(wc)
                env_worker_config["worker_id"] = env_idx
                
                env_log_dir = os.path.join(
                    config["paths"]["logs_dir"], f"{worker_id}_env_{env_idx}"
                )
                os.makedirs(env_log_dir, exist_ok=True)
                
                return RealisticTradingEnv(
                    data=data,
                    timeframes=config["data"]["timeframes"],
                    window_sizes=config["environment"]["observation"]["window_sizes"],
                    features_config=config["data"]["features_config"]["timeframes"],
                    max_steps=config["environment"]["max_steps"],
                    initial_balance=config["portfolio"]["initial_balance"],
                    commission=config["environment"]["commission"],
                    reward_scaling=config["environment"]["reward_scaling"],
                    enable_logging=True,
                    log_dir=env_log_dir,
                    worker_config=env_worker_config,
                    config=config,
                    exploration_tutor=config.get("reward_shaping", {}).get("exploration_tutor", {}),
                    # Realistic Constraints
                    live_mode=False,
                    min_hold_steps=6,  # 30m
                    cooldown_steps=3,  # 15m
                    min_notional=10.0,
                    circuit_breaker_pct=0.15
                )
            return _init
        
        # Create DummyVecEnv for this worker
        worker_env = DummyVecEnv([make_worker_env(worker_idx)])
        
        # Wrap with VecNormalize
        gamma = agent_config.get("gamma", config["agent"]["gamma"])
        worker_env = VecNormalize(
            worker_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
            training=True
        )
        logger.info(f"✅ Created DummyVecEnv + VecNormalize for {worker_id}")
        
        # Load existing VecNormalize stats if resuming
        vec_normalize_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        if resume and os.path.exists(vec_normalize_path):
            worker_env = VecNormalize.load(vec_normalize_path, worker_env)
            logger.info(f"✅ Loaded VecNormalize stats from {vec_normalize_path}")

        # Policy kwargs
        policy_kwargs = copy.deepcopy(
            config["agent"]["features_extractor_kwargs"]["policy_kwargs"]
        )
        activation_fn_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU,
        }
        if "activation_fn" in policy_kwargs:
            activation_fn_str = policy_kwargs["activation_fn"]
            act_fn_name = activation_fn_str.split(".")[-1]
            activation_fn = activation_fn_map.get(act_fn_name)
            if activation_fn:
                policy_kwargs["activation_fn"] = activation_fn
            else:
                policy_kwargs["activation_fn"] = nn.ReLU
        
        # Create worker-specific checkpoint directory
        worker_checkpoint_dir = os.path.join(checkpoint_dir, worker_id)
        os.makedirs(worker_checkpoint_dir, exist_ok=True)
        
        # Callbacks
        worker_callbacks = []
        worker_checkpoint_callback = CheckpointCallback(
            save_freq=config["training"]["checkpointing"]["save_freq"],
            save_path=worker_checkpoint_dir,
            name_prefix=f"{worker_id}_model",
        )
        worker_callbacks.append(worker_checkpoint_callback)
        
        worker_metrics_monitor = MetricsMonitor(
            config=config,
            num_workers=1,
            log_interval=max(1000, config["training"]["checkpointing"]["save_freq"] // 10),
        )
        worker_callbacks.append(worker_metrics_monitor)
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Hyperparameters
        learning_rate = agent_config.get("learning_rate", config["agent"]["learning_rate"])
        n_steps = agent_config.get("n_steps", config["agent"]["n_steps"])
        batch_size = agent_config.get("batch_size", config["agent"]["batch_size"])
        n_epochs = agent_config.get("n_epochs", config["agent"]["n_epochs"])
        gae_lambda = agent_config.get("gae_lambda", config["agent"]["gae_lambda"])
        clip_range = agent_config.get("clip_range", config["agent"]["clip_range"])
        ent_coef = agent_config.get("ent_coef", config["agent"]["ent_coef"])
        
        total_timesteps = config["training"]["timesteps_per_instance"]
        
        # Create Model
        worker_model = PPO(
            "MultiInputPolicy",
            worker_env,
            device=device,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=config["agent"]["vf_coef"],
            max_grad_norm=config["agent"]["max_grad_norm"],
            tensorboard_log=os.path.join(config["paths"]["logs_dir"], f"tensorboard_{worker_id}"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
        )
        
        logger.info(f"🚀 Training {worker_id} for {total_timesteps} steps...")
        worker_model.learn(
            total_timesteps=total_timesteps,
            callback=worker_callbacks,
            tb_log_name=f"ppo_{worker_id}"
        )
        
        # Save Final Model
        worker_final_path = os.path.join(final_export_dir, f"{worker_id}_final.zip")
        worker_model.save(worker_final_path)
        logger.info(f"✅ {worker_id} model saved: {worker_final_path}")
        
        # Save VecNormalize stats
        worker_vec_path = os.path.join(final_export_dir, f"{worker_id}_vecnormalize.pkl")
        worker_env.save(worker_vec_path)
        logger.info(f"✅ {worker_id} VecNormalize stats saved: {worker_vec_path}")
        
        # Save main vecnormalize.pkl from the first worker (as baseline)
        if worker_idx == 0:
            main_vec_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
            worker_env.save(main_vec_path)
            logger.info(f"✅ Main VecNormalize stats saved (from {worker_id}): {main_vec_path}")
            
            # Save RNG states
            rng_states = SeedManager.get_rng_states()
            rng_states_path = os.path.join(checkpoint_dir, "rng_states.json")
            with open(rng_states_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_states = {}
                for key, value in rng_states.items():
                    if key == 'numpy_random_state':
                        serializable_states[key] = {
                            'state_type': str(value[0]),
                            'keys': value[1].tolist() if hasattr(value[1], 'tolist') else value[1],
                            'pos': int(value[2]),
                            'has_gauss': int(value[3]),
                            'cached_gaussian': float(value[4])
                        }
                    elif key == 'torch_random_state' or key == 'torch_cuda_random_state':
                        continue
                    else:
                        serializable_states[key] = value
                json.dump(serializable_states, f, indent=2)
            logger.info(f"✅ RNG states saved to {rng_states_path}")

        logger.info(f"🏁 WORKER {worker_id} FINISHED!")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR IN WORKER {worker_id}: {e}", exc_info=True)
        raise

def main(
    config_path: str,
    resume: bool,
    num_envs: int,
    use_subproc: bool,
    progress_bar: bool,
    timeout: Optional[int],
    fine_tune: bool,
    steps: Optional[int],
    log_level: str,
    checkpoint_dir: str = None,
):
    import multiprocessing
    
    logger = logging.getLogger(__name__)
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger("adan_trading_bot").setLevel(numeric_level)
    
    try:
        # Load Config
        config = ConfigLoader.load_config(config_path)
        logger.info("📋 Loaded config.yaml")
        
        # Override steps if provided
        if steps:
            config["training"]["timesteps_per_instance"] = steps
            logger.info(f"Overriding total_timesteps with: {steps}")

        # Checkpoint Dirs
        if checkpoint_dir is None:
            checkpoint_dir = config["paths"]["trained_models_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_export_dir = os.path.join(checkpoint_dir, "final")
        os.makedirs(final_export_dir, exist_ok=True)

        logger.info("="*80)
        logger.info("🔥 MULTI-AGENT PARALLEL TRAINING ACTIVATED")
        logger.info("Training 4 INDEPENDENT PPO models in PARALLEL PROCESSES")
        logger.info("="*80)
        
        worker_ids = ["w1", "w2", "w3", "w4"]
        processes = []
        
        # Launch processes
        for i, worker_id in enumerate(worker_ids):
            p = multiprocessing.Process(
                target=train_worker,
                args=(worker_id, i, config, resume, checkpoint_dir, final_export_dir)
            )
            p.start()
            processes.append(p)
            logger.info(f"🚀 Started process for {worker_id} (PID: {p.pid})")
            
        # Wait for all
        for p in processes:
            p.join()
            
        logger.info("✅ All workers completed.")
        
    except Exception as e:
        logger.error(f"❌ Main process error: {e}", exc_info=True)
        # Terminate children if main fails
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ADAN Agents in Parallel")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of environments (ignored)")
    parser.add_argument("--use-subproc", action="store_true", help="Use SubprocVecEnv (ignored)")
    parser.add_argument("--progress-bar", action="store_true", help="Show progress bar")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune mode")
    parser.add_argument("--steps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint dir")

    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=args.resume,
        num_envs=args.num_envs,
        use_subproc=args.use_subproc,
        progress_bar=args.progress_bar,
        timeout=args.timeout,
        fine_tune=args.fine_tune,
        steps=args.steps,
        log_level=args.log_level,
        checkpoint_dir=args.checkpoint_dir,
    )
