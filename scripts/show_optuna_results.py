#!/usr/bin/python3
"""
Advanced Optuna Results Analyzer - Per Worker Optimization Results

This script analyzes Optuna study results and extracts the top 3 hyperparameter sets
for each worker individually, providing detailed performance analysis.
"""

import optuna
import logging
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WorkerOptimizationAnalyzer:
    """Analyzes Optuna results to find top N hyperparameters per worker."""

    def __init__(self, storage_name: str, study_name: str, top_n: int = 3):
        self.storage_name = storage_name
        self.study_name = study_name
        self.top_n = top_n
        self.study = None
        self.worker_ids = ["w1", "w2", "w3", "w4"]
        self.worker_names = {
            "w1": "Conservative Worker",
            "w2": "Moderate Worker",
            "w3": "Aggressive Worker",
            "w4": "Adaptive Worker",
        }

    def load_study(self) -> bool:
        """Load the Optuna study from database."""
        try:
            self.study = optuna.load_study(
                study_name=self.study_name, storage=self.storage_name
            )
            logger.info(f"✅ Successfully loaded study: '{self.study_name}' from {self.storage_name}")
            return True
        except (KeyError, ValueError) as e:
            logger.error(
                f"❌ Study '{self.study_name}' not found in '{self.storage_name}'"
            )
            logger.info("Please ensure the study name is correct and you have run 'optimize_hyperparams.py' first.")
            return False

    def analyze_trials(self) -> Dict[str, Any]:
        """Analyze all trials and categorize them."""
        if not self.study:
            return {}

        all_trials = self.study.trials
        completed_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in all_trials if t.state == optuna.trial.TrialState.FAIL
        ]

        return {
            "total": len(all_trials),
            "completed": completed_trials,
            "pruned": len(pruned_trials),
            "failed": len(failed_trials),
        }

    def find_top_worker_trials(self, completed_trials: List) -> Dict[str, List[Dict]]:
        """Find the top N best trials for each worker individually."""
        top_workers = {worker_id: [] for worker_id in self.worker_ids}

        for worker_id in self.worker_ids:
            worker_trials = []
            for trial in completed_trials:
                worker_score = trial.user_attrs.get(f"{worker_id}_score")
                if worker_score is not None:
                    worker_trials.append({
                        "trial": trial,
                        "score": worker_score,
                        "trial_number": trial.number,
                    })

            # Sort trials by score in descending order
            sorted_trials = sorted(worker_trials, key=lambda x: x['score'], reverse=True)

            # Get the top N trials
            top_workers[worker_id] = sorted_trials[:self.top_n]

        return top_workers

    def extract_worker_params(self, trial, worker_id: str) -> Dict[str, Any]:
        """Extract hyperparameters for a specific worker from a trial."""
        params = {}
        worker_params_key = f"{worker_id}_params"
        if worker_params_key in trial.user_attrs:
            return trial.user_attrs[worker_params_key]

        prefix = f"{worker_id}_"
        for param_name, value in trial.params.items():
            if param_name.startswith(prefix):
                clean_name = param_name[len(prefix):]
                params[clean_name] = value
        return params

    def get_worker_behavior(self, trial, worker_id: str) -> Dict[str, Any]:
        """Extract behavior analysis for a specific worker."""
        worker_behaviors = trial.user_attrs.get("worker_behaviors", {})
        return worker_behaviors.get(worker_id, {})

    def print_study_summary(self, analysis: Dict[str, Any]):
        """Print overall study summary."""
        print("\n" + "=" * 80)
        print(f"🎯 OPTUNA STUDY ANALYSIS - TOP {self.top_n} HYPERPARAMETERS PER WORKER")
        print("=" * 80)
        print(f"📊 Study: {self.study_name}")
        print(f"💾 Storage: {self.storage_name}")
        print(f"📈 Total trials: {analysis['total']}")
        print(f"   ✅ Completed: {len(analysis['completed'])}")
        print(f"   ✂️  Pruned: {analysis['pruned']}")
        print(f"   ❌ Failed: {analysis['failed']}")

    def print_worker_results(self, top_workers: Dict[str, List[Dict]]):
        """Print detailed results for the top N trials of each worker."""
        for worker_id in self.worker_ids:
            print(f"\n{'=' * 70}")
            print(f"🤖 {self.worker_names[worker_id].upper()} ({worker_id.upper()}) - TOP {self.top_n} RESULTS")
            print("=" * 70)

            top_n_trials = top_workers.get(worker_id)
            if not top_n_trials:
                print("❌ No successful trials found for this worker")
                continue

            for rank, worker_data in enumerate(top_n_trials, 1):
                trial = worker_data["trial"]
                score = worker_data["score"]

                print(f"\n--- RANK #{rank} ---")
                print(f"🏆 Best Trial: #{trial.number}")
                print(f"📊 Individual Score: {score:.4f}")
                print(f"🌐 Global Trial Score: {trial.value:.4f}")

                if "duration_minutes" in trial.user_attrs:
                    duration = trial.user_attrs["duration_minutes"]
                    print(f"⏱️  Duration: {duration:.1f} minutes")

                worker_params = self.extract_worker_params(trial, worker_id)
                if worker_params:
                    print(f"\n🔧 OPTIMAL HYPERPARAMETERS (Rank #{rank}):")
                    for param_name, value in worker_params.items():
                        if isinstance(value, float):
                            print(f"   • {param_name}: {value:.6f}")
                        else:
                            print(f"   • {param_name}: {value}")

                behavior = self.get_worker_behavior(trial, worker_id)
                if behavior:
                    print(f"\n📈 PERFORMANCE METRICS (Rank #{rank}):")
                    metrics = [
                        ("Total Trades", "total_trades", None),
                        ("Win Rate", "win_rate", ".1%"),
                        ("Total PnL", "total_pnl", ".2f"),
                        ("Portfolio Growth", "portfolio_growth", ".2%"),
                        ("Max Drawdown", "max_drawdown", ".2%"),
                        ("Profit Consistency", "profit_consistency", ".2f"),
                    ]
                    for name, key, fmt in metrics:
                        if key in behavior:
                            value = behavior[key]
                            if fmt:
                                print(f"   • {name}: {value:{fmt}}")
                            else:
                                print(f"   • {name}: {value}")

                    if "timeframe_trades" in behavior:
                        tf_trades = behavior["timeframe_trades"]
                        print("   • Timeframe Distribution:")
                        for tf, count in tf_trades.items():
                            print(f"     - {tf}: {count} trades")

    def print_global_best(self, completed_trials: List):
        """Print the best global trial for comparison."""
        if not completed_trials:
            return

        best_global = max(completed_trials, key=lambda t: t.value)
        print(f"\n{'=' * 60}")
        print("🌟 BEST GLOBAL TRIAL (FOR REFERENCE)")
        print("=" * 60)
        print(f"🏆 Trial: #{best_global.number}")
        print(f"📊 Global Score: {best_global.value:.4f}")
        print("\n📊 Individual Worker Scores in Best Global Trial:")
        for worker_id in self.worker_ids:
            score = best_global.user_attrs.get(f"{worker_id}_score", "N/A")
            print(f"   • {worker_id.upper()}: {score:.4f}" if isinstance(score, (int, float)) else f"   • {worker_id.upper()}: {score}")

    def generate_summary_table(self, top_workers: Dict[str, List[Dict]]):
        """Generate a summary table of the best result for each worker."""
        print(f"\n{'=' * 80}")
        print("📋 SUMMARY TABLE - BEST SCORE PER WORKER")
        print("=" * 80)
        print(f"{'Worker':<20} {'Trial#':<8} {'Score':<10} {'Params Preview':<45}")
        print("-" * 80)

        for worker_id in self.worker_ids:
            if worker_id in top_workers and top_workers[worker_id]:
                data = top_workers[worker_id][0]  # Show only the #1 result
                trial = data["trial"]
                score = data["score"]
                params = self.extract_worker_params(trial, worker_id)
                param_preview = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(params.items())[:3]]
                preview_str = ", ".join(param_preview) + (f", +{len(params) - 3} more" if len(params) > 3 else "")
                print(f"{self.worker_names[worker_id]:<20} #{data['trial_number']:<7} {score:<10.4f} {preview_str:<45}")
            else:
                print(f"{self.worker_names[worker_id]:<20} {'N/A':<8} {'N/A':<10} {'No successful trials':<45}")

    def export_results(self, top_workers: Dict[str, List[Dict]], filename: str = "best_worker_params.json"):
        """Export the top N results for each worker to a JSON file."""
        try:
            export_data = {}
            for worker_id, top_trials in top_workers.items():
                export_data[worker_id] = {
                    "worker_name": self.worker_names[worker_id],
                    "top_results": []
                }
                for data in top_trials:
                    trial = data['trial']
                    export_data[worker_id]["top_results"].append({
                        "rank": len(export_data[worker_id]["top_results"]) + 1,
                        "trial_number": data["trial_number"],
                        "score": data["score"],
                        "global_score_in_trial": trial.value,
                        "parameters": self.extract_worker_params(trial, worker_id),
                        "behavior_metrics": self.get_worker_behavior(trial, worker_id),
                    })

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"\n💾 Results for Top {self.top_n} trials per worker exported to: {filename}")
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")

    def run_analysis(self):
        """Run the complete analysis."""
        print("🔍 Starting Optuna Results Analysis...")
        if not self.load_study():
            return

        analysis = self.analyze_trials()
        if not analysis["completed"]:
            logger.error("❌ No completed trials found!")
            return

        self.print_study_summary(analysis)
        top_workers = self.find_top_worker_trials(analysis["completed"])

        if not any(top_workers.values()):
            logger.error("❌ No worker-specific data found in this study!")
            logger.info("💡 Make sure the study name is correct and that the running optimizer saves 'wX_score' attributes.")
            return

        self.print_worker_results(top_workers)
        self.print_global_best(analysis["completed"])
        self.generate_summary_table(top_workers)
        self.export_results(top_workers)

        print(f"\n✅ Analysis complete! Found top {self.top_n} parameters for workers with data.")


def main():
    """Main function to run the analyzer, accepting a study name from command line."""
    # Default study name, which might be different from the optimizer's default.
    default_study_name = "adan_progressive_optimization"
    study_name = default_study_name

    # Check for a study name provided as a command-line argument.
    if len(sys.argv) > 1:
        study_name = sys.argv[1]
        print(f"🔧 Using study name provided via command line: '{study_name}'")
    else:
        print(f"ℹ️ No study name provided. Using default: '{study_name}'.")
        print(f"   To specify a study, run: python3 {sys.argv[0]} <your_study_name>")

    storage_name = f"sqlite:///{study_name}.db"
    top_n_results = 3

    analyzer = WorkerOptimizationAnalyzer(storage_name, study_name, top_n=top_n_results)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
