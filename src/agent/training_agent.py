"""
AI Training Agent for Dog Aggression Detection

Inspired by n-autoresearch's experiment automation patterns.
This agent automatically:
1. Runs training experiments with different hyperparameters
2. Tracks all results in an experiment log
3. Adapts search strategy based on results (explore vs exploit)
4. Keeps best models, discards underperformers
5. Suggests next experiments based on what worked

Usage:
    agent = TrainingAgent(config)
    agent.run()  # Fully automated training loop
"""

import json
import copy
import random
import time
from pathlib import Path
from datetime import datetime

import yaml


class ExperimentTracker:
    """
    Tracks all training experiments: config, metrics, status.
    Persists to JSON for full experiment history.
    """

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self, log_path="experiments/experiment_log.json"):
        self.log_path = Path(log_path)
        if not self.log_path.is_absolute():
            self.log_path = self.PROJECT_ROOT / self.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.experiments = self._load()

    def _load(self):
        if self.log_path.exists():
            with open(self.log_path, "r") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def register(self, experiment_id, config, hypothesis=""):
        """Register a new experiment before training starts."""
        entry = {
            "id": experiment_id,
            "status": "running",
            "hypothesis": hypothesis,
            "config": config,
            "metrics": None,
            "timestamp": datetime.now().isoformat(),
        }
        self.experiments.append(entry)
        self._save()
        return entry

    def complete(self, experiment_id, metrics, status="completed"):
        """Mark experiment as complete with final metrics."""
        for exp in self.experiments:
            if exp["id"] == experiment_id:
                exp["status"] = status
                exp["metrics"] = metrics
                exp["completed_at"] = datetime.now().isoformat()
                break
        self._save()

    def get_best(self, metric="best_f1", mode="max"):
        """Get the best experiment by a given metric."""
        completed = [e for e in self.experiments if e["status"] == "completed"]
        if not completed:
            return None

        if mode == "max":
            return max(completed, key=lambda e: e["metrics"].get(metric, 0))
        return min(completed, key=lambda e: e["metrics"].get(metric, float("inf")))

    def get_history(self):
        """Get all completed experiments sorted by metric."""
        return sorted(
            [e for e in self.experiments if e["status"] == "completed"],
            key=lambda e: e["metrics"].get("best_f1", 0),
            reverse=True,
        )

    def get_stats(self):
        """Get summary statistics of all experiments."""
        completed = [e for e in self.experiments if e["status"] == "completed"]
        crashed = [e for e in self.experiments if e["status"] == "crashed"]

        if not completed:
            return {"total": len(self.experiments), "completed": 0}

        f1_scores = [e["metrics"]["best_f1"] for e in completed]

        return {
            "total": len(self.experiments),
            "completed": len(completed),
            "crashed": len(crashed),
            "best_f1": max(f1_scores),
            "avg_f1": sum(f1_scores) / len(f1_scores),
            "worst_f1": min(f1_scores),
        }


class SearchStrategy:
    """
    Decides which hyperparameters to try next.
    Adapts between explore (random search) and exploit (refine best).

    Inspired by n-autoresearch's adaptive search strategy.
    """

    def __init__(self, search_space, seed=42):
        self.search_space = search_space
        self.mode = "explore"  # explore, exploit, adaptive
        self.rng = random.Random(seed)

    def suggest(self, tracker):
        """
        Suggest next hyperparameter configuration.

        In EXPLORE mode: random sampling from search space
        In EXPLOIT mode: small perturbations of best config
        In ADAPTIVE mode: auto-switch based on experiment history
        """
        stats = tracker.get_stats()

        # Auto-adapt strategy
        if self.mode == "adaptive":
            self._adapt(stats, tracker)

        if self.mode == "exploit" and tracker.get_best():
            return self._exploit(tracker)
        return self._explore()

    def _explore(self):
        """Random sampling from search space."""
        config = {}
        hypothesis_parts = []

        for param, values in self.search_space.items():
            if isinstance(values, list):
                if all(isinstance(v, (int, float)) for v in values) and len(values) == 2:
                    # Continuous range [min, max]
                    if isinstance(values[0], int) and isinstance(values[1], int):
                        val = self.rng.randint(values[0], values[1])
                    else:
                        val = round(self.rng.uniform(values[0], values[1]), 6)
                else:
                    # Categorical choices
                    val = self.rng.choice(values)
                config[param] = val
                hypothesis_parts.append(f"{param}={val}")

        hypothesis = f"Explore: {', '.join(hypothesis_parts)}"
        return config, hypothesis

    def _exploit(self, tracker):
        """Small perturbations around the best configuration."""
        best = tracker.get_best()
        best_config = best["config"]
        config = {}
        hypothesis_parts = []

        for param, values in self.search_space.items():
            base_val = best_config.get(param)

            if base_val is None:
                # Param not in best config, sample randomly
                if isinstance(values, list):
                    config[param] = self.rng.choice(values)
                continue

            if isinstance(base_val, float):
                # Perturb by ±20%
                delta = base_val * 0.2
                new_val = round(base_val + self.rng.uniform(-delta, delta), 6)
                # Clamp to search space bounds
                if isinstance(values, list) and len(values) == 2:
                    new_val = max(values[0], min(values[1], new_val))
                config[param] = new_val
                hypothesis_parts.append(f"{param}: {base_val}->{new_val}")

            elif isinstance(base_val, int):
                # Small integer perturbation
                config[param] = base_val
                hypothesis_parts.append(f"{param}={base_val} (kept)")

            else:
                # Categorical: 70% keep, 30% random
                if self.rng.random() < 0.7:
                    config[param] = base_val
                else:
                    config[param] = self.rng.choice(values)
                    hypothesis_parts.append(f"{param}: {base_val}->{config[param]}")

        hypothesis = f"Exploit (refine best): {', '.join(hypothesis_parts)}"
        return config, hypothesis

    def _adapt(self, stats, tracker):
        """Auto-adapt strategy based on experiment history."""
        if stats["completed"] < 3:
            self.mode = "explore"
            return

        history = tracker.get_history()
        recent = history[:5]

        # If recent experiments are all close to best, switch to exploit
        best_f1 = stats["best_f1"]
        near_best = sum(1 for e in recent
                        if abs(e["metrics"]["best_f1"] - best_f1) < 0.02)

        if near_best >= 3:
            self.mode = "exploit"
            print("  [agent] Strategy: EXPLOIT (refining around best)")
        else:
            self.mode = "explore"
            print("  [agent] Strategy: EXPLORE (searching broadly)")


class TrainingAgent:
    """
    AI Training Agent - Automated experiment runner.

    Runs a loop of:
    1. Suggest hyperparameters
    2. Train model with those params
    3. Evaluate results
    4. Keep/discard based on improvement
    5. Adapt strategy and repeat
    """

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    def __init__(self, config_path="config/config.yaml"):
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.PROJECT_ROOT / config_path
        with open(config_path, "r") as f:
            self.base_config = yaml.safe_load(f)

        agent_cfg = self.base_config.get("agent", {})
        self.max_experiments = agent_cfg.get("max_experiments", 20)
        self.improvement_threshold = agent_cfg.get("improvement_threshold", 0.005)

        self.tracker = ExperimentTracker()
        self.strategy = SearchStrategy(
            search_space=agent_cfg.get("search_space", {}),
            seed=self.base_config["project"].get("seed", 42),
        )
        self.strategy.mode = agent_cfg.get("search_strategy", "adaptive")

    def run(self, stage="classifier"):
        """
        Run the automated training loop.

        Args:
            stage: 'detector' or 'classifier'
        """
        print("\n" + "=" * 60)
        print("  AI TRAINING AGENT")
        print("=" * 60)
        print(f"  Stage:           {stage}")
        print(f"  Max experiments: {self.max_experiments}")
        print(f"  Strategy:        {self.strategy.mode}")
        print("=" * 60 + "\n")

        for i in range(self.max_experiments):
            print(f"\n{'─' * 50}")
            print(f"  EXPERIMENT {i+1}/{self.max_experiments}")
            print(f"{'─' * 50}")

            # 1. Suggest hyperparameters
            suggested_params, hypothesis = self.strategy.suggest(self.tracker)
            print(f"  Hypothesis: {hypothesis}")

            # 2. Build experiment config
            exp_config = self._build_experiment_config(suggested_params, stage)
            exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"

            # 3. Register experiment
            self.tracker.register(exp_id, suggested_params, hypothesis)

            # 4. Train
            try:
                metrics = self._run_training(exp_config, stage)
                self.tracker.complete(exp_id, metrics, status="completed")
            except Exception as e:
                print(f"  CRASHED: {e}")
                self.tracker.complete(exp_id, {"error": str(e)}, status="crashed")
                continue

            # 5. Check improvement
            best = self.tracker.get_best()
            if best and best["id"] == exp_id:
                print(f"\n  >> NEW BEST! F1: {metrics.get('best_f1', 0):.4f}")
            else:
                current_best = best["metrics"]["best_f1"] if best else 0
                print(f"  Result F1: {metrics.get('best_f1', 0):.4f} "
                      f"(best: {current_best:.4f})")

            # 6. Print running stats
            stats = self.tracker.get_stats()
            print(f"\n  Stats: {stats['completed']}/{stats['total']} completed, "
                  f"best F1: {stats.get('best_f1', 0):.4f}, "
                  f"avg F1: {stats.get('avg_f1', 0):.4f}")

        # Final report
        self._print_final_report()

    def _build_experiment_config(self, params, stage):
        """Merge suggested params into base config."""
        config = copy.deepcopy(self.base_config)
        stage_config = config[stage]

        for key, value in params.items():
            if key in stage_config:
                stage_config[key] = value

        return config

    def _run_training(self, config, stage):
        """Run a single training experiment."""
        if stage == "detector":
            from src.training.train_detector import train_detector
            return train_detector(config)
        else:
            from src.training.train_classifier import train_classifier
            return train_classifier(config)

    def _print_final_report(self):
        """Print summary of all experiments."""
        stats = self.tracker.get_stats()
        history = self.tracker.get_history()

        print("\n" + "=" * 60)
        print("  AGENT FINAL REPORT")
        print("=" * 60)
        print(f"  Total experiments: {stats['total']}")
        print(f"  Completed:         {stats['completed']}")
        print(f"  Crashed:           {stats.get('crashed', 0)}")
        print(f"  Best F1:           {stats.get('best_f1', 0):.4f}")
        print(f"  Average F1:        {stats.get('avg_f1', 0):.4f}")

        if history:
            print(f"\n  TOP 5 EXPERIMENTS:")
            for i, exp in enumerate(history[:5]):
                print(f"    {i+1}. {exp['id']} — "
                      f"F1: {exp['metrics'].get('best_f1', 0):.4f} — "
                      f"{exp['hypothesis']}")

        best = self.tracker.get_best()
        if best:
            print(f"\n  BEST CONFIG:")
            for k, v in best["config"].items():
                print(f"    {k}: {v}")

            print(f"\n  Best model saved at: {best['metrics'].get('run_dir', 'N/A')}")

        print("=" * 60 + "\n")

    def run_single(self, stage="classifier", **override_params):
        """
        Run a single experiment with specific params (useful for manual tuning).
        """
        config = copy.deepcopy(self.base_config)
        for key, value in override_params.items():
            if key in config[stage]:
                config[stage][key] = value

        exp_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracker.register(exp_id, override_params, "Manual experiment")

        try:
            metrics = self._run_training(config, stage)
            self.tracker.complete(exp_id, metrics)
            print(f"\n  Manual experiment complete — F1: {metrics.get('best_f1', 0):.4f}")
        except Exception as e:
            self.tracker.complete(exp_id, {"error": str(e)}, status="crashed")
            print(f"\n  Manual experiment crashed: {e}")

        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Training Agent")
    parser.add_argument("--stage", type=str, default="classifier",
                        choices=["detector", "classifier"],
                        help="Which model to train")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config file")
    parser.add_argument("--max-experiments", type=int, default=None,
                        help="Override max experiments")
    args = parser.parse_args()

    agent = TrainingAgent(config_path=args.config)
    if args.max_experiments:
        agent.max_experiments = args.max_experiments

    agent.run(stage=args.stage)
