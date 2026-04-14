"""
Training Callbacks

Reusable callbacks for training loops:
- EarlyStopping: stop training when metric plateaus
- MetricsLogger: log metrics to JSON for experiment tracking
"""

import json
from pathlib import Path


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement
        mode: 'min' (lower is better) or 'max' (higher is better)
        min_delta: Minimum change to qualify as an improvement
    """

    def __init__(self, patience=7, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None

    def step(self, value):
        """
        Check if training should stop.

        Args:
            value: Current metric value

        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class MetricsLogger:
    """
    Log training metrics to a JSON file for experiment tracking.
    Each call to log() appends one epoch's metrics.
    """

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.history = []

    def log(self, metrics_dict):
        """Append metrics for one epoch."""
        self.history.append(metrics_dict)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_best(self, metric_name, mode="max"):
        """Get the best value for a given metric."""
        if not self.history:
            return None
        values = [h[metric_name] for h in self.history if metric_name in h]
        if mode == "max":
            return max(values) if values else None
        return min(values) if values else None
