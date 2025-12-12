import json
import time
from pathlib import Path

class AdaptiveCheckpointController:
    """
    A lightweight adaptive checkpoint controller that decides
    whether to checkpoint or skip, based on task runtime,
    cost-benefit ratio, and a dynamic feedback policy.

    How It Adapts:
    1. OBSERVE: Track runtime of each completed task
    2. DECIDE: Checkpoint if (runtime/max_seen >= threshold) AND (budget available)
    3. LEARN: After recovery, adjust threshold based on benefit

    Adaptive Formula:
        benefit = recovery_time_saved - checkpoint_overhead
        delta = -sign(benefit) × learning_rate × |benefit| / (|benefit| + 1)
        threshold_new = clamp(threshold_old + delta, 0.05, 1.0)

    Key Insight:
        - Positive benefit (checkpoint saved time) → Lower threshold → Checkpoint MORE
        - Negative benefit (checkpoint wasted time) → Raise threshold → Checkpoint LESS

    See ADAPTIVE_CHECKPOINTING_EXPLAINED.md for detailed explanation.
    """

    def __init__(self,
                 checkpoint_dir="./ray_ckpts",
                 budget_checkpoints=3,
                 cost_threshold=0.35,
                 learning_rate=0.1):
        """
        Args:
            checkpoint_dir (str): Directory to store checkpoints and metadata.
            budget_checkpoints (int): Maximum # of checkpoints allowed per run.
            cost_threshold (float): Initial threshold for task runtime ratio.
            learning_rate (float): Learning rate for adaptive threshold updates.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.budget = budget_checkpoints
        self.used = 0
        self.cost_threshold = cost_threshold
        self.learning_rate = learning_rate
        self.max_time_seen = 1e-6
        self.decision_log = []
        self.history = []  # (task_name, runtime, decision, recovered)

    # --------------------------------------------------------------
    # Core Decision Functions
    # --------------------------------------------------------------
    def observe_task(self, task_name: str, runtime: float):
        """Record runtime of completed task."""
        self.max_time_seen = max(self.max_time_seen, runtime)
        self.history.append((task_name, runtime, None, False))

    def should_checkpoint(self, task_name: str, estimated_cost: float):
        """
        Decide whether to checkpoint a task based on its estimated cost
        relative to the maximum observed runtime and remaining budget.
        """
        if self.used >= self.budget:
            decision = False
        else:
            ratio = estimated_cost / max(self.max_time_seen, 1e-6)
            decision = ratio >= self.cost_threshold

        if decision:
            self.used += 1

        self.decision_log.append({
            "task": task_name,
            "est_cost": estimated_cost,
            "max_seen": self.max_time_seen,
            "threshold": self.cost_threshold,
            "decision": decision,
            "timestamp": time.time()
        })
        return decision

    # --------------------------------------------------------------
    # Feedback Loop
    # --------------------------------------------------------------
    def adjust_policy(self, recovery_time_saved: float, checkpoint_overhead: float):
        """
        Update cost threshold adaptively based on observed benefit.
        If checkpointing saved significant recovery time, lower threshold
        (checkpoint more often); else, increase threshold.
        """
        benefit = (recovery_time_saved - checkpoint_overhead)
        direction = -1 if benefit > 0 else 1
        delta = direction * self.learning_rate * abs(benefit) / (abs(benefit) + 1)
        self.cost_threshold = max(0.05, min(1.0, self.cost_threshold + delta))
        print(f"[ADAPT] New cost threshold: {self.cost_threshold:.3f} (benefit={benefit:.3f})")

    # --------------------------------------------------------------
    # Metadata and Logging
    # --------------------------------------------------------------
    def save_metadata(self):
        """Persist decision log to JSON for analysis or visualization."""
        metadata_path = self.checkpoint_dir / "adaptive_decisions.json"
        with open(metadata_path, "w") as f:
            json.dump(self.decision_log, f, indent=2)
        print(f"[ACC] Decisions saved to {metadata_path}")

    def summarize(self):
        """Print a human-readable summary of checkpoint decisions."""
        print("\n=== Adaptive Controller Summary ===")
        print(f"Budget used: {self.used}/{self.budget}")
        print(f"Final threshold: {self.cost_threshold:.3f}")
        if not self.decision_log:
            print("No decisions recorded.")
            return
        for d in self.decision_log:
            print(f"{d['task']:>15} | cost={d['est_cost']:.3f} | "
                  f"max={d['max_seen']:.3f} | thr={d['threshold']:.2f} | "
                  f"ckpt={d['decision']}")

# --------------------------------------------------------------
# Example Usage (standalone test)
# --------------------------------------------------------------
if __name__ == "__main__":
    acc = AdaptiveCheckpointController(budget_checkpoints=3, cost_threshold=0.4)

    simulated_tasks = {
        "ingest": 0.12,
        "transform": 0.38,
        "train_epoch_1": 0.45,
        "train_epoch_2": 0.80,
        "eval": 0.15,
    }

    for name, runtime in simulated_tasks.items():
        acc.observe_task(name, runtime)
        decision = acc.should_checkpoint(name, estimated_cost=runtime)
        print(f"Task {name:<15} | runtime={runtime:.3f}s | checkpoint={decision}")

    # Simulate feedback from recovery experiment
    acc.adjust_policy(recovery_time_saved=0.6, checkpoint_overhead=0.1)
    acc.save_metadata()
    acc.summarize()
