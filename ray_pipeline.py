import ray
import time
import random
from typing import Dict, List, Any

from adaptive_controller import AdaptiveCheckpointController
from checkpoint_store import CheckpointStore


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Global checkpoint store
STORE = CheckpointStore(store_dir="./ray_ckpts")


# --------------------------------------------------------------
# Ray Remote Tasks
# --------------------------------------------------------------

@ray.remote
def ingest_task(num_records: int) -> Dict[str, Any]:
    """Simulate data ingestion."""
    print(f"[INGEST] Processing {num_records} records...")
    time.sleep(0.12)  # Simulate work
    return {
        "records": [{"id": i, "value": random.random()} for i in range(num_records)],
        "count": num_records
    }


@ray.remote
def transform_task(records: List[Dict], k: float) -> Dict[str, Any]:
    """Simulate data transformation."""
    print(f"[TRANSFORM] Transforming {len(records)} records with k={k}...")
    time.sleep(0.38)  # Simulate work
    transformed = [{"id": r["id"], "transformed_value": r["value"] * k} for r in records]
    return {
        "records": transformed,
        "count": len(transformed)
    }


@ray.remote
def train_task(records: List[Dict], epoch: int, simulate_failure: bool = False) -> Dict[str, Any]:
    """Simulate model training."""
    print(f"[TRAIN] Training epoch {epoch} with {len(records)} records...")

    # Simulate potential failure
    if simulate_failure and random.random() < 0.3:
        raise RuntimeError(f"Simulated failure in epoch {epoch}")

    time.sleep(0.45 + epoch * 0.15)  # Simulate increasing work per epoch
    return {
        "epoch": epoch,
        "loss": 1.0 / (epoch + 1),
        "accuracy": min(0.95, 0.5 + epoch * 0.1)
    }


@ray.remote
def eval_task(model_metrics: Dict) -> Dict[str, Any]:
    """Simulate model evaluation."""
    print(f"[EVAL] Evaluating model from epoch {model_metrics['epoch']}...")
    time.sleep(0.15)  # Simulate work
    return {
        "final_accuracy": model_metrics["accuracy"],
        "final_loss": model_metrics["loss"],
        "status": "success"
    }


# --------------------------------------------------------------
# Integrated Pipeline with Adaptive Controller
# --------------------------------------------------------------

def run_pipeline(num_records: int = 100, num_epochs: int = 3, simulate_failures: bool = False):
    """
    Run complete Ray pipeline with adaptive checkpoint control.

    Args:
        num_records: Number of records to process
        num_epochs: Number of training epochs
        simulate_failures: Whether to simulate random failures
    """
    print("\n" + "="*60)
    print("Starting Adaptive Ray Pipeline")
    print("="*60)

    # Initialize adaptive controller
    controller = AdaptiveCheckpointController(
        checkpoint_dir="./ray_ckpts",
        budget_checkpoints=3,
        cost_threshold=0.35,
        learning_rate=0.1
    )

    pipeline_start = time.time()

    # --------------------------------------------------------------
    # Stage 1: Ingest
    # --------------------------------------------------------------
    print("\n--- Stage 1: Ingest ---")
    t_start = time.time()
    ingest_result = ray.get(ingest_task.remote(num_records))
    elapsed = time.time() - t_start

    controller.observe_task("ingest", elapsed)
    should_ckpt = controller.should_checkpoint("ingest", elapsed)
    print(f"[INGEST] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

    if should_ckpt:
        STORE.save("ingest_ckpt", ingest_result)

    # --------------------------------------------------------------
    # Stage 2: Transform
    # --------------------------------------------------------------
    print("\n--- Stage 2: Transform ---")
    t_start = time.time()

    try:
        transform_result = ray.get(transform_task.remote(ingest_result["records"], k=2.5))
        elapsed = time.time() - t_start

        controller.observe_task("transform", elapsed)
        should_ckpt = controller.should_checkpoint("transform", elapsed)
        print(f"[TRANSFORM] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

        if should_ckpt:
            STORE.save("transform_ckpt", transform_result)

    except Exception as e:
        print(f"[FAILURE] Transform failed: {e}")
        if STORE.exists("transform_ckpt"):
            print("[RECOVERY] Restoring transform from checkpoint...")
            transform_result = STORE.load("transform_ckpt")
            recovery_time_saved = 0.38
            checkpoint_overhead = 0.05
            controller.adjust_policy(recovery_time_saved, checkpoint_overhead)
        else:
            raise

    # --------------------------------------------------------------
    # Stage 3: Training Loop
    # --------------------------------------------------------------
    print("\n--- Stage 3: Training ---")
    model_metrics = None

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        t_start = time.time()

        try:
            model_metrics = ray.get(
                train_task.remote(
                    transform_result["records"],
                    epoch=epoch,
                    simulate_failure=simulate_failures
                )
            )
            elapsed = time.time() - t_start

            task_name = f"train_epoch_{epoch}"
            controller.observe_task(task_name, elapsed)
            should_ckpt = controller.should_checkpoint(task_name, elapsed)
            print(f"  [TRAIN] epoch={epoch} runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

            if should_ckpt:
                STORE.save(f"train_epoch_{epoch}_ckpt", model_metrics)

        except Exception as e:
            print(f"  [FAILURE] Training epoch {epoch} failed: {e}")

            # Try to recover from latest checkpoint
            recovered = False
            for prev_epoch in range(epoch - 1, -1, -1):
                ckpt_key = f"train_epoch_{prev_epoch}_ckpt"
                if STORE.exists(ckpt_key):
                    print(f"  [RECOVERY] Restoring from epoch {prev_epoch} checkpoint...")
                    model_metrics = STORE.load(ckpt_key)

                    # Calculate recovery benefit
                    recovery_time_saved = (epoch - prev_epoch) * 0.45
                    checkpoint_overhead = 0.08
                    controller.adjust_policy(recovery_time_saved, checkpoint_overhead)

                    recovered = True
                    break

            if not recovered:
                print("  [RECOVERY] No checkpoint available, restarting from scratch...")
                # In a real scenario, you'd restart the entire training
                # For demo purposes, we'll continue with a fresh model
                model_metrics = {"epoch": epoch, "loss": 1.0, "accuracy": 0.5}

    # --------------------------------------------------------------
    # Stage 4: Evaluation
    # --------------------------------------------------------------
    print("\n--- Stage 4: Evaluation ---")
    t_start = time.time()

    try:
        eval_result = ray.get(eval_task.remote(model_metrics))
        elapsed = time.time() - t_start

        controller.observe_task("eval", elapsed)
        should_ckpt = controller.should_checkpoint("eval", elapsed)
        print(f"[EVAL] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

        if should_ckpt:
            STORE.save("eval_ckpt", eval_result)

    except Exception as e:
        print(f"[FAILURE] Evaluation failed: {e}")
        if STORE.exists("eval_ckpt"):
            print("[RECOVERY] Restoring evaluation from checkpoint...")
            eval_result = STORE.load("eval_ckpt")
            recovery_time_saved = 0.15
            checkpoint_overhead = 0.03
            controller.adjust_policy(recovery_time_saved, checkpoint_overhead)
        else:
            raise

    # --------------------------------------------------------------
    # Pipeline Complete
    # --------------------------------------------------------------
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Total pipeline runtime: {pipeline_elapsed:.3f}s")
    print(f"\nFinal Results:")
    print(f"  Accuracy: {eval_result['final_accuracy']:.3f}")
    print(f"  Loss: {eval_result['final_loss']:.3f}")
    print(f"  Status: {eval_result['status']}")

    # Save controller metadata and print summary
    controller.save_metadata()
    controller.summarize()

    return eval_result


# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Adaptive Checkpoint Controller - Ray Pipeline Demo")
    print("="*60)

    # Clean previous checkpoints
    STORE.clear_all()

    # Run pipeline without failures
    print("\n\n### RUN 1: Normal execution (no failures) ###")
    result1 = run_pipeline(num_records=100, num_epochs=3, simulate_failures=False)

    # Optionally: Run with simulated failures to see recovery
    print("\n\n### RUN 2: With simulated failures ###")
    STORE.clear_all()  # Clean checkpoints between runs
    result2 = run_pipeline(num_records=100, num_epochs=3, simulate_failures=True)

    # Shutdown Ray
    ray.shutdown()
    print("\n\nPipeline demo complete!")
