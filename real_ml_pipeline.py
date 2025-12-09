"""
Real ML Pipeline with Adaptive Checkpointing
Train a small neural network on MNIST dataset with Ray and adaptive checkpoint control.
"""

import ray
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, Any
import io

from adaptive_controller import AdaptiveCheckpointController
from checkpoint_store import CheckpointStore


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Global checkpoint store
STORE = CheckpointStore(store_dir="./ray_ckpts")


# --------------------------------------------------------------
# Neural Network Model
# --------------------------------------------------------------

class SimpleNN(nn.Module):
    """Small neural network for MNIST classification."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# --------------------------------------------------------------
# Ray Remote Tasks
# --------------------------------------------------------------

@ray.remote
def load_data_task(batch_size: int = 64, subset_size: int = 1000) -> Dict[str, Any]:
    """Load and preprocess MNIST dataset."""
    print(f"[DATA] Loading MNIST dataset (subset of {subset_size} samples)...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download full dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Use subset for faster demo
    train_subset = Subset(train_dataset, range(subset_size))
    test_subset = Subset(test_dataset, range(subset_size // 5))

    # Note: We return the data indices and config, not the DataLoader objects
    # because DataLoaders can't be serialized by Ray
    return {
        "train_size": len(train_subset),
        "test_size": len(test_subset),
        "batch_size": batch_size,
        "subset_size": subset_size
    }


@ray.remote
def initialize_model_task() -> Dict[str, Any]:
    """Initialize model, optimizer, and loss function."""
    print("[MODEL] Initializing neural network...")

    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Serialize model state
    model_buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_buffer)
    model_buffer.seek(0)

    return {
        "model_bytes": model_buffer.getvalue(),
        "model_info": {
            "params": sum(p.numel() for p in model.parameters()),
            "architecture": str(model)
        }
    }


@ray.remote
def train_epoch_task(
    model_bytes: bytes,
    epoch: int,
    batch_size: int,
    subset_size: int,
    simulate_failure: bool = False
) -> Dict[str, Any]:
    """Train model for one epoch."""
    print(f"[TRAIN] Starting epoch {epoch}...")

    # Simulate potential failure
    if simulate_failure and epoch > 0 and torch.rand(1).item() < 0.25:
        raise RuntimeError(f"Simulated failure during epoch {epoch}")

    # Deserialize model
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model_buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(model_buffer)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_subset = Subset(train_dataset, range(subset_size))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    print(f"  [TRAIN] Epoch {epoch} complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Serialize updated model
    model_buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'accuracy': accuracy
    }, model_buffer)
    model_buffer.seek(0)

    return {
        "model_bytes": model_buffer.getvalue(),
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "samples_trained": total
    }


@ray.remote
def evaluate_model_task(model_bytes: bytes, subset_size: int) -> Dict[str, Any]:
    """Evaluate trained model on test set."""
    print("[EVAL] Evaluating model on test set...")

    # Deserialize model
    model = SimpleNN()
    model_buffer = io.BytesIO(model_bytes)
    checkpoint = torch.load(model_buffer)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_subset = Subset(test_dataset, range(subset_size // 5))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # Evaluation
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f"  [EVAL] Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

    return {
        "test_loss": avg_loss,
        "test_accuracy": accuracy,
        "samples_evaluated": total,
        "model_bytes": model_bytes  # Keep model for potential further use
    }


# --------------------------------------------------------------
# Integrated ML Pipeline with Adaptive Controller
# --------------------------------------------------------------

def run_ml_pipeline(
    num_epochs: int = 5,
    batch_size: int = 64,
    subset_size: int = 1000,
    simulate_failures: bool = False
):
    """
    Run complete ML training pipeline with adaptive checkpoint control.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        subset_size: Size of MNIST subset to use (for faster demo)
        simulate_failures: Whether to simulate random failures
    """
    print("\n" + "="*60)
    print("Real ML Pipeline with Adaptive Checkpointing")
    print("="*60)

    # Initialize adaptive controller
    controller = AdaptiveCheckpointController(
        checkpoint_dir="./ray_ckpts",
        budget_checkpoints=3,
        cost_threshold=0.30,  # Lower threshold for ML tasks
        learning_rate=0.1
    )

    pipeline_start = time.time()

    # --------------------------------------------------------------
    # Stage 1: Data Loading
    # --------------------------------------------------------------
    print("\n--- Stage 1: Data Loading ---")
    t_start = time.time()
    data_config = ray.get(load_data_task.remote(batch_size, subset_size))
    elapsed = time.time() - t_start

    controller.observe_task("data_loading", elapsed)
    should_ckpt = controller.should_checkpoint("data_loading", elapsed)
    print(f"[DATA] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")
    print(f"       Train samples: {data_config['train_size']}, Test samples: {data_config['test_size']}")

    if should_ckpt:
        STORE.save("data_config_ckpt", data_config)

    # --------------------------------------------------------------
    # Stage 2: Model Initialization
    # --------------------------------------------------------------
    print("\n--- Stage 2: Model Initialization ---")
    t_start = time.time()
    model_state = ray.get(initialize_model_task.remote())
    elapsed = time.time() - t_start

    controller.observe_task("model_init", elapsed)
    should_ckpt = controller.should_checkpoint("model_init", elapsed)
    print(f"[MODEL] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")
    print(f"        Parameters: {model_state['model_info']['params']}")

    if should_ckpt:
        STORE.save("model_init_ckpt", model_state)

    # --------------------------------------------------------------
    # Stage 3: Training Loop
    # --------------------------------------------------------------
    print("\n--- Stage 3: Training ---")
    current_model_bytes = model_state["model_bytes"]
    training_history = []

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        t_start = time.time()

        try:
            train_result = ray.get(
                train_epoch_task.remote(
                    current_model_bytes,
                    epoch=epoch,
                    batch_size=batch_size,
                    subset_size=subset_size,
                    simulate_failure=simulate_failures
                )
            )
            elapsed = time.time() - t_start
            current_model_bytes = train_result["model_bytes"]
            training_history.append(train_result)

            task_name = f"train_epoch_{epoch}"
            controller.observe_task(task_name, elapsed)
            should_ckpt = controller.should_checkpoint(task_name, elapsed)

            print(f"  [TRAIN] epoch={epoch} runtime={elapsed:.3f}s | checkpoint={should_ckpt}")
            print(f"          Loss: {train_result['loss']:.4f}, Accuracy: {train_result['accuracy']:.2f}%")

            if should_ckpt:
                STORE.save(f"train_epoch_{epoch}_ckpt", train_result)

        except Exception as e:
            print(f"  [FAILURE] Training epoch {epoch} failed: {e}")

            # Try to recover from latest checkpoint
            recovered = False
            for prev_epoch in range(epoch - 1, -1, -1):
                ckpt_key = f"train_epoch_{prev_epoch}_ckpt"
                if STORE.exists(ckpt_key):
                    print(f"  [RECOVERY] Restoring from epoch {prev_epoch} checkpoint...")
                    train_result = STORE.load(ckpt_key)
                    current_model_bytes = train_result["model_bytes"]

                    # Calculate actual recovery benefit
                    recovery_time_saved = (epoch - prev_epoch) * 3.0  # Estimated time per epoch
                    checkpoint_overhead = 0.5  # Estimated checkpoint overhead
                    controller.adjust_policy(recovery_time_saved, checkpoint_overhead)

                    recovered = True
                    break

            if not recovered:
                print("  [RECOVERY] No checkpoint available, restarting from initial model...")
                if STORE.exists("model_init_ckpt"):
                    model_state = STORE.load("model_init_ckpt")
                    current_model_bytes = model_state["model_bytes"]
                else:
                    raise RuntimeError("Cannot recover: no checkpoints available")

    # --------------------------------------------------------------
    # Stage 4: Evaluation
    # --------------------------------------------------------------
    print("\n--- Stage 4: Evaluation ---")
    t_start = time.time()

    try:
        eval_result = ray.get(
            evaluate_model_task.remote(current_model_bytes, subset_size)
        )
        elapsed = time.time() - t_start

        controller.observe_task("evaluation", elapsed)
        should_ckpt = controller.should_checkpoint("evaluation", elapsed)
        print(f"[EVAL] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

        if should_ckpt:
            STORE.save("eval_ckpt", eval_result)

    except Exception as e:
        print(f"[FAILURE] Evaluation failed: {e}")
        if STORE.exists("eval_ckpt"):
            print("[RECOVERY] Restoring evaluation from checkpoint...")
            eval_result = STORE.load("eval_ckpt")
            recovery_time_saved = 2.0
            checkpoint_overhead = 0.3
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
    print(f"  Test Accuracy: {eval_result['test_accuracy']:.2f}%")
    print(f"  Test Loss: {eval_result['test_loss']:.4f}")
    print(f"  Total epochs trained: {len(training_history)}")

    # Training history
    if training_history:
        print(f"\nTraining History:")
        for result in training_history:
            print(f"  Epoch {result['epoch']}: Loss={result['loss']:.4f}, "
                  f"Accuracy={result['accuracy']:.2f}%")

    # Save controller metadata and print summary
    controller.save_metadata()
    controller.summarize()

    return eval_result


# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Adaptive Checkpoint Controller - Real ML Demo")
    print("Training a Neural Network on MNIST")
    print("="*60)

    # Clean previous checkpoints
    STORE.clear_all()

    # Run pipeline without failures
    print("\n\n### RUN 1: Normal training (no failures) ###")
    result1 = run_ml_pipeline(
        num_epochs=5,
        batch_size=64,
        subset_size=1000,
        simulate_failures=False
    )

    # Run with simulated failures to see recovery
    print("\n\n### RUN 2: Training with simulated failures ###")
    STORE.clear_all()
    result2 = run_ml_pipeline(
        num_epochs=5,
        batch_size=64,
        subset_size=1000,
        simulate_failures=True
    )

    # Shutdown Ray
    ray.shutdown()
    print("\n\nML Pipeline demo complete!")
