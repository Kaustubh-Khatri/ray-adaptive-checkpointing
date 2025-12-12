"""
CIFAR-10 Training Pipeline with Ray and Adaptive Checkpointing

This pipeline demonstrates:
1. Ray STATELESS tasks - for data loading and preprocessing
2. Ray STATEFUL actors - for maintaining model training state
3. Adaptive checkpoint controller - for intelligent checkpoint placement

Key Concepts:
- Stateless @ray.remote functions: No persistent state between calls
- Stateful @ray.remote classes (Actors): Maintain state across method calls
- Adaptive checkpointing: Learns optimal checkpoint frequency
"""

import ray
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import io
import pickle
from typing import Dict, Any, Tuple

from adaptive_controller import AdaptiveCheckpointController
from checkpoint_store import CheckpointStore


# Initialize Ray
ray.init(ignore_reinit_error=True)

# Global checkpoint store
STORE = CheckpointStore(store_dir="./ray_ckpts")


# --------------------------------------------------------------
# STATELESS RAY TASKS
# These are functions decorated with @ray.remote
# They don't maintain state between calls
# --------------------------------------------------------------

@ray.remote
def load_cifar10_data() -> Dict[str, Any]:
    """
    STATELESS TASK: Load and preprocess CIFAR-10 dataset.

    Ray Execution:
    - Runs as a standalone task on any available worker
    - No state is preserved after completion
    - Result is stored in Ray's object store
    - Can be executed multiple times independently
    """
    print("[STATELESS TASK] Loading CIFAR-10 data...")

    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"  Loaded {x_train.shape[0]} training samples, {x_test.shape[0]} test samples")

    # Return as dictionary (will be stored in Ray object store)
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_size": len(x_train),
        "test_size": len(x_test)
    }


@ray.remote
def create_data_subset(x_data, y_data, subset_size: int) -> Dict[str, Any]:
    """
    STATELESS TASK: Create a subset of data for faster training.

    Ray Execution:
    - Takes data from Ray object store (x_data, y_data are ObjectRefs)
    - Processes independently without maintaining state
    - Returns result to object store
    """
    print(f"[STATELESS TASK] Creating data subset of size {subset_size}...")

    indices = np.random.choice(len(x_data), subset_size, replace=False)

    return {
        "x_subset": x_data[indices],
        "y_subset": y_data[indices],
        "size": subset_size
    }


@ray.remote
def augment_data(x_batch, y_batch) -> Tuple:
    """
    STATELESS TASK: Apply data augmentation.

    Ray Execution:
    - Stateless transformation
    - Can run in parallel for different batches
    - No shared state
    """
    print(f"[STATELESS TASK] Augmenting {len(x_batch)} samples...")

    # Simple augmentation: random flips
    augmented_x = []
    for img in x_batch:
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        augmented_x.append(img)

    return np.array(augmented_x), y_batch


# --------------------------------------------------------------
# STATEFUL RAY ACTOR
# This is a class decorated with @ray.remote
# It maintains state across method calls
# --------------------------------------------------------------

@ray.remote
class CIFAR10Trainer:
    """
    STATEFUL ACTOR: Maintains model training state across epochs.

    Ray Execution:
    - Runs as a long-lived actor on a dedicated worker
    - Maintains model, optimizer, and training state in memory
    - Methods are called via actor handles (trainer.method.remote())
    - State persists between method calls
    - Perfect for iterative training processes

    Contrast with Stateless:
    - Stateless tasks would need to reload model each time
    - Actor keeps model in memory for faster epoch-to-epoch training
    """

    def __init__(self, learning_rate: float = 0.001):
        """Initialize the trainer with model and optimizer state."""
        print("[STATEFUL ACTOR] Initializing CIFAR10Trainer actor...")

        # Build the model (from your specification)
        self.model = self._build_model()

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Actor state - persists across method calls
        self.current_epoch = 0
        self.history = []
        self.learning_rate = learning_rate

        print(f"  Model initialized with {self.model.count_params()} parameters")

    def _build_model(self):
        """Build the CNN model as specified."""
        input_layer = layers.Input((32, 32, 3))

        x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)

        x = layers.Dense(128)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(rate=0.5)(x)

        output_layer = layers.Dense(10, activation='softmax')(x)

        return models.Model(input_layer, output_layer)

    def train_epoch(self, x_train, y_train, batch_size: int = 64,
                    simulate_failure: bool = False) -> Dict[str, Any]:
        """
        Train for one epoch.

        Actor State Management:
        - Uses self.model (persistent across calls)
        - Updates self.current_epoch (state maintained)
        - Appends to self.history (accumulates over time)

        This is STATEFUL - the model weights update and persist!
        """
        print(f"[STATEFUL ACTOR] Training epoch {self.current_epoch}...")

        # Simulate potential failure
        if simulate_failure and self.current_epoch > 0 and np.random.random() < 0.25:
            raise RuntimeError(f"Simulated failure during epoch {self.current_epoch}")

        # Train for one epoch
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=1,
            verbose=0,
            validation_split=0.1
        )

        # Extract metrics
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]

        print(f"  Epoch {self.current_epoch}: "
              f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Update actor state
        result = {
            "epoch": self.current_epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc)
        }
        self.history.append(result)
        self.current_epoch += 1

        return result

    def evaluate(self, x_test, y_test) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Actor State: Uses the current model state from training.
        """
        print(f"[STATEFUL ACTOR] Evaluating model after {self.current_epoch} epochs...")

        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)

        print(f"  Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

        return {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "total_epochs": self.current_epoch
        }

    def get_model_weights(self) -> bytes:
        """
        Serialize current model weights for checkpointing.

        Actor State: Extracts weights from the persistent model.
        """
        print(f"[STATEFUL ACTOR] Serializing model weights...")

        # Serialize model weights
        weights_buffer = io.BytesIO()
        np.savez_compressed(weights_buffer,
                           *self.model.get_weights())
        weights_buffer.seek(0)

        return weights_buffer.getvalue()

    def set_model_weights(self, weights_bytes: bytes):
        """
        Restore model weights from checkpoint.

        Actor State: Updates the persistent model with checkpoint data.
        """
        print(f"[STATEFUL ACTOR] Restoring model weights from checkpoint...")

        # Deserialize weights
        weights_buffer = io.BytesIO(weights_bytes)
        weights_data = np.load(weights_buffer, allow_pickle=True)

        # Load weights back into model
        weights_list = [weights_data[f'arr_{i}'] for i in range(len(weights_data.files))]
        self.model.set_weights(weights_list)

    def get_training_state(self) -> Dict[str, Any]:
        """
        Get complete training state for checkpointing.

        Actor State: Captures all persistent state.
        """
        return {
            "epoch": self.current_epoch,
            "history": self.history,
            "model_weights": self.get_model_weights(),
            "learning_rate": self.learning_rate
        }

    def restore_training_state(self, state: Dict[str, Any]):
        """
        Restore complete training state from checkpoint.

        Actor State: Resets all persistent state.
        """
        print(f"[STATEFUL ACTOR] Restoring training state from epoch {state['epoch']}...")

        self.current_epoch = state["epoch"]
        self.history = state["history"]
        self.learning_rate = state["learning_rate"]
        self.set_model_weights(state["model_weights"])


# --------------------------------------------------------------
# Main Pipeline with Adaptive Controller
# --------------------------------------------------------------

def run_cifar10_pipeline(
    num_epochs: int = 10,
    batch_size: int = 64,
    train_subset_size: int = 5000,
    test_subset_size: int = 1000,
    simulate_failures: bool = False
):
    """
    Complete CIFAR-10 training pipeline demonstrating:
    1. Stateless Ray tasks for data operations
    2. Stateful Ray actors for model training
    3. Adaptive checkpoint controller
    """
    print("\n" + "="*70)
    print("CIFAR-10 Training with Ray and Adaptive Checkpointing")
    print("="*70)

    # Initialize adaptive controller
    controller = AdaptiveCheckpointController(
        checkpoint_dir="./ray_ckpts",
        budget_checkpoints=4,
        cost_threshold=0.30,
        learning_rate=0.1
    )

    pipeline_start = time.time()

    # --------------------------------------------------------------
    # Stage 1: Data Loading (STATELESS TASK)
    # --------------------------------------------------------------
    print("\n--- Stage 1: Data Loading (STATELESS) ---")
    print("Ray Execution: Launches a stateless task on any available worker")
    print("               Task completes and result stored in object store")

    t_start = time.time()
    data_ref = load_cifar10_data.remote()  # Returns ObjectRef
    data = ray.get(data_ref)  # Retrieve from object store
    elapsed = time.time() - t_start

    controller.observe_task("data_loading", elapsed)
    should_ckpt = controller.should_checkpoint("data_loading", elapsed)
    print(f"\n[CHECKPOINT DECISION] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

    if should_ckpt:
        STORE.save("data_ckpt", data)

    # --------------------------------------------------------------
    # Stage 2: Create Subsets (STATELESS TASK)
    # --------------------------------------------------------------
    print("\n--- Stage 2: Create Data Subsets (STATELESS) ---")
    print("Ray Execution: Another stateless task, independent of previous task")

    t_start = time.time()
    train_subset_ref = create_data_subset.remote(
        data["x_train"], data["y_train"], train_subset_size
    )
    test_subset_ref = create_data_subset.remote(
        data["x_test"], data["y_test"], test_subset_size
    )

    # Both tasks run in parallel
    train_subset, test_subset = ray.get([train_subset_ref, test_subset_ref])
    elapsed = time.time() - t_start

    controller.observe_task("data_subset", elapsed)
    should_ckpt = controller.should_checkpoint("data_subset", elapsed)
    print(f"\n[CHECKPOINT DECISION] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

    if should_ckpt:
        STORE.save("subset_ckpt", {"train": train_subset, "test": test_subset})

    # --------------------------------------------------------------
    # Stage 3: Initialize Trainer (STATEFUL ACTOR)
    # --------------------------------------------------------------
    print("\n--- Stage 3: Initialize Trainer (STATEFUL ACTOR) ---")
    print("Ray Execution: Creates a stateful actor that persists in memory")
    print("               Actor maintains model, optimizer, and training state")

    t_start = time.time()
    trainer = CIFAR10Trainer.remote(learning_rate=0.001)  # Returns ActorHandle
    ray.get(trainer.__ray_ready__.remote())  # Wait for actor initialization
    elapsed = time.time() - t_start

    controller.observe_task("trainer_init", elapsed)
    should_ckpt = controller.should_checkpoint("trainer_init", elapsed)
    print(f"\n[CHECKPOINT DECISION] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

    # --------------------------------------------------------------
    # Stage 4: Training Loop (STATEFUL ACTOR METHODS)
    # --------------------------------------------------------------
    print("\n--- Stage 4: Training Loop (STATEFUL ACTOR) ---")
    print("Ray Execution: Repeatedly calls actor methods")
    print("               Actor state (model weights) persists between calls")
    print("               Each epoch builds on previous epoch's weights")

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")
        t_start = time.time()

        try:
            # Call actor method (returns ObjectRef)
            result_ref = trainer.train_epoch.remote(
                train_subset["x_subset"],
                train_subset["y_subset"],
                batch_size=batch_size,
                simulate_failure=simulate_failures
            )
            result = ray.get(result_ref)  # Wait for completion
            elapsed = time.time() - t_start

            task_name = f"train_epoch_{epoch}"
            controller.observe_task(task_name, elapsed)
            should_ckpt = controller.should_checkpoint(task_name, elapsed)

            print(f"\n  [CHECKPOINT DECISION] epoch={epoch} runtime={elapsed:.3f}s | "
                  f"checkpoint={should_ckpt}")

            if should_ckpt:
                # Get complete training state from actor
                state_ref = trainer.get_training_state.remote()
                state = ray.get(state_ref)
                STORE.save(f"train_epoch_{epoch}_ckpt", state)
                print(f"  [STORED] Saved model weights and training state")

        except Exception as e:
            print(f"\n  [FAILURE] Training epoch {epoch} failed: {e}")

            # Recovery: Find most recent checkpoint
            recovered = False
            for prev_epoch in range(epoch - 1, -1, -1):
                ckpt_key = f"train_epoch_{prev_epoch}_ckpt"
                if STORE.exists(ckpt_key):
                    print(f"  [RECOVERY] Restoring from epoch {prev_epoch} checkpoint...")

                    # Load checkpoint state
                    state = STORE.load(ckpt_key)

                    # Restore state to actor
                    ray.get(trainer.restore_training_state.remote(state))

                    # Calculate recovery benefit
                    recovery_time_saved = (epoch - prev_epoch) * 15.0  # Est. time per epoch
                    checkpoint_overhead = 1.0
                    controller.adjust_policy(recovery_time_saved, checkpoint_overhead)

                    recovered = True
                    break

            if not recovered:
                print("  [RECOVERY] No checkpoint available!")
                raise

    # --------------------------------------------------------------
    # Stage 5: Evaluation (STATEFUL ACTOR METHOD)
    # --------------------------------------------------------------
    print("\n--- Stage 5: Evaluation (STATEFUL ACTOR) ---")
    print("Ray Execution: Calls actor method to evaluate trained model")

    t_start = time.time()
    eval_ref = trainer.evaluate.remote(
        test_subset["x_subset"],
        test_subset["y_subset"]
    )
    eval_result = ray.get(eval_ref)
    elapsed = time.time() - t_start

    controller.observe_task("evaluation", elapsed)
    should_ckpt = controller.should_checkpoint("evaluation", elapsed)
    print(f"\n[CHECKPOINT DECISION] runtime={elapsed:.3f}s | checkpoint={should_ckpt}")

    # --------------------------------------------------------------
    # Pipeline Complete
    # --------------------------------------------------------------
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)
    print(f"Total runtime: {pipeline_elapsed:.3f}s")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {eval_result['test_accuracy']*100:.2f}%")
    print(f"  Test Loss: {eval_result['test_loss']:.4f}")
    print(f"  Total Epochs: {eval_result['total_epochs']}")

    # Controller summary
    controller.save_metadata()
    controller.summarize()

    return eval_result


# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CIFAR-10 Training with Ray: Stateless vs Stateful Execution")
    print("="*70)

    # Clean previous checkpoints
    STORE.clear_all()

    print("\n### Demonstration: Normal Training ###")
    result = run_cifar10_pipeline(
        num_epochs=10,
        batch_size=64,
        train_subset_size=5000,
        test_subset_size=1000,
        simulate_failures=False
    )

    print("\n\n### Demonstration: Training with Failures ###")
    STORE.clear_all()
    result_with_failures = run_cifar10_pipeline(
        num_epochs=10,
        batch_size=64,
        train_subset_size=5000,
        test_subset_size=1000,
        simulate_failures=True
    )

    # Shutdown Ray
    ray.shutdown()
    print("\n\nCIFAR-10 pipeline complete!")
