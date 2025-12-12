# Adaptive Checkpoint Controller for Ray

A lightweight, intelligent checkpoint controller for Ray distributed systems that adaptively decides when to checkpoint based on task runtime, cost-benefit analysis, and dynamic feedback learning.

Includes **real ML training examples** with PyTorch (MNIST) and TensorFlow (CIFAR-10), demonstrating both Ray's stateless tasks and stateful actors!

## Features

- **Adaptive Decision Making**: Dynamically determines which tasks to checkpoint based on runtime patterns
- **Budget-Aware**: Respects checkpoint budget constraints to avoid excessive storage overhead
- **Self-Learning**: Adjusts checkpoint strategy based on observed recovery benefits
- **Cost-Benefit Analysis**: Compares checkpoint overhead vs. recovery time savings
- **Detailed Logging**: Exports decision metadata for analysis and visualization
- **Real ML Support**: Checkpoints actual model weights (PyTorch, TensorFlow) with state restoration on failure
- **Ray Execution Patterns**: Demonstrates stateless tasks vs stateful actors with comprehensive documentation

## What's Included

This repository contains:

1. **Simulated Pipeline** (`ray_pipeline.py`) - Demo with fake data to show checkpoint decision logic
2. **PyTorch MNIST Pipeline** (`real_ml_pipeline.py`) - Neural network training with stateless tasks
3. **TensorFlow CIFAR-10 Pipeline** (`cifar10_pipeline.py`) - CNN training with stateless tasks AND stateful actors
4. **Adaptive Controller** (`adaptive_controller.py`) - Core decision engine that works with all pipelines
5. **Ray Execution Guide** (`RAY_EXECUTION_GUIDE.md`) - Comprehensive documentation on stateless vs stateful execution

## Architecture

The system consists of three main components:

1. **AdaptiveCheckpointController**: Core decision engine that determines checkpoint placement
2. **CheckpointStore**: Simple persistent storage for pipeline intermediate states and model weights
3. **Ray Pipeline**: Distributed data processing or ML training pipeline with integrated checkpoint control

## Installation

```bash
# Install all dependencies (Ray, PyTorch, TensorFlow)
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ray torch torchvision tensorflow
```

## Quick Start

### 1. Run the Standalone Controller Test

```bash
python adaptive_controller.py
```

This runs a simulation with predefined task runtimes to demonstrate the decision logic.

### 2. Run the Simulated Ray Pipeline

```bash
python ray_pipeline.py
```

This executes a simulated distributed pipeline with fake data to demonstrate checkpoint decisions.

### 3. Run PyTorch MNIST Training

```bash
python real_ml_pipeline.py
```

Trains a 3-layer neural network on MNIST using **stateless Ray tasks**:
- Loads real MNIST dataset
- Trains with PyTorch
- Saves actual model weights in checkpoints
- Demonstrates recovery with model state restoration

Quick demo:
```bash
python run_ml_demo.py
```

### 4. Run TensorFlow CIFAR-10 Training (NEW!)

```bash
python cifar10_pipeline.py
```

Trains a CNN on CIFAR-10 demonstrating **both stateless tasks AND stateful actors**:
- **Stateless tasks**: Data loading, preprocessing (no state preserved)
- **Stateful actors**: Model training (persistent state across epochs)
- Shows how Ray handles each execution pattern differently
- Includes adaptive checkpointing of actor state

Quick demo:
```bash
python run_cifar10_demo.py
```

**See also:**
- `RAY_EXECUTION_GUIDE.md` - Detailed explanation of stateless vs stateful execution
- `ADAPTIVE_CHECKPOINTING_EXPLAINED.md` - Deep dive into how adaptive learning works

## Usage

### Basic Integration

```python
from adaptive_controller import AdaptiveCheckpointController
from checkpoint_store import CheckpointStore

# Initialize controller
controller = AdaptiveCheckpointController(
    checkpoint_dir="./ray_ckpts",
    budget_checkpoints=3,
    cost_threshold=0.35,
    learning_rate=0.1
)

# Initialize storage
store = CheckpointStore(store_dir="./ray_ckpts")

# In your pipeline
import time

# Execute task
t_start = time.time()
result = your_task_function()
elapsed = time.time() - t_start

# Record observation
controller.observe_task("task_name", elapsed)

# Decide whether to checkpoint
should_ckpt = controller.should_checkpoint("task_name", elapsed)

if should_ckpt:
    store.save("task_name_ckpt", result)
```

### Recovery Pattern

```python
try:
    result = ray.get(task.remote())
except Exception as e:
    print(f"Task failed: {e}")

    # Attempt recovery from checkpoint
    if store.exists("task_ckpt"):
        print("Restoring from checkpoint...")
        result = store.load("task_ckpt")

        # Provide feedback to controller
        recovery_time_saved = 2.5   # Time saved by not recomputing
        checkpoint_overhead = 0.3   # Time spent checkpointing
        controller.adjust_policy(recovery_time_saved, checkpoint_overhead)
    else:
        raise
```

### End of Pipeline

```python
# Save decision log
controller.save_metadata()

# Print summary
controller.summarize()
```

## Configuration Parameters

### AdaptiveCheckpointController

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | str | `"./ray_ckpts"` | Directory for checkpoints and metadata |
| `budget_checkpoints` | int | `3` | Maximum number of checkpoints allowed |
| `cost_threshold` | float | `0.35` | Initial runtime ratio threshold for checkpointing |
| `learning_rate` | float | `0.1` | Rate at which threshold adapts based on feedback |

### Key Methods

**observe_task(task_name: str, runtime: float)**
- Records the runtime of a completed task
- Updates max observed runtime for future comparisons

**should_checkpoint(task_name: str, estimated_cost: float) -> bool**
- Decides whether to checkpoint based on:
  - Remaining checkpoint budget
  - Runtime ratio vs. max observed time
  - Current cost threshold
- Returns `True` if checkpoint should be created

**adjust_policy(recovery_time_saved: float, checkpoint_overhead: float)**
- Adjusts `cost_threshold` based on observed benefit
- Lowers threshold if checkpointing saved time (checkpoint more)
- Raises threshold if overhead exceeded benefit (checkpoint less)

**save_metadata()**
- Exports decision log to `adaptive_decisions.json`

**summarize()**
- Prints human-readable summary of all checkpoint decisions

## How It Works

### Decision Algorithm

The controller uses a **3-phase adaptive approach**:

1. **Observation Phase**: Tracks runtime of each completed task
   ```python
   controller.observe_task("train_epoch_0", elapsed=15.3)
   # Updates max_time_seen to track longest task
   ```

2. **Decision Phase**: For each task, computes:
   ```python
   ratio = estimated_cost / max_time_seen  # e.g., 15.3 / 15.3 = 1.0
   decision = (ratio >= cost_threshold) AND (used < budget)
   # Example: (1.0 >= 0.35) AND (1 < 3) → True (checkpoint!)
   ```

3. **Learning Phase**: After recovery, adjusts threshold based on benefit:
   ```python
   benefit = recovery_time_saved - checkpoint_overhead
   # Example: 14.0 - 1.0 = 13.0 (positive = good!)

   delta = -sign(benefit) × learning_rate × |benefit| / (|benefit| + 1)
   # Example: -1 × 0.1 × 13.0/14.0 = -0.093

   new_threshold = clamp(old_threshold + delta, 0.05, 1.0)
   # Example: 0.35 - 0.093 = 0.257 (lower = checkpoint more!)
   ```

### Adaptive Behavior

The controller **learns** from experience:

| Outcome | Benefit | Action | Effect |
|---------|---------|--------|--------|
| **Checkpoint saved time** | Positive (+14s) | Lower threshold | Checkpoint MORE often |
| **Checkpoint wasted time** | Negative (-2s) | Raise threshold | Checkpoint LESS often |
| **Marginal benefit** | Small (+0.5s) | Small adjustment | Gentle tuning |

**Evolution over time:**
```
Initial:  threshold=0.35 (checkpoint if task is >35% of max runtime)
           ↓
Recovery: Saved 14 seconds by checkpointing
           ↓
Adapted:  threshold=0.257 (checkpoint if task is >26% of max runtime)
           ↓
Result:   More aggressive checkpointing in failure-prone environment
```

**Budget constraint**: Stops checkpointing when budget is exhausted, regardless of threshold

**📖 For detailed mathematical explanation, see `ADAPTIVE_CHECKPOINTING_EXPLAINED.md`**

## Example Output

### Simulated Pipeline
```
--- Stage 2: Transform ---
[TRANSFORM] Transforming 100 records with k=2.5...
[TRANSFORM] runtime=0.380s | checkpoint=True
[STORE] Saved checkpoint: transform_ckpt

--- Stage 3: Training ---
  Epoch 1/3
  [TRAIN] epoch=0 runtime=0.452s | checkpoint=True
  [STORE] Saved checkpoint: train_epoch_0_ckpt

=== Adaptive Controller Summary ===
Budget used: 2/3
Final threshold: 0.350
```

### PyTorch MNIST Pipeline
```
--- Stage 2: Model Initialization ---
[MODEL] Initializing neural network...
[MODEL] runtime=0.034s | checkpoint=False
        Parameters: 109386

--- Stage 3: Training ---
  Epoch 1/5
  [TRAIN] Epoch 0 complete - Loss: 0.4532, Accuracy: 87.30%
  [TRAIN] epoch=0 runtime=3.245s | checkpoint=True
  [STORE] Saved checkpoint: train_epoch_0_ckpt

--- Stage 4: Evaluation ---
[EVAL] Test Loss: 0.3821, Test Accuracy: 89.50%
```

### TensorFlow CIFAR-10 Pipeline (Stateless + Stateful)
```
--- Stage 1: Data Loading (STATELESS) ---
Ray Execution: Launches a stateless task on any available worker
[STATELESS TASK] Loading CIFAR-10 data...
  Loaded 50000 training samples, 10000 test samples
[CHECKPOINT DECISION] runtime=2.341s | checkpoint=False

--- Stage 3: Initialize Trainer (STATEFUL ACTOR) ---
Ray Execution: Creates a stateful actor that persists in memory
[STATEFUL ACTOR] Initializing CIFAR10Trainer actor...
  Model initialized with 664266 parameters

--- Stage 4: Training Loop (STATEFUL ACTOR) ---
Ray Execution: Repeatedly calls actor methods
               Actor state (model weights) persists between calls

  Epoch 1/10
  [STATEFUL ACTOR] Training epoch 0...
  Epoch 0: loss=1.8234, acc=0.3421, val_loss=1.6543, val_acc=0.3890
  [CHECKPOINT DECISION] epoch=0 runtime=14.523s | checkpoint=True
  [STORED] Saved model weights and training state

  Epoch 2/10
  [FAILURE] Training epoch 1 failed: Simulated failure
  [RECOVERY] Restoring from epoch 0 checkpoint...
  [STATEFUL ACTOR] Restoring training state from epoch 0...
  [ADAPT] New cost threshold: 0.270 (benefit=14.000)

Final Results:
  Test Accuracy: 68.40%
  Test Loss: 0.9234
  Total Epochs: 10
```

## File Structure

```
adaptive-controller/
├── adaptive_controller.py     # Core controller logic
├── checkpoint_store.py         # Persistent storage utility
├── ray_pipeline.py             # Simulated pipeline (demo)
├── real_ml_pipeline.py         # PyTorch MNIST pipeline (stateless tasks)
├── cifar10_pipeline.py         # TensorFlow CIFAR-10 (stateless + stateful)
├── run_ml_demo.py                      # Quick MNIST demo
├── run_cifar10_demo.py                 # Quick CIFAR-10 demo
├── RAY_EXECUTION_GUIDE.md              # Stateless vs Stateful execution explained
├── ADAPTIVE_CHECKPOINTING_EXPLAINED.md # Deep dive into adaptive learning
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── data/                       # MNIST/CIFAR-10 datasets (auto-downloaded)
└── ray_ckpts/                  # Generated checkpoints and logs
    ├── *.pkl                   # Checkpoint files (includes model weights)
    └── adaptive_decisions.json # Decision metadata
```

## Use Cases

- **Long-running ML pipelines**: Checkpoint expensive training epochs with actual model weights
- **Distributed model training**: Recover from node failures without restarting from scratch
- **ETL workflows**: Protect against data processing failures
- **Multi-stage analytics**: Recover from intermediate stage failures
- **Resource-constrained environments**: Optimize checkpoint overhead vs. recovery benefit

## What Gets Checkpointed?

### In Simulated Pipeline
- Task metadata and fake metrics (small dictionaries)
- Random data transformations
- Simulated training state

### In PyTorch MNIST Pipeline (Stateless Tasks)
- **Complete model state**: All neural network weights (~500KB for SimpleNN)
- **Optimizer state**: Adam optimizer momentum buffers
- **Training metrics**: Real loss and accuracy values
- **Epoch information**: Current training progress

### In TensorFlow CIFAR-10 Pipeline (Stateful Actors)
- **Complete actor state**: All persistent variables in CIFAR10Trainer
- **Model weights**: CNN with 664K parameters (~2.5MB)
- **Training history**: Metrics from all completed epochs
- **Current epoch**: Which iteration the training is on
- **Optimizer state**: Maintained within the actor

**Key Difference**:
- **Stateless tasks** checkpoint serialized data passed between tasks
- **Stateful actors** checkpoint the entire actor state including in-memory model

When recovery happens:
1. **Stateless**: Deserialize checkpoint data and pass to new task
2. **Stateful**: Restore checkpoint state to the same actor instance
3. Controller learns whether checkpointing was beneficial and adjusts future decisions

## Performance Considerations

### Simulated Pipeline
- Checkpoint overhead: ~50-200ms for typical data structures
- Storage: ~2-10KB per checkpoint
- Decision overhead: <1ms per task

### PyTorch MNIST Pipeline (Stateless)
- Checkpoint overhead: ~100-500ms for model serialization
- Storage: ~500KB per checkpoint (110K parameters)
- Decision overhead: <1ms per task
- Recommended for training tasks >1s runtime
- Recovery: ~100ms to restore model state vs. minutes to retrain

### TensorFlow CIFAR-10 Pipeline (Stateful Actors)
- Checkpoint overhead: ~200-1000ms for actor state serialization
- Storage: ~2.5MB per checkpoint (664K parameters)
- Decision overhead: <1ms per task
- Actor overhead: Dedicated worker reserved for actor lifetime
- Recommended for iterative training >10s per epoch
- Recovery: ~200ms to restore actor state vs. 10-20 minutes to retrain
- **Advantage**: Model stays in memory between epochs (no reload overhead)

## Ray Execution Patterns Explained

### Stateless Tasks vs Stateful Actors

This repository demonstrates both Ray execution patterns:

| Pattern | Use Case | Example in Repo | Key Benefit |
|---------|----------|-----------------|-------------|
| **Stateless Tasks** | Data loading, preprocessing, one-shot computations | `load_cifar10_data()`, `create_data_subset()` | Can run on any worker, easy parallelism |
| **Stateful Actors** | Iterative training, maintaining state | `CIFAR10Trainer` class | Model stays in memory between calls |

**Read `RAY_EXECUTION_GUIDE.md` for:**
- Detailed execution diagrams
- When to use each pattern
- How Ray schedules tasks vs actors
- Visual comparison with examples
- Complete resource timeline

## Advanced Usage

### Using with Your Own PyTorch Model

Replace `SimpleNN` with your own model:

```python
@ray.remote
def train_epoch_task(model_bytes: bytes, epoch: int):
    # Load your model
    model = YourCustomModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Deserialize checkpoint
    checkpoint = torch.load(io.BytesIO(model_bytes))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Your training loop
    for data, target in train_loader:
        # ... training code ...
        pass

    # Serialize updated model
    model_buffer = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy
    }, model_buffer)

    return {"model_bytes": model_buffer.getvalue(), ...}
```

### Using with Your Own TensorFlow Model (Stateful Actor)

```python
@ray.remote
class YourCustomTrainer:
    def __init__(self):
        # Build your custom model
        self.model = self.build_your_model()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.epoch = 0

    def build_your_model(self):
        # Your custom architecture
        input_layer = layers.Input((your_input_shape))
        # ... your layers ...
        return models.Model(input_layer, output_layer)

    def train_epoch(self, x_train, y_train):
        # Model is already in memory (actor state)
        history = self.model.fit(x_train, y_train, epochs=1)
        self.epoch += 1
        return {"epoch": self.epoch, "loss": history.history['loss'][0]}

    def get_training_state(self):
        # Serialize complete actor state
        weights_bytes = ...  # Serialize model weights
        return {
            "epoch": self.epoch,
            "model_weights": weights_bytes
        }

    def restore_training_state(self, state):
        # Restore actor state from checkpoint
        self.epoch = state["epoch"]
        # Load weights back into self.model
        ...

# Use the actor
trainer = YourCustomTrainer.remote()
for epoch in range(num_epochs):
    result = ray.get(trainer.train_epoch.remote(x, y))
```

### Custom Recovery Metrics

You can provide custom metrics for more accurate adaptive learning:

```python
# Measure actual recovery time
recovery_start = time.time()
result = store.load("checkpoint")
actual_recovery_time = time.time() - recovery_start

# Estimate time saved (what would have been recomputed)
estimated_recompute_time = sum_of_previous_task_runtimes

controller.adjust_policy(
    recovery_time_saved=estimated_recompute_time,
    checkpoint_overhead=actual_recovery_time
)
```

### Multiple Pipeline Runs

The controller learns across failures within a single run. For persistent learning across multiple runs, save and restore the threshold:

```python
# After first run
final_threshold = controller.cost_threshold

# Before second run
controller = AdaptiveCheckpointController(
    cost_threshold=final_threshold  # Use learned value
)
```

## Troubleshooting

**Issue**: Controller never checkpoints
- Check if `cost_threshold` is too high
- Verify tasks are being observed with `observe_task()`
- Ensure budget is not already exhausted

**Issue**: Too many checkpoints
- Increase `cost_threshold` (e.g., 0.5 or 0.6)
- Reduce `budget_checkpoints`
- Lower `learning_rate` for slower adaptation

**Issue**: Checkpoint overhead too high
- Use smaller intermediate results
- Reduce checkpoint frequency by raising threshold
- Consider compression in CheckpointStore

## Pipeline Details

### PyTorch MNIST Pipeline (Stateless)

**Model Architecture:**
- Input: 28x28 MNIST images (784 features)
- Hidden Layer 1: 128 neurons with ReLU
- Hidden Layer 2: 64 neurons with ReLU
- Output: 10 classes (digits 0-9)
- Total Parameters: 109,386

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch Size: 64
- Dataset: MNIST (1000 training samples)

**Checkpoint Contents:**
```python
{
    'model_state_dict': OrderedDict(...),    # All layer weights
    'optimizer_state_dict': {...},            # Adam state
    'epoch': 0,
    'loss': 0.4532,
    'accuracy': 87.30
}
```

### TensorFlow CIFAR-10 Pipeline (Stateful)

**Model Architecture (Your Specification):**
- Input: 32x32x3 CIFAR-10 images
- Conv2D: 32 filters → BatchNorm → LeakyReLU
- Conv2D: 32 filters, stride=2 → BatchNorm → LeakyReLU
- Conv2D: 64 filters → BatchNorm → LeakyReLU
- Conv2D: 64 filters, stride=2 → BatchNorm → LeakyReLU
- Flatten → Dense(128) → BatchNorm → LeakyReLU → Dropout(0.5)
- Dense(10, softmax)
- Total Parameters: 664,266

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical Crossentropy
- Batch Size: 64
- Dataset: CIFAR-10 (5000 training, 1000 test samples)

**Actor State Checkpoint:**
```python
{
    "epoch": 3,
    "history": [epoch_0_metrics, epoch_1_metrics, ...],
    "model_weights": <numpy arrays of 664K parameters>,
    "learning_rate": 0.001
}
```

### Recovery Process

**Stateless (PyTorch MNIST):**
1. Task fails during epoch N
2. Controller searches for most recent checkpoint
3. Loads serialized model state
4. Launches new task with checkpoint data
5. Adjusts policy based on benefit

**Stateful (TensorFlow CIFAR-10):**
1. Actor method fails during epoch N
2. Controller searches for most recent checkpoint
3. Calls `actor.restore_training_state.remote(state)`
4. Actor updates its in-memory state (self.model, self.epoch, etc.)
5. Continue calling methods on the same actor
6. Calculates benefit = (epochs_saved × time_per_epoch) - checkpoint_overhead
7. Adjusts threshold: if benefit > 0, checkpoint more frequently next time

## Future Enhancements

- [ ] Distributed checkpoint storage (S3, HDFS)
- [ ] Compression for large model checkpoints
- [ ] Multi-run learning persistence
- [ ] Checkpoint prioritization based on downstream dependencies
- [ ] Integration with Ray Train and Ray's native checkpoint API
- [ ] Visualization dashboard for decision analysis
- [ ] Support for larger models (ResNet, Transformers)
- [ ] Gradient checkpointing for memory optimization

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
