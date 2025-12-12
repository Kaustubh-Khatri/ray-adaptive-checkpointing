# Ray Execution Patterns: Stateless vs Stateful

This guide explains how Ray handles tasks and actors in distributed ML training pipelines, and how the adaptive checkpoint controller enhances fault tolerance.

**Pipelines Demonstrated:**
- **TensorFlow CIFAR-10** (`cifar10_pipeline.py`) - Shows both stateless tasks AND stateful actors
- **PyTorch MNIST** (`real_ml_pipeline.py`) - Shows stateless tasks only

## Table of Contents
1. [Ray Stateless Tasks](#ray-stateless-tasks)
2. [Ray Stateful Actors](#ray-stateful-actors)
3. [Comparison: Stateless vs Stateful](#comparison-stateless-vs-stateful)
4. [Adaptive Controller Integration](#adaptive-controller-integration)
5. [Complete Execution Flow](#complete-execution-flow)
6. [Real-World Examples](#real-world-examples)

---

## Ray Stateless Tasks

### What Are Stateless Tasks?

Stateless tasks are **functions** decorated with `@ray.remote`. They:
- Execute independently without maintaining state
- Can run on any available worker node
- Return results to Ray's object store
- Complete and release resources when done

### Example from CIFAR-10 Pipeline

```python
@ray.remote
def load_cifar10_data() -> Dict[str, Any]:
    """STATELESS: Loads data, returns result, then terminates."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # ... preprocessing ...
    return {"x_train": x_train, "y_train": y_train, ...}
```

### How Ray Executes Stateless Tasks

```
┌─────────────────────────────────────────────────────────────┐
│ Driver Program                                              │
│                                                             │
│  data_ref = load_cifar10_data.remote()  ──────┐           │
│                                                │           │
└────────────────────────────────────────────────┼───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Ray Scheduler                                               │
│  • Finds available worker                                   │
│  • Schedules task                                           │
└────────────────────────────────────────────────┬───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Worker Node 1                                               │
│  • Receives task                                            │
│  • Executes load_cifar10_data()                            │
│  • Stores result in object store                           │
│  • Releases resources                                       │
└────────────────────────────────────────────────┬───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Ray Object Store                                            │
│  ObjectRef(data_id) → {x_train, y_train, x_test, y_test}  │
└────────────────────────────────────────────────┬───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Driver Program                                              │
│  data = ray.get(data_ref)  ← Retrieves from object store   │
└─────────────────────────────────────────────────────────────┘
```

### Characteristics of Stateless Tasks

| Aspect | Behavior |
|--------|----------|
| **State** | No persistent state between calls |
| **Execution** | One task = one function call = one completion |
| **Resource Usage** | Allocated when task starts, released when done |
| **Parallelism** | Multiple instances can run simultaneously |
| **Fault Tolerance** | Failed tasks can be restarted on different workers |
| **Use Case** | Data loading, preprocessing, one-time computations |

### Multiple Stateless Tasks in Parallel

```python
# Launch two stateless tasks in parallel
train_ref = create_data_subset.remote(x_train, y_train, 5000)
test_ref = create_data_subset.remote(x_test, y_test, 1000)

# Both execute simultaneously on different workers
train_subset, test_subset = ray.get([train_ref, test_ref])
```

```
Worker 1: create_data_subset(train data) ──┐
                                            ├──→ Both run in parallel
Worker 2: create_data_subset(test data)  ──┘
```

---

## Ray Stateful Actors

### What Are Stateful Actors?

Stateful actors are **classes** decorated with `@ray.remote`. They:
- Maintain persistent state across method calls
- Run on a dedicated worker node for their lifetime
- Keep data in memory (e.g., model weights, optimizer state)
- Process method calls sequentially on the same instance

### Example from CIFAR-10 Pipeline

```python
@ray.remote
class CIFAR10Trainer:
    """STATEFUL: Maintains model and training state."""

    def __init__(self, learning_rate: float = 0.001):
        self.model = self._build_model()  # Persistent state
        self.current_epoch = 0             # Persistent state
        self.history = []                  # Persistent state

    def train_epoch(self, x_train, y_train):
        """Updates persistent model state."""
        self.model.fit(x_train, y_train, epochs=1)
        self.current_epoch += 1  # State updated
        return {"epoch": self.current_epoch}
```

### How Ray Executes Stateful Actors

```
┌─────────────────────────────────────────────────────────────┐
│ Driver Program                                              │
│                                                             │
│  trainer = CIFAR10Trainer.remote(lr=0.001)  ──────┐       │
│                                                    │       │
└────────────────────────────────────────────────────┼───────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Ray Scheduler                                               │
│  • Finds available worker                                   │
│  • Creates CIFAR10Trainer actor instance                    │
│  • Reserves worker for this actor's lifetime                │
└────────────────────────────────────────────────┬───────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Worker Node 2 (Dedicated to this actor)                    │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │ CIFAR10Trainer Instance              │                  │
│  │                                      │                  │
│  │  self.model = <TF Model>  ←─────────┼─ Persists here  │
│  │  self.current_epoch = 0              │                  │
│  │  self.history = []                   │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Waits for method calls...                                 │
└─────────────────────────────────────────────────────────────┘
```

### Calling Actor Methods

```
┌─────────────────────────────────────────────────────────────┐
│ Driver Program                                              │
│                                                             │
│  result_ref = trainer.train_epoch.remote(x, y)  ──────┐   │
│                                                        │   │
└────────────────────────────────────────────────────────┼───┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Worker Node 2 (Actor's dedicated worker)                   │
│                                                             │
│  ┌──────────────────────────────────────┐                  │
│  │ CIFAR10Trainer Instance              │                  │
│  │                                      │                  │
│  │  self.model.fit(x, y, epochs=1)      │ ← Uses state    │
│  │  self.current_epoch += 1             │ ← Updates state │
│  │  self.history.append(result)         │ ← Updates state │
│  │                                      │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Returns result to object store                            │
└─────────────────────────────────────────────────────────────┘
```

### Characteristics of Stateful Actors

| Aspect | Behavior |
|--------|----------|
| **State** | Persistent state maintained in memory |
| **Execution** | Sequential method calls on same instance |
| **Resource Usage** | Reserved for actor's lifetime |
| **Parallelism** | One actor = one worker (but can have multiple actors) |
| **Fault Tolerance** | If actor crashes, state is lost (unless checkpointed) |
| **Use Case** | Iterative training, maintaining model state, servers |

### Sequential Method Calls on Actor

```python
# All methods execute on the SAME actor instance
for epoch in range(10):
    result_ref = trainer.train_epoch.remote(x, y)
    # Each epoch builds on previous epoch's model weights
```

```
Epoch 0: model weights = W0 ──┐
                               ├─→ trainer.train_epoch() ─→ W1
Epoch 1: model weights = W1 ──┘                           ──┐
                                                            ├─→ W2
Epoch 2: model weights = W2 ────────────────────────────────┘
                                  (same actor, same memory)
```

---

## Comparison: Stateless vs Stateful

### Side-by-Side Comparison

| Feature | Stateless Tasks | Stateful Actors |
|---------|----------------|-----------------|
| **Decoration** | `@ray.remote` on function | `@ray.remote` on class |
| **Invocation** | `func.remote()` | `Actor.remote()` then `actor.method.remote()` |
| **State** | No state between calls | State persists across calls |
| **Memory** | Released after completion | Kept in memory for lifetime |
| **Worker** | Any available worker | Dedicated worker |
| **Parallelism** | Many tasks run in parallel | Methods run sequentially per actor |
| **Use Case** | ETL, preprocessing, stateless ops | Training, caching, stateful services |

### Visual Comparison: Training Without vs With Actors

#### Without Actors (Stateless - Inefficient for Training)

```python
@ray.remote
def train_one_epoch(model_weights, x, y):
    """Every call needs to reload model - SLOW!"""
    model = build_model()
    model.set_weights(model_weights)  # Reload every time
    model.fit(x, y, epochs=1)
    return model.get_weights()

# Each epoch pays deserialization cost
for epoch in range(10):
    weights = ray.get(train_one_epoch.remote(weights, x, y))
```

```
Task 1: deserialize → train → serialize → ✓
Task 2: deserialize → train → serialize → ✓  (Unnecessary overhead!)
Task 3: deserialize → train → serialize → ✓
```

#### With Actors (Stateful - Efficient)

```python
@ray.remote
class Trainer:
    def __init__(self):
        self.model = build_model()  # Once!

    def train_epoch(self, x, y):
        self.model.fit(x, y, epochs=1)  # Uses in-memory model
        return {"epoch": self.epoch}

trainer = Trainer.remote()
for epoch in range(10):
    ray.get(trainer.train_epoch.remote(x, y))
```

```
Actor initialization: build_model() → Keep in memory
Epoch 0: train (in-memory model) → ✓
Epoch 1: train (in-memory model) → ✓  (No deserialization!)
Epoch 2: train (in-memory model) → ✓
```

---

## Adaptive Controller Integration

### How the Controller Works with Ray

The adaptive checkpoint controller observes task/actor execution and makes intelligent checkpoint decisions.

```
┌─────────────────────────────────────────────────────────────┐
│ Adaptive Checkpoint Controller                              │
│                                                             │
│  1. observe_task(name, runtime)                             │
│     ↓                                                       │
│  2. should_checkpoint(name, estimated_cost)                 │
│     ↓                                                       │
│  3. Decision: YES/NO based on:                             │
│     • Runtime ratio vs max observed                         │
│     • Budget remaining                                      │
│     • Current threshold                                     │
└─────────────────────────────────────────────────────────────┘
```

### Integration with Stateless Tasks

```python
@ray.remote
def load_cifar10_data():
    # ... load data ...
    return data

# In pipeline
t_start = time.time()
data = ray.get(load_cifar10_data.remote())
elapsed = time.time() - t_start

# Controller observes and decides
controller.observe_task("data_loading", elapsed)
should_ckpt = controller.should_checkpoint("data_loading", elapsed)

if should_ckpt:
    STORE.save("data_ckpt", data)  # Save to persistent storage
```

### Integration with Stateful Actors

```python
@ray.remote
class CIFAR10Trainer:
    def get_training_state(self):
        """Returns complete state for checkpointing."""
        return {
            "epoch": self.current_epoch,
            "history": self.history,
            "model_weights": self.model.get_weights()
        }

    def restore_training_state(self, state):
        """Restores from checkpoint."""
        self.current_epoch = state["epoch"]
        self.history = state["history"]
        self.model.set_weights(state["model_weights"])

# In pipeline
for epoch in range(10):
    t_start = time.time()
    result = ray.get(trainer.train_epoch.remote(x, y))
    elapsed = time.time() - t_start

    controller.observe_task(f"train_epoch_{epoch}", elapsed)
    should_ckpt = controller.should_checkpoint(f"train_epoch_{epoch}", elapsed)

    if should_ckpt:
        # Get state from actor
        state = ray.get(trainer.get_training_state.remote())
        STORE.save(f"epoch_{epoch}_ckpt", state)
```

### Recovery Process

```
┌─────────────────────────────────────────────────────────────┐
│ Normal Execution                                            │
│                                                             │
│  Epoch 0: Train ──→ Checkpoint ✓                           │
│  Epoch 1: Train ──→ Checkpoint ✓                           │
│  Epoch 2: Train ──→ FAILURE! ✗                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Recovery                                                    │
│                                                             │
│  1. Search for most recent checkpoint                       │
│  2. Found: epoch_1_ckpt                                     │
│  3. Load checkpoint state                                   │
│  4. Restore state to actor:                                 │
│     trainer.restore_training_state.remote(state)            │
│  5. Continue from epoch 1                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Adaptive Learning                                           │
│                                                             │
│  benefit = (epochs_saved × time_per_epoch) - ckpt_overhead  │
│           = (1 × 15.0) - 1.0 = 14.0 seconds saved          │
│                                                             │
│  Controller adjusts threshold:                              │
│    • Positive benefit → Lower threshold                     │
│    • Checkpoint more frequently next time                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Execution Flow

### CIFAR-10 Pipeline with All Components

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1: Data Loading (STATELESS)                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Driver: load_cifar10_data.remote()                             │
│      ↓                                                           │
│  Worker: Execute task → Return data                             │
│      ↓                                                           │
│  Controller: Observe runtime → Should checkpoint?               │
│      ↓                                                           │
│  Decision: NO (fast task, ~3s)                                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Stage 2: Create Subsets (STATELESS, PARALLEL)                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Driver: create_data_subset.remote(train) ──┐                   │
│         create_data_subset.remote(test)  ───┤                   │
│                                              │                   │
│  Worker 1: Process train subset              │                  │
│  Worker 2: Process test subset               ├─→ Parallel       │
│                                              │                   │
│  Both complete → Results to object store ────┘                  │
│      ↓                                                           │
│  Controller: Observe runtime → Should checkpoint?               │
│  Decision: YES (valuable processed data)                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Stage 3: Initialize Trainer (STATEFUL ACTOR)                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Driver: CIFAR10Trainer.remote()                                │
│      ↓                                                           │
│  Ray Scheduler: Allocate dedicated worker                       │
│      ↓                                                           │
│  Worker 3: Create actor instance                                │
│           • Build model (664K parameters)                        │
│           • Initialize optimizer                                 │
│           • State kept in memory                                 │
│      ↓                                                           │
│  Actor ready and waiting for method calls                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Stage 4: Training Loop (STATEFUL ACTOR METHODS)                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each epoch:                                                 │
│                                                                  │
│    Driver: trainer.train_epoch.remote(x, y)                     │
│        ↓                                                         │
│    Worker 3 (Actor): Execute on same instance                   │
│                      • Use in-memory model                       │
│                      • Update weights                            │
│                      • Increment epoch counter                   │
│        ↓                                                         │
│    Controller: Observe runtime → Should checkpoint?             │
│        ↓                                                         │
│    IF YES:                                                       │
│      Driver: state = trainer.get_training_state.remote()        │
│      STORE.save("epoch_N_ckpt", state)                          │
│                                                                  │
│    IF FAILURE:                                                   │
│      Load most recent checkpoint                                │
│      Driver: trainer.restore_training_state.remote(state)       │
│      Controller: adjust_policy(benefit)                         │
│      Continue training                                           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Stage 5: Evaluation (STATEFUL ACTOR METHOD)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Driver: trainer.evaluate.remote(x_test, y_test)                │
│      ↓                                                           │
│  Worker 3 (Actor): Evaluate using trained model                 │
│                    • Model has weights from all epochs           │
│                    • Compute test accuracy                       │
│      ↓                                                           │
│  Return: {"test_accuracy": 0.72, "test_loss": 0.85}            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Resource Timeline

```
Time  │ Worker 1        │ Worker 2        │ Worker 3 (Actor)
──────┼─────────────────┼─────────────────┼──────────────────────
  0s  │ load_data()     │ (idle)          │ (idle)
  3s  │ (done)          │ subset(train)   │ CIFAR10Trainer()
  5s  │ (idle)          │ subset(test)    │ (initializing...)
  6s  │ (idle)          │ (done)          │ [Ready, waiting]
  7s  │ (idle)          │ (idle)          │ train_epoch(0)
 22s  │ (idle)          │ (idle)          │ [checkpoint saved]
 23s  │ (idle)          │ (idle)          │ train_epoch(1)
 38s  │ (idle)          │ (idle)          │ [checkpoint saved]
...   │ ...             │ ...             │ ...
150s  │ (idle)          │ (idle)          │ evaluate()
155s  │ (done)          │ (done)          │ (done)
```

**Key Observations:**
- Worker 3 is **dedicated** to the actor for the entire training process
- Workers 1 and 2 complete stateless tasks and become idle
- Actor maintains model state in memory throughout training
- Checkpoints save complete actor state for recovery

---

## Summary

### When to Use Stateless Tasks
✅ Data loading
✅ Preprocessing
✅ One-time computations
✅ Embarrassingly parallel operations
✅ Tasks that don't build on previous results

### When to Use Stateful Actors
✅ Iterative training (each epoch builds on previous)
✅ Maintaining model state
✅ Caching expensive computations
✅ Serving models
✅ Stateful services

### How Adaptive Controller Helps
✅ Observes task/actor execution times
✅ Decides optimal checkpoint placement
✅ Learns from recovery events
✅ Balances checkpoint overhead vs recovery benefit
✅ Saves model state for fault tolerance

### The Power of Ray + Adaptive Checkpointing
1. **Ray handles distribution**: Tasks run on any worker, actors on dedicated workers
2. **Controller handles intelligence**: Learns when to checkpoint based on cost/benefit
3. **Together**: Fault-tolerant, efficient distributed training with minimal manual tuning

---

## Real-World Examples

### Example 1: PyTorch MNIST (Stateless Only)

**File**: `real_ml_pipeline.py`

This pipeline uses **only stateless tasks** for everything:

```python
# Stateless task 1: Load data
@ray.remote
def load_data_task():
    return {"x_train": ..., "y_train": ...}

# Stateless task 2: Initialize model
@ray.remote
def initialize_model_task():
    model = SimpleNN()
    return {"model_bytes": serialize(model)}

# Stateless task 3: Train one epoch
@ray.remote
def train_epoch_task(model_bytes, x, y):
    model = deserialize(model_bytes)  # Reload model
    model.fit(x, y)
    return {"model_bytes": serialize(model)}  # Return updated model

# Pipeline execution
data = ray.get(load_data_task.remote())
model_state = ray.get(initialize_model_task.remote())

for epoch in range(10):
    model_state = ray.get(
        train_epoch_task.remote(model_state["model_bytes"], x, y)
    )
```

**Characteristics:**
- ✅ Simple: Each task is independent
- ✅ Flexible: Tasks can run on different workers
- ❌ Overhead: Model must be serialized/deserialized each epoch
- ❌ Slower: ~100-200ms extra per epoch for 110K parameters

**When to use this pattern:**
- Simpler workflows with few iterations
- When worker availability varies
- When model is small (<1MB)

---

### Example 2: TensorFlow CIFAR-10 (Stateless + Stateful)

**File**: `cifar10_pipeline.py`

This pipeline uses **both patterns optimally**:

**Stateless for one-time operations:**
```python
# Stateless: Data loading (run once)
@ray.remote
def load_cifar10_data():
    return {"x_train": ..., "y_train": ...}

# Stateless: Data preprocessing (run once)
@ray.remote
def create_data_subset(x, y, size):
    return {"x_subset": ..., "y_subset": ...}
```

**Stateful for iterative training:**
```python
# Stateful: Training (multiple epochs on same instance)
@ray.remote
class CIFAR10Trainer:
    def __init__(self):
        self.model = build_model()  # Build once, keep in memory
        self.epoch = 0

    def train_epoch(self, x, y):
        # Model already in memory - no reload needed!
        self.model.fit(x, y, epochs=1)
        self.epoch += 1
        return {"epoch": self.epoch}

# Pipeline execution
data = ray.get(load_cifar10_data.remote())  # Stateless
subset = ray.get(create_data_subset.remote(data))  # Stateless

trainer = CIFAR10Trainer.remote()  # Stateful actor
for epoch in range(10):
    result = ray.get(trainer.train_epoch.remote(x, y))  # Same actor
```

**Characteristics:**
- ✅ Efficient: Model stays in memory (no reload overhead)
- ✅ Fast: ~0ms overhead between epochs
- ✅ Optimal: Uses stateless for one-shots, stateful for iterations
- ✅ Scalable: Can create multiple actors for parallel training

**When to use this pattern:**
- Iterative training (>5 epochs)
- Large models (>1MB)
- When you want maximum efficiency

---

### Comparison: Same Training Job

**Task**: Train 664K parameter CNN for 10 epochs on CIFAR-10

| Approach | Model Reload Overhead | Total Extra Time | Code Complexity |
|----------|----------------------|------------------|-----------------|
| **All Stateless** (like PyTorch MNIST) | 200ms × 10 epochs = 2s | ~2 seconds | Simple |
| **Stateful Actor** (TensorFlow CIFAR-10) | 0ms (stays in memory) | ~0 seconds | Slightly more complex |

**Winner**: Stateful actors for iterative training (saves 2s on small model, 10s+ on large models)

---

### Example 3: Hybrid Pattern for Distributed Training

You can combine both patterns for advanced workflows:

```python
# Stateless: Distribute data loading across workers
@ray.remote
def load_shard(shard_id):
    return load_data_shard(shard_id)

# Launch multiple parallel data loading tasks
shard_refs = [load_shard.remote(i) for i in range(10)]
shards = ray.get(shard_refs)  # All load in parallel

# Stateful: Create multiple trainer actors for data parallelism
@ray.remote
class Trainer:
    def __init__(self, model_config):
        self.model = build_model(model_config)

    def train_on_shard(self, shard_data):
        self.model.fit(shard_data)
        return self.model.get_weights()

# Create 4 trainer actors (one per GPU)
trainers = [Trainer.remote(config) for _ in range(4)]

# Train each actor on different shards in parallel
results = ray.get([
    trainers[i].train_on_shard.remote(shards[i])
    for i in range(4)
])

# Aggregate results (stateless task)
@ray.remote
def aggregate_weights(weight_list):
    return average(weight_list)

final_weights = ray.get(aggregate_weights.remote(results))
```

**This pattern combines:**
- Stateless parallelism for data loading
- Stateful actors for maintaining model state
- Stateless aggregation for combining results

---

### Decision Flowchart

```
┌─────────────────────────────────────┐
│ Need to maintain state across calls?│
└──────────┬────────────┬─────────────┘
           │            │
          YES          NO
           │            │
           ▼            ▼
    ┌──────────┐  ┌─────────┐
    │ STATEFUL │  │STATELESS│
    │  ACTOR   │  │  TASK   │
    └──────────┘  └─────────┘
           │            │
           ▼            ▼
    Examples:      Examples:
    • Training     • Data loading
    • Caching      • Preprocessing
    • Serving      • Aggregation
    • Iterative    • One-shot ops
      computation
```

---

### Performance Guidelines

**Use Stateless Tasks when:**
- Operation runs once or rarely
- Model/data is small (<1MB)
- Need maximum flexibility in worker assignment
- Operations are embarrassingly parallel

**Use Stateful Actors when:**
- Iterative operations (training epochs, game simulations)
- Large models (>1MB) that are expensive to serialize
- Need to maintain state (model weights, cache, counters)
- Sequential operations on the same data

**Use Both when:**
- Pipeline has distinct phases (load → train → aggregate)
- Want to combine parallel data loading with stateful training
- Need both flexibility and efficiency

---

### Common Pitfalls

#### Pitfall 1: Using Stateless for Iterative Training

```python
# ❌ BAD: Stateless for training loop
for epoch in range(100):
    model = ray.get(train_epoch.remote(serialize(model), data))
    # Serialization overhead 100 times!
```

```python
# ✅ GOOD: Stateful actor for training loop
trainer = Trainer.remote()
for epoch in range(100):
    ray.get(trainer.train_epoch.remote(data))
    # No serialization overhead!
```

#### Pitfall 2: Using Stateful for One-Time Operations

```python
# ❌ BAD: Actor for one-time data load
@ray.remote
class DataLoader:
    def load(self):
        return load_data()

loader = DataLoader.remote()
data = ray.get(loader.load.remote())  # Unnecessary actor overhead
```

```python
# ✅ GOOD: Stateless task for one-time operation
@ray.remote
def load_data():
    return load_data()

data = ray.get(load_data.remote())  # Simple and efficient
```

#### Pitfall 3: Forgetting to Checkpoint Actor State

```python
# ❌ BAD: Actor with no checkpointing
trainer = Trainer.remote()
for epoch in range(100):
    trainer.train_epoch.remote(data)
    # If actor crashes at epoch 99, lose everything!
```

```python
# ✅ GOOD: Checkpoint actor state with adaptive controller
trainer = Trainer.remote()
for epoch in range(100):
    trainer.train_epoch.remote(data)

    if controller.should_checkpoint(f"epoch_{epoch}", elapsed):
        state = ray.get(trainer.get_state.remote())
        STORE.save(f"epoch_{epoch}_ckpt", state)
```

---

## Conclusion

**Key Takeaways:**

1. **Stateless tasks** = Functions that complete and release resources
2. **Stateful actors** = Objects that persist and maintain state
3. **Adaptive checkpointing** = Intelligence layer that learns optimal checkpoint placement

**Best Practice:**
- Use stateless for ETL and one-time operations
- Use stateful for iterative training and stateful services
- Use adaptive checkpointing for fault tolerance
- Combine all three for production ML pipelines

**See the code:**
- `cifar10_pipeline.py` - Complete example with both patterns
- `real_ml_pipeline.py` - Stateless-only example
- `adaptive_controller.py` - Checkpoint intelligence
