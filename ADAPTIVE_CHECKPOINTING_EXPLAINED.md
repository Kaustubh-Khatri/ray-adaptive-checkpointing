# How Adaptive Checkpointing Works

This document explains in detail how the adaptive checkpoint controller learns and adapts its checkpointing strategy based on runtime observations and recovery events.

## Table of Contents
1. [Overview](#overview)
2. [The Adaptation Mechanism](#the-adaptation-mechanism)
3. [Step-by-Step Example](#step-by-step-example)
4. [Learning from Failures](#learning-from-failures)
5. [Real-World Scenarios](#real-world-scenarios)
6. [Mathematical Foundation](#mathematical-foundation)

---

## Overview

### What Makes It "Adaptive"?

Traditional checkpointing systems use **fixed rules**:
- ❌ "Checkpoint every 10 minutes"
- ❌ "Checkpoint every 5 epochs"
- ❌ "Checkpoint when 50% complete"

**Adaptive checkpointing** uses **dynamic learning**:
- ✅ Observes actual task runtimes
- ✅ Adjusts checkpoint frequency based on cost/benefit
- ✅ Learns from recovery events
- ✅ Balances checkpoint overhead vs recovery savings

### The Core Idea

```
Traditional:  Fixed Schedule → Checkpoint
                    ↓
              Wasteful or Insufficient

Adaptive:     Observe Runtime → Predict Cost → Decide → Learn from Recovery
                    ↓                               ↓
              Optimal Checkpointing          Better Decisions Next Time
```

---

## The Adaptation Mechanism

### Three Phases of Adaptation

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: OBSERVATION                                         │
│ Track runtime of each task as it completes                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 2: DECISION                                            │
│ Compare task runtime to threshold and budget                │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Phase 3: LEARNING                                            │
│ Adjust threshold based on recovery outcomes                 │
└──────────────────────────────────────────────────────────────┘
```

### Phase 1: Observation

**What happens:**
```python
controller.observe_task("train_epoch_0", elapsed=15.3)
```

**Internal state updates:**
```python
self.max_time_seen = max(15.3, self.max_time_seen)  # Track longest task
self.history.append(("train_epoch_0", 15.3, None, False))
```

**Why it matters:**
- Builds understanding of task runtime patterns
- Identifies expensive vs cheap tasks
- Creates baseline for future comparisons

### Phase 2: Decision

**What happens:**
```python
should_ckpt = controller.should_checkpoint("train_epoch_0", estimated_cost=15.3)
```

**Decision algorithm:**
```python
# Step 1: Check if budget allows
if self.used >= self.budget:
    return False  # No more checkpoints allowed

# Step 2: Calculate runtime ratio
ratio = estimated_cost / max(self.max_time_seen, 1e-6)
# Example: 15.3 / 15.3 = 1.0

# Step 3: Compare to threshold
decision = ratio >= self.cost_threshold
# Example: 1.0 >= 0.35 → True (checkpoint!)

# Step 4: Update budget if checkpointing
if decision:
    self.used += 1  # Consumed one checkpoint slot
```

**Decision factors:**

| Factor | Initial Value | Adapts? | Purpose |
|--------|---------------|---------|---------|
| `max_time_seen` | 1e-6 | ✅ Yes | Track longest task runtime |
| `cost_threshold` | 0.35 | ✅ Yes | Minimum ratio to checkpoint |
| `budget` | 3 | ❌ No | Hard limit on checkpoint count |
| `used` | 0 | ✅ Yes | Tracks consumed budget |

### Phase 3: Learning (The Adaptive Part!)

**What happens:**
```python
controller.adjust_policy(recovery_time_saved=14.0, checkpoint_overhead=1.0)
```

**Adaptive algorithm:**
```python
# Step 1: Calculate net benefit
benefit = recovery_time_saved - checkpoint_overhead
# Example: 14.0 - 1.0 = 13.0 (positive = good!)

# Step 2: Determine adjustment direction
direction = -1 if benefit > 0 else 1
# Positive benefit → -1 → Lower threshold → Checkpoint MORE
# Negative benefit → +1 → Raise threshold → Checkpoint LESS

# Step 3: Calculate adjustment magnitude (dampened)
delta = direction * learning_rate * abs(benefit) / (abs(benefit) + 1)
# Example: -1 * 0.1 * 13.0 / 14.0 = -0.093

# Step 4: Update threshold (clamped to [0.05, 1.0])
self.cost_threshold = max(0.05, min(1.0, self.cost_threshold + delta))
# Example: 0.35 + (-0.093) = 0.257
```

**The adaptation formula:**
```
                    learning_rate × |benefit|
delta = direction × ─────────────────────────
                        |benefit| + 1

New threshold = clamp(old_threshold + delta, 0.05, 1.0)
```

**Why this formula?**
- **Dampening**: Division by `|benefit| + 1` prevents extreme adjustments
- **Proportional**: Larger benefits cause larger adjustments
- **Bounded**: Clamped to [0.05, 1.0] to prevent instability

---

## Step-by-Step Example

### Scenario: Training a CNN for 10 Epochs

**Initial state:**
```python
cost_threshold = 0.35
budget = 3
used = 0
max_time_seen = 0
```

### Epoch 0: First Checkpoint Decision

```python
# Task completes
elapsed = 15.3 seconds

# Observation
controller.observe_task("train_epoch_0", 15.3)
# Internal: max_time_seen = 15.3

# Decision
ratio = 15.3 / 15.3 = 1.0
decision = 1.0 >= 0.35  # True
# Checkpoint: YES (used = 1/3)
```

**Why checkpoint?**
- Ratio is 1.0 (100% of max observed)
- Far exceeds threshold of 0.35
- Budget available (1/3 used)

### Epoch 1: Second Checkpoint Decision

```python
# Task completes
elapsed = 14.8 seconds

# Observation
controller.observe_task("train_epoch_1", 14.8)
# Internal: max_time_seen = 15.3 (unchanged)

# Decision
ratio = 14.8 / 15.3 = 0.967
decision = 0.967 >= 0.35  # True
# Checkpoint: YES (used = 2/3)
```

**Why checkpoint?**
- Ratio is 0.967 (96.7% of max)
- Still well above threshold
- Budget available (2/3 used)

### Epoch 2: FAILURE Occurs!

```python
# Task fails during training
# Controller initiates recovery

# Find most recent checkpoint
checkpoint_found = "train_epoch_1_ckpt"  # From epoch 1

# Calculate recovery benefit
epochs_lost = 2 - 1 = 1 epoch
recovery_time_saved = 1 * 15.0 = 15.0 seconds  # Avoided retraining 1 epoch
checkpoint_overhead = 1.0 seconds  # Time spent saving checkpoint

# ADAPTIVE LEARNING
controller.adjust_policy(
    recovery_time_saved=15.0,
    checkpoint_overhead=1.0
)
```

**Learning calculation:**
```python
benefit = 15.0 - 1.0 = 14.0  # Strongly positive!
direction = -1  # Positive benefit → lower threshold
delta = -1 * 0.1 * 14.0 / 15.0 = -0.093

new_threshold = 0.35 + (-0.093) = 0.257
```

**Adaptation outcome:**
```
Old threshold: 0.35  →  New threshold: 0.257
                    (Lower = checkpoint more often)
```

**Why lower the threshold?**
- Checkpointing saved 14 seconds of work
- Benefit far outweighed overhead (14x)
- Controller learns: "Checkpoints are valuable here, do it more!"

### Epoch 3: Decision with New Threshold

```python
# Task completes
elapsed = 14.5 seconds

# Observation
controller.observe_task("train_epoch_3", 14.5)

# Decision with NEW lower threshold
ratio = 14.5 / 15.3 = 0.948
decision = 0.948 >= 0.257  # True (would have been True before too)
# Checkpoint: YES (used = 3/3, budget exhausted)
```

### Epoch 4: Budget Exhausted

```python
# Task completes
elapsed = 14.9 seconds

# Decision
budget_available = (3 < 3)  # False
# Checkpoint: NO (budget exhausted, regardless of threshold)
```

**Key insight:**
- Even with lower threshold, budget constraint still applies
- This prevents unlimited checkpointing
- Forces strategic use of checkpoint slots

---

## Learning from Failures

### Scenario 1: Checkpoint Saved Time (Positive Benefit)

```
Timeline:
  Epoch 0: Train (15s) → Checkpoint (1s overhead)
  Epoch 1: Train (15s) → Checkpoint (1s overhead)
  Epoch 2: FAILURE at 10s mark
  Recovery: Load epoch 1 checkpoint (0.1s)
  Epoch 2: Resume training from epoch 1

Calculation:
  Time WITHOUT checkpoint:
    - Epoch 0: 15s
    - Epoch 1: 15s
    - Epoch 2 (failed): 10s
    - Restart from scratch: 15s + 15s + 15s = 45s
    Total: 55s

  Time WITH checkpoint:
    - Epoch 0: 15s + 1s (ckpt) = 16s
    - Epoch 1: 15s + 1s (ckpt) = 16s
    - Epoch 2 (failed): 10s
    - Recovery: 0.1s
    - Epoch 2 (resume): 15s
    Total: 57.1s

Wait, that's worse! Let's recalculate correctly...

Actually:
  recovery_time_saved = work that would need to be redone
                      = 15s (epoch 0) + 15s (epoch 1) = 30s
  checkpoint_overhead = 1s + 1s = 2s

  benefit = 30s - 2s = 28s saved

Adaptation:
  benefit = 28.0 (strongly positive)
  delta = -1 * 0.1 * 28/29 = -0.097
  new_threshold = 0.35 - 0.097 = 0.253 ✓
```

**Learning**: Checkpoints are very valuable, lower threshold to checkpoint more!

### Scenario 2: Checkpoint Wasted (Negative Benefit)

```
Timeline:
  Epoch 0: Train (0.5s) → Checkpoint (2s overhead - expensive!)
  Epoch 1: Train (0.5s) → No failure
  Complete successfully

Calculation:
  recovery_time_saved = 0 (no failure occurred)
  checkpoint_overhead = 2s

  benefit = 0 - 2s = -2s (negative = wasteful)

Adaptation:
  benefit = -2.0 (negative)
  direction = +1 (raise threshold)
  delta = +1 * 0.1 * 2/3 = +0.067
  new_threshold = 0.35 + 0.067 = 0.417 ✗
```

**Learning**: Checkpointing was wasteful, raise threshold to checkpoint less!

### Scenario 3: Marginal Benefit (Small Positive)

```
Calculation:
  recovery_time_saved = 1.5s
  checkpoint_overhead = 1.0s

  benefit = 1.5 - 1.0 = 0.5s (small positive)

Adaptation:
  benefit = 0.5
  delta = -1 * 0.1 * 0.5/1.5 = -0.033
  new_threshold = 0.35 - 0.033 = 0.317 (small adjustment)
```

**Learning**: Slight benefit, make small adjustment toward more checkpointing.

---

## Real-World Scenarios

### Scenario A: Frequent Failures, Long Tasks

**Environment:**
- Tasks take 300 seconds each
- Failures occur every ~3 tasks
- Checkpoint overhead: 5 seconds

**Initial behavior (threshold=0.35):**
```
Task 0: 300s → ratio=1.0 → Checkpoint ✓
Task 1: 300s → ratio=1.0 → Checkpoint ✓
Task 2: 300s → ratio=1.0 → Checkpoint ✓
Task 3: FAILURE → Recover from Task 2
```

**Learning:**
```python
benefit = 300s (saved task 3 recompute) - 5s (overhead) = 295s
delta = -1 * 0.1 * 295/296 = -0.0997
new_threshold = 0.35 - 0.0997 = 0.250
```

**Adapted behavior (threshold=0.250):**
- Lower threshold means checkpoint even earlier
- More aggressive checkpointing for failure-prone environment
- Minimizes recomputation on failures

### Scenario B: Rare Failures, Fast Tasks

**Environment:**
- Tasks take 2 seconds each
- Failures are extremely rare (1 in 1000)
- Checkpoint overhead: 1 second

**Initial behavior (threshold=0.35):**
```
Task 0: 2s → ratio=1.0 → Checkpoint ✓
Task 1: 2s → ratio=1.0 → Checkpoint ✓
...
Task 1000: Complete with no failures
```

**Problem:**
- Wasted 1000 checkpoints × 1s = 1000s overhead
- Zero recovery benefit (no failures)

**Learning (if measured):**
```python
benefit = 0s (no failures) - 1000s (overhead) = -1000s
delta = +1 * 0.1 * 1000/1001 = +0.0999
new_threshold = 0.35 + 0.0999 = 0.450
```

**Adapted behavior (threshold=0.450):**
- Higher threshold means checkpoint less frequently
- Reduces overhead for stable environment
- Balances risk vs overhead

### Scenario C: Variable Task Runtimes

**Environment:**
- Task runtimes vary: 5s, 10s, 100s, 8s, 95s...
- Budget: 3 checkpoints
- Need to identify expensive tasks

**Evolution:**

**Initial (threshold=0.35):**
```
Task A: 5s   → ratio=5/5=1.0   → Checkpoint ✓ (used 1/3)
Task B: 10s  → ratio=10/10=1.0 → Checkpoint ✓ (used 2/3)
Task C: 100s → ratio=100/100=1.0 → Checkpoint ✓ (used 3/3)
Task D: 8s   → ratio=8/100=0.08 → No checkpoint (budget exhausted anyway)
Task E: 95s  → ratio=95/100=0.95 → No checkpoint (budget exhausted)
```

**Problem**: Wasted checkpoints on fast tasks (A, B)

**After learning (threshold lowered to 0.30 from recovery):**
```
Task A: 5s   → ratio=0.05 → No checkpoint (too fast)
Task B: 10s  → ratio=0.10 → No checkpoint (still too fast)
Task C: 100s → ratio=1.0  → Checkpoint ✓ (used 1/3)
Task D: 8s   → ratio=0.08 → No checkpoint
Task E: 95s  → ratio=0.95 → Checkpoint ✓ (used 2/3) - expensive task!
```

**Improvement**: Saves budget for actually expensive tasks!

---

## Mathematical Foundation

### Cost-Benefit Ratio

The controller computes expected value of checkpointing:

```
Expected Value = P(failure) × time_saved - (1 - P(failure)) × overhead

Where:
  P(failure) = probability of failure (estimated from history)
  time_saved = runtime from last checkpoint to current point
  overhead = time to save checkpoint
```

**Simplification in current implementation:**

Since we don't track failure probability directly, we use:

```
benefit = recovery_time_saved - checkpoint_overhead
```

This is measured **after** a recovery event, giving empirical evidence of value.

### Threshold Adjustment Formula

```
                    α × |benefit|
delta = direction × ──────────────
                     |benefit| + 1

threshold_new = clamp(threshold_old + delta, min=0.05, max=1.0)
```

**Parameters:**
- `α` (learning_rate): Controls adaptation speed (default 0.1)
- `direction`: -1 (lower threshold) if benefit > 0, else +1
- `clamp`: Prevents threshold from becoming too extreme

**Properties:**
1. **Bounded**: Always in [0.05, 1.0]
2. **Proportional**: Larger benefits → larger adjustments
3. **Dampened**: Large benefits don't cause extreme changes
4. **Stable**: Converges as benefit → 0

### Dampening Function Analysis

```
f(benefit) = |benefit| / (|benefit| + 1)
```

**Behavior:**
```
benefit = 0.1   → f = 0.091  (small adjustment)
benefit = 1.0   → f = 0.500  (moderate adjustment)
benefit = 10.0  → f = 0.909  (large but dampened)
benefit = 100.0 → f = 0.990  (asymptotic to 1.0)
```

**Why dampen?**
- Prevents overreaction to single extreme event
- Ensures stability in stochastic environments
- Allows gradual convergence to optimal threshold

### Convergence Analysis

The threshold converges when:
```
benefit ≈ 0  (checkpoint overhead ≈ recovery benefit)
```

At convergence:
- Checkpointing exactly when cost/benefit is balanced
- Optimal use of checkpoint budget
- Minimal wasted effort

---

## Summary

### How Adaptation Works

1. **Observe**: Track task runtimes to understand workload
2. **Decide**: Checkpoint when ratio ≥ threshold AND budget available
3. **Learn**: Adjust threshold based on recovery outcomes

### What Makes It Adaptive

| Traditional | Adaptive |
|-------------|----------|
| Fixed schedule | Dynamic based on runtime |
| Same for all tasks | Identifies expensive tasks |
| No learning | Learns from failures |
| Ignores overhead | Balances cost vs benefit |
| Static threshold | Adapts threshold over time |

### Key Formulas

**Decision:**
```python
checkpoint_decision = (runtime / max_time_seen >= threshold) AND (used < budget)
```

**Adaptation:**
```python
benefit = recovery_time_saved - checkpoint_overhead
delta = -sign(benefit) × learning_rate × |benefit| / (|benefit| + 1)
threshold_new = clamp(threshold_old + delta, 0.05, 1.0)
```

### The Intelligence

The controller is "intelligent" because it:
- ✅ **Observes** runtime patterns
- ✅ **Predicts** which tasks are expensive
- ✅ **Decides** strategically within budget
- ✅ **Learns** from recovery events
- ✅ **Adapts** strategy over time
- ✅ **Balances** overhead vs benefit

This is **reinforcement learning** for checkpoint placement!
