"""
Quick demo script to run CIFAR-10 training with Ray adaptive checkpointing.
"""

from cifar10_pipeline import run_cifar10_pipeline, ray, STORE

if __name__ == "__main__":
    print("="*70)
    print("CIFAR-10 Training Demo with Ray Adaptive Checkpointing")
    print("="*70)
    print("\nThis demo will train a CNN on CIFAR-10 using:")
    print("  • Ray STATELESS tasks for data loading")
    print("  • Ray STATEFUL actors for model training")
    print("  • Adaptive checkpoint controller for fault tolerance")
    print("\nTraining Configuration:")
    print("  • 5 epochs")
    print("  • 5000 training samples")
    print("  • 1000 test samples")
    print("  • Batch size: 64")
    print("\n" + "="*70 + "\n")

    # Clean previous checkpoints
    STORE.clear_all()

    # Run training
    result = run_cifar10_pipeline(
        num_epochs=5,
        batch_size=64,
        train_subset_size=5000,
        test_subset_size=1000,
        simulate_failures=False
    )

    print("\n" + "="*70)
    print(f"Training Complete! Final Test Accuracy: {result['test_accuracy']*100:.2f}%")
    print("="*70)

    # Shutdown Ray
    ray.shutdown()
