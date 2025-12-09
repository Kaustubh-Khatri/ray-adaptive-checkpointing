"""
Quick demo script to run the real ML pipeline.
"""

from real_ml_pipeline import run_ml_pipeline, ray, STORE

if __name__ == "__main__":
    print("="*70)
    print("Quick ML Demo - Training Neural Network on MNIST")
    print("="*70)

    # Clean previous checkpoints
    STORE.clear_all()

    # Run a quick training session
    print("\nTraining a small neural network on MNIST subset...")
    print("This will train for 3 epochs with adaptive checkpointing.\n")

    result = run_ml_pipeline(
        num_epochs=3,
        batch_size=64,
        subset_size=500,  # Smaller subset for faster demo
        simulate_failures=False
    )

    print("\n" + "="*70)
    print(f"Training Complete! Final Test Accuracy: {result['test_accuracy']:.2f}%")
    print("="*70)

    # Shutdown Ray
    ray.shutdown()
