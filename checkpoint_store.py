import pickle
from pathlib import Path
from typing import Any

class CheckpointStore:
    """
    Simple checkpoint store for Ray pipeline states.
    Stores intermediate results as pickle files.
    """

    def __init__(self, store_dir: str = "./ray_ckpts"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)

    def save(self, key: str, data: Any):
        """Save checkpoint data with given key."""
        filepath = self.store_dir / f"{key}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"[STORE] Saved checkpoint: {key}")

    def load(self, key: str) -> Any:
        """Load checkpoint data by key."""
        filepath = self.store_dir / f"{key}.pkl"
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {key}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"[STORE] Loaded checkpoint: {key}")
        return data

    def exists(self, key: str) -> bool:
        """Check if checkpoint exists."""
        filepath = self.store_dir / f"{key}.pkl"
        return filepath.exists()

    def delete(self, key: str):
        """Delete checkpoint by key."""
        filepath = self.store_dir / f"{key}.pkl"
        if filepath.exists():
            filepath.unlink()
            print(f"[STORE] Deleted checkpoint: {key}")

    def clear_all(self):
        """Clear all checkpoints in store."""
        for file in self.store_dir.glob("*.pkl"):
            file.unlink()
        print(f"[STORE] Cleared all checkpoints")
