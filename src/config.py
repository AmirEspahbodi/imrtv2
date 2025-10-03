# image_retrieval/config.py

from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class LossConfig:
    """Configuration for the loss function."""
    num_classes: int = 1500 # Example: total number of unique products
    embedding_dim: int = 768
    margin: float = 0.5
    alpha: float = 32.0 # Scaling factor for the loss

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    device: str = "cuda"
    accelerator: str = "auto" # For PyTorch Lightning or Accelerate
    fast_dev_run: bool = False # For debugging: run one batch of train/val

@dataclass
class ProjectConfig:
    """Main configuration container."""
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

# Instantiate the main config
cfg = ProjectConfig()