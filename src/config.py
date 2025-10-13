# image_retrieval/config.py

from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class NetworkConfig:
    """Configuration for the CoAtNetSideViT model."""
    backbone_trainable_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    vit1_feature_strame: List[int] = field(default_factory=lambda: [2, 3])
    vit2_feature_strame: List[int] = field(default_factory=lambda: [1, 2])
    side_input_size: int = 224

@dataclass
class DatasetConfig:
    """Configuration for data loading and processing."""
    data_path: str = "/path/to/your/product_images"
    image_size: Tuple[int, int] = (384, 384)
    image_channel_num: int = 3
    num_views: int = 3
    num_classes: int = 118
    batch_size: int = 32
    num_workers: int = 8

@dataclass
class LossConfig:
    """Configuration for the loss function."""
    num_classes: int = 1500
    embedding_dim: int = 384  # Updated for new model output
    margin: float = 0.1
    alpha: float = 8.0

@dataclass
class TrainConfig:
    """Configuration for the training process."""
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    criterion: str = "proxy_anchor"

@dataclass
class ProjectConfig:
    """Main configuration container."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

# Instantiate the main config
cfg = ProjectConfig()