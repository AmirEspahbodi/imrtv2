# image_retrieval/data/dataset.py

import os
import torch
from typing import Callable, Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T, datasets
from safetensors.torch import load_file
from pathlib import Path


class CarAccessoriesDataset(Dataset):
    """
    Custom PyTorch Dataset for car accessories.
    Now returns key_states and value_states for the new model.
    """
    def __init__(self, preload_path, data_path: str, num_views: int, transform: Callable):
        super().__init__()
        self.preload_path = preload_path
        self.data_path = data_path
        self.num_views = num_views
        self.transform = transform
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[List[str], int]]:
        """
        Creates a list of samples. Each sample is a tuple containing
        a list of image paths for the views and the class label.
        """
        samples = []
        class_dirs = sorted(os.listdir(self.data_path))
        label_map = {name: i for i, name in enumerate(class_dirs)}
        
        for class_dir in class_dirs:
            class_path = os.path.join(self.data_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # This is a simplified view selection. A more robust implementation
            # would intelligently select the 3 canonical views from the 50 images.
            # Here we assume the first `num_views` images are the canonical ones.
            image_files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path)])
            if len(image_files) >= self.num_views:
                view_paths = image_files[:self.num_views]
                label = label_map[class_dir]
                samples.append((view_paths, label))
        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        view_paths, label = self.samples[index]
        
        images = []
        keys_states = []
        values_states = []
        for path in view_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
                key_states, value_states = self.preload(path)
                keys_states.append(key_states)
                values_states.append(value_states)
            except Exception as e:
                print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
                # Return a dummy tensor if an image fails to load
                return self.__getitem__((index + 1) % len(self))
        
        # Stack the views into a single tensor: (num_views, C, H, W)
        image_tensor = torch.stack(images, dim=0)
        key_states_tensor = torch.stack(keys_states, dim=0)
        value_states_tensor = torch.stack(values_states, dim=0)
        return image_tensor, key_states_tensor, value_states_tensor, label

    def preload(self, path):
        # print(f"path = {path}")
        # print(f"type path = {type(path)}")
        
        states_path = os.path.join(self.preload_path, Path(path).stem + ".safetensors")
        states = load_file(states_path)
        key_states, value_states = states["key_states"], states["value_states"]
        return key_states, value_states
