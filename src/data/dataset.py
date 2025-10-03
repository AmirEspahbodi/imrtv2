# image_retrieval/data/dataset.py

import os
import torch
from typing import Callable, Dict, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from safetensors.torch import load_file
from pathlib import Path


class CarAccessoriesDataset(Dataset):
    """
    Custom PyTorch Dataset for car accessories.
    Now works with single images instead of multiple views.
    """
    def __init__(self, preload_path: str, data_path: str, transform: Callable):
        super().__init__()
        self.preload_path = preload_path
        self.data_path = data_path
        self.transform = transform
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[str, int]]:
        """
        Creates a list of samples. Each sample is a tuple containing
        an image path and the class label.
        """
        samples = []
        class_dirs = sorted(os.listdir(self.data_path))
        label_map = {name: i for i, name in enumerate(class_dirs)}
        
        for class_dir in class_dirs:
            class_path = os.path.join(self.data_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            # Get all image files in the class directory
            image_files = sorted([
                os.path.join(class_path, f) 
                for f in os.listdir(class_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ])
            
            # Add each image as a separate sample
            for image_path in image_files:
                label = label_map[class_dir]
                samples.append((image_path, label))
        
        print(f"Created dataset with {len(samples)} samples from {len(class_dirs)} classes")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path, label = self.samples[index]
        
        try:
            # Load and transform the single image
            img = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(img)
            
            # Load precomputed key_states and value_states
            key_states, value_states = self._preload(image_path)
            
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Error: {e}")
            # Return next sample if current one fails
            return self.__getitem__((index + 1) % len(self))
        
        return image_tensor, key_states, value_states, torch.tensor(label, dtype=torch.long)

    def _preload(self, image_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load precomputed key_states and value_states from safetensors file.
        """
        # Extract filename without extension
        filename = Path(image_path).stem
        states_path = os.path.join(self.preload_path, filename + ".safetensors")
        
        try:
            states = load_file(states_path)
            key_states = states["key_states"]
            value_states = states["value_states"]
            return key_states, value_states
        except Exception as e:
            raise RuntimeError(f"Warning: Could not load states from {states_path}. Error: {e}")


def get_transforms(image_size: Tuple[int, int]) -> Callable:
    """
    Returns the image transformations for training and validation.
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])