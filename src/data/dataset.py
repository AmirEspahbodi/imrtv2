# image_retrieval/data/dataset.py

import os
from typing import Callable, Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class CarAccessoriesDataset(Dataset):
    """
    UPDATED Custom PyTorch Dataset for car accessories.
    Handles both multi-view (training) and single-view (evaluation) scenarios.
    """
    def __init__(self, data_path: str, num_views: int, transform: Callable, is_train: bool = True):
        super().__init__()
        self.data_path = data_path
        self.num_views = num_views
        self.transform = transform
        self.is_train = is_train
        self.samples = self._create_samples()
        
    def _create_samples(self) -> List[Tuple[List[str], int]]:
        samples = []
        class_dirs = sorted(os.listdir(self.data_path))
        label_map = {name: i for i, name in enumerate(class_dirs)}
        
        for class_dir in class_dirs:
            class_path = os.path.join(self.data_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            image_files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path)])

            # If training, we require the full number of views
            if self.is_train:
                if len(image_files) >= self.num_views:
                    view_paths = image_files[:self.num_views]
                    label = label_map[class_dir]
                    samples.append((view_paths, label))
            # If evaluating, we can use any product with at least one image
            else:
                if len(image_files) > 0:
                    # Take the first image as the representative for evaluation
                    view_paths = [image_files[0]] 
                    label = label_map[class_dir]
                    samples.append((view_paths, label))
        
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in {self.data_path}. Check your data path and structure.")
            
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        view_paths, label = self.samples[index]
        
        images = []
        for path in view_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except Exception as e:
                print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
                return self.__getitem__((index + 1) % len(self))
        
        image_tensor = torch.stack(images, dim=0)
        
        return {
            "images": image_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }

# get_transforms function remains the same
def get_transforms(image_size: Tuple[int, int]) -> Callable:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])