# image_retrieval/data/dataset.py

import os
from typing import Callable, Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class CarAccessoriesDataset(Dataset):
    """
    Custom PyTorch Dataset for car accessories.
    Assumes a directory structure like:
    /data_path/
        /product_0001/
            view_1.jpg
            view_2.jpg
            view_3.jpg
            ... (augmented images)
        /product_0002/
            ...
    """
    def __init__(self, data_path: str, num_views: int, transform: Callable):
        super().__init__()
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
        for path in view_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.transform(img))
            except Exception as e:
                print(f"Warning: Could not load image {path}. Skipping. Error: {e}")
                # Return a dummy tensor if an image fails to load
                return self.__getitem__((index + 1) % len(self))
        
        # Stack the views into a single tensor: (num_views, C, H, W)
        image_tensor = torch.stack(images, dim=0)
        
        return {
            "images": image_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }

def get_transforms(image_size: Tuple[int, int]) -> Callable:
    """
    Returns the image transformations for training and validation.
    The normalization stats must match those used for pre-training the ViT[cite: 164, 165].
    """
    # These are standard ImageNet stats, often used for ViT pre-training
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])