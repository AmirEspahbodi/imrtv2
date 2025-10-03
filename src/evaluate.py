# image_retrieval/evaluate.py

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.config import cfg
from src.data.dataset import CarAccessoriesDataset, get_transforms
from src.models.architecture import MultiViewModel

@torch.no_grad()
def extract_embeddings(dataloader, model, device):
    """Extracts embeddings for all items in a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Extracting Embeddings"):
        images = batch["images"].to(device)
        labels = batch["label"]
        
        embeddings = model(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
        
    return torch.cat(all_embeddings), torch.cat(all_labels)

def precision_at_k(retrieved_indices, true_label, k):
    """Calculates Precision@k."""
    return int(true_label in retrieved_indices[:k])

def average_precision(retrieved_indices, true_label):
    """Calculates Average Precision (AP) for a single query."""
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(retrieved_indices):
        if p == true_label:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if num_hits == 0:
        return 0.0
    return score / num_hits

def main():
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    # --- Load Model from Checkpoint ---
    model = MultiViewModel(cfg.model).to(device)
    checkpoint_path = Path("./checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError("Best model checkpoint not found. Please train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with accuracy {checkpoint['accuracy']:.4f}")

    # --- Data Pipeline for Test Set ---
    # NOTE: Replace 'path/to/test_data' with your actual path
    test_transform = get_transforms(cfg.data.image_size)
    test_dataset = CarAccessoriesDataset(
        data_path='path/to/test_data', 
        num_views=cfg.data.num_views, 
        transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.data.batch_size, 
        num_workers=cfg.data.num_workers
    )

    # --- Evaluation ---
    # In a typical retrieval setup, the test set serves as both the query set and the gallery
    gallery_embeddings, gallery_labels = extract_embeddings(test_loader, model, device)
    query_embeddings, query_labels = gallery_embeddings, gallery_labels
    
    print(f"Gallery size: {len(gallery_labels)} items")

    # Calculate pairwise cosine similarity
    similarity_matrix = torch.matmul(query_embeddings, gallery_embeddings.T)
    
    # Get ranked lists of indices for each query
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

    # --- Calculate Metrics ---
    k_values = [1, 5, 10]
    precisions = {k: [] for k in k_values}
    aps = []

    for i in tqdm(range(len(query_labels)), desc="Calculating Metrics"):
        true_label = query_labels[i].item()
        
        # Get gallery labels corresponding to the retrieved indices
        # We ignore the first result since it will always be the query image itself
        retrieved_labels = gallery_labels[sorted_indices[i, 1:]] 
        
        for k in k_values:
            precisions[k].append(precision_at_k(retrieved_labels, true_label, k))
        
        aps.append(average_precision(retrieved_labels, true_label))

    # --- Print Results ---
    print("\n--- Retrieval Performance ---")
    for k in k_values:
        print(f"Precision@{k}: {np.mean(precisions[k]):.4f}")
    
    mAP = np.mean(aps)
    print(f"mAP (mean Average Precision): {mAP:.4f}")

if __name__ == "__main__":
    main()