# image_retrieval/losses/proxy_anchor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProxyAnchorLoss(nn.Module):
    """
    Implementation of Proxy-Anchor Loss.
    This loss avoids sampling pairs/triplets by using class proxies, making
    training more efficient and stable[cite: 50, 53]. Its key innovation is
    treating proxies as anchors, which leverages sample-to-sample information
    more efficiently than older proxy methods[cite: 55, 56].
    """
    def __init__(self, num_classes: int, embedding_dim: int, margin: float = 0.5, alpha: float = 32.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha
        
        # Proxies: learnable representative vectors for each class
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize both embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Cosine similarity matrix between embeddings and proxies
        cos_sim = F.linear(embeddings, proxies) # Shape: (B, num_classes)
        
        # Create a one-hot mask for positive proxy identification
        one_hot_labels = F.one_hot(labels, self.num_classes).float()
        
        # --- Positive Pull ---
        # Select similarity to the positive proxy for each sample
        positive_cos_sim = cos_sim[one_hot_labels.bool()]
        
        # Loss for positive pairs (we want to pull them closer)
        # The log-sum-exp trick is used for numerical stability
        loss_pos = torch.log(1 + torch.sum(torch.exp(-self.alpha * (positive_cos_sim - self.margin))))

        # --- Negative Push ---
        # Select similarities to all negative proxies for each sample
        negative_cos_sim_mask = 1 - one_hot_labels
        negative_cos_sim = cos_sim[negative_cos_sim_mask.bool()].view(embeddings.size(0), -1)
        
        # Loss for negative pairs (we want to push them apart)
        loss_neg = torch.log(1 + torch.sum(torch.exp(self.alpha * (negative_cos_sim + self.margin))))

        # Total loss
        # The paper suggests 1/N_pos and 1/N_neg scaling
        loss = (loss_pos / len(positive_cos_sim)) + (loss_neg / negative_cos_sim.numel())
        
        return loss

    def map_embeddings_to_proxy_space(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map embeddings to the proxy space for evaluation"""
        return F.normalize(embeddings, p=2, dim=1)

    def get_normalized_proxies(self) -> torch.Tensor:
        """Get normalized proxies for evaluation"""
        return F.normalize(self.proxies, p=2, dim=1)
