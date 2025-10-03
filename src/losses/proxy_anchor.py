# image_retrieval/losses/proxy_anchor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss with robustness to embedding-dimension mismatches.

    Behavior:
    - If `embedding_dim` is provided at construction, proxies are created with that size.
    - If `embedding_dim` is None, proxies are lazily initialized on first forward() using the
      incoming embeddings' dimension.
    - If proxies exist but model embeddings have a different last-dim, we create a
      learnable linear projector `embed_to_proxy` that maps embeddings -> proxy_dim.
    """

    def __init__(self, num_classes: int, embedding_dim: int | None = None, margin: float = 0.5, alpha: float = 32.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha

        # proxies may be created now or lazily in forward
        if embedding_dim is not None:
            p = torch.randn(num_classes, embedding_dim)
            self.proxies = nn.Parameter(p)
            nn.init.kaiming_normal_(self.proxies, mode="fan_out")
            self.proxy_dim = embedding_dim
        else:
            # not created yet
            self.proxies = None
            self.proxy_dim = None

        # projector (embedding_dim -> proxy_dim) created on demand when there's a mismatch
        self.embed_to_proxy = None

    def _ensure_proxies(self, embeddings: torch.Tensor):
        """
        Ensure proxies exist (create lazily if needed) and that a projector exists
        if embeddings.dim != proxy_dim.
        """
        emb_dim = embeddings.size(1)

        if self.proxies is None:
            # lazy init proxies to embedding dimension
            p = torch.randn(self.num_classes, emb_dim, device=embeddings.device, dtype=embeddings.dtype)
            self.proxies = nn.Parameter(p)
            nn.init.kaiming_normal_(self.proxies, mode="fan_out")
            self.proxy_dim = emb_dim
            self.embed_to_proxy = None  # no projector needed
            return

        # proxies already exist: check shape mismatch
        if self.proxy_dim != emb_dim:
            # create (or verify) projector: embeddings_dim -> proxy_dim
            if (self.embed_to_proxy is None) or (self.embed_to_proxy.weight.shape[1] != emb_dim) or (self.embed_to_proxy.weight.shape[0] != self.proxy_dim):
                # define linear layer: in_features=emb_dim, out_features=proxy_dim
                linear = nn.Linear(emb_dim, self.proxy_dim, bias=False).to(embeddings.device).to(embeddings.dtype)
                # reasonable init
                nn.init.kaiming_normal_(linear.weight, mode="fan_out")
                # register as module
                self.embed_to_proxy = linear

    def map_embeddings_to_proxy_space(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map embeddings to the same dimension as proxies, if necessary, and L2-normalize.
        Returns embeddings in proxy space (B, proxy_dim).
        """
        self._ensure_proxies(embeddings)

        if self.embed_to_proxy is not None:
            embeddings_proj = self.embed_to_proxy(embeddings)
            return F.normalize(embeddings_proj, p=2, dim=1)
        else:
            return F.normalize(embeddings, p=2, dim=1)

    def get_normalized_proxies(self) -> torch.Tensor:
        """Return L2-normalized proxies (num_classes, proxy_dim)."""
        if self.proxies is None:
            raise RuntimeError("Proxies have not been initialized yet. Call forward() with a batch first.")
        return F.normalize(self.proxies, p=2, dim=1)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, E) where E is model embedding dim
        labels: (B,)
        """
        # Map embeddings into proxy space and normalize
        embeddings_in_p = self.map_embeddings_to_proxy_space(embeddings)  # (B, proxy_dim)
        proxies = self.get_normalized_proxies()  # (num_classes, proxy_dim)

        # Cosine similarities (B, num_classes)
        cos_sim = F.linear(embeddings_in_p, proxies)

        # One-hot mask for positives
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float().to(cos_sim.device)

        # Positive similarities (length B)
        positive_cos_sim = cos_sim[one_hot_labels.bool()]

        # Positive loss (pull)
        loss_pos = torch.log(1 + torch.sum(torch.exp(-self.alpha * (positive_cos_sim - self.margin))))

        # Negative similarities (B, num_classes-1)
        negative_mask = 1.0 - one_hot_labels
        negative_cos_sim = cos_sim[negative_mask.bool()].view(embeddings_in_p.size(0), -1)

        # Negative loss (push)
        loss_neg = torch.log(1 + torch.sum(torch.exp(self.alpha * (negative_cos_sim + self.margin))))

        loss = (loss_pos / max(1, positive_cos_sim.numel())) + (loss_neg / max(1, negative_cos_sim.numel()))

        return loss
