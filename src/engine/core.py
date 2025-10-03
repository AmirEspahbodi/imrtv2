# image_retrieval/engine/core.py

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
from src.evaluate_model import evaluate_model, prepare_batch


class Trainer:
    """
    Encapsulates the training and evaluation logic for the retrieval model.
    """
    def __init__(
        self,
        model: nn.Module,
        frozen_model: nn.Module,
        train_dataset,
        val_dataset,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        cfg: Any,
        scheduler: Any = None,
        scaler: torch.cuda.amp.GradScaler = None,
    ):
        self.model = model
        self.frozen_model = frozen_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.scaler = scaler
        self.cfg = cfg
        self.train_loader, self.val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    def train_one_epoch(self) -> float:
        """
        Performs one full epoch of training.
        """
        
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training", colour="green")

        for batch in progress_bar:
            X_side, key_states, value_states, y_true, _ = prepare_batch(
                batch, self.cfg, self.frozen_model, self.device
            )

            self.optimizer.zero_grad()
            with torch.amp.autocast(enabled=self.scaler is not None, device_type=self.device.type):
                embeddings = self.model(X_side, key_states, value_states)
                loss = self.loss_fn(embeddings, y_true)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Performs evaluation on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(self.val_loader, desc="Evaluating", colour="yellow")
        for batch in progress_bar:
            # prepare_batch returns the same inputs as in training
            X_side, key_states, value_states, y_true, _ = prepare_batch(
                batch, self.cfg, self.frozen_model, self.device
            )

            # No need to zero_grad in evaluation
            with torch.amp.autocast(enabled=self.scaler is not None, device_type=self.device.type):
                embeddings = self.model(X_side, key_states, value_states)
                loss = self.loss_fn(embeddings, y_true)

            # Map embeddings into proxy space (handles mismatched dims / projector)
            embeddings_in_proxy_space = self.loss_fn.map_embeddings_to_proxy_space(embeddings)
            proxies = self.loss_fn.get_normalized_proxies()  # (num_classes, proxy_dim)

            # Calculate proxy-based accuracy using matched dims
            cos_sim = torch.matmul(embeddings_in_proxy_space, proxies.T)
            predicted_labels = torch.argmax(cos_sim, dim=1)

            correct_predictions += (predicted_labels == y_true).sum().item()
            total_samples += y_true.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

        return {"val_loss": avg_loss, "proxy_accuracy": accuracy}



def initialize_dataloader(cfg, train_dataset, val_dataset):
    # (No changes from previous version)
    batch_size = cfg.train.get("batch_size", 16)
    num_workers = cfg.train.get("num_workers", 4)
    pin_memory = cfg.train.get("pin_memory", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader

