# image_retrieval/engine/core.py

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from src.utils.func import select_target_type



class Trainer:
    """
    Encapsulates the training and evaluation logic for the retrieval model.
    """
    def __init__(
        self,
        model: nn.Module,
        frozen_model: nn.Module,
        train_loader,
        val_loader,
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
        self.train_loader = train_loader
        self.val_loader = val_loader

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
                # print(f"X_side.shape = {X_side.shape}")
                # print(f"key_states.shape = {key_states.shape}")
                # print(f"value_states.shape = {value_states.shape}")
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
        For validation, we compute a proxy-based accuracy: for each image, is
        its closest proxy the one corresponding to its true class? This is a
        good indicator of embedding space quality.
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Get normalized proxies from the loss function module
        proxies = torch.nn.functional.normalize(self.loss_fn.proxies, p=2, dim=1)
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating", colour="yellow")
        for batch in progress_bar:
            X_side, key_states, value_states, y_true, _ = prepare_batch(
                batch, self.cfg, self.frozen_model, self.device
            )

            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                embeddings = self.model(X_side, key_states, value_states)
                loss = self.loss_fn(embeddings, y_true)
            
            # Calculate proxy-based accuracy
            cos_sim = torch.matmul(embeddings, proxies.T)
            predicted_labels = torch.argmax(cos_sim, dim=1)
            
            correct_predictions += (predicted_labels == y_true).sum().item()
            total_samples += y_true.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return {"val_loss": avg_loss, "proxy_accuracy": accuracy}
    


def prepare_batch(batch, cfg, frozen_encoder, device):
    """Prepares a single batch of data, moving it to the correct device."""
    X_side, key_states, value_states, y = batch
    key_states, value_states = key_states.to(device), value_states.to(device)
    key_states, value_states = (
        key_states.transpose(0, 1),
        value_states.transpose(0, 1),
    )
    X_side, y = X_side.to(device), y.to(device)
    y_true = select_target_type(y, cfg.train.criterion)
    return X_side, key_states, value_states, y_true, y


