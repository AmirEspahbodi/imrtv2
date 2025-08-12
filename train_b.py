import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Assuming these are your custom utility/loss imports
from src.utils.func import select_target_type, print_msg
from src.scheduler import LossWeightsScheduler
from evaluate_model import evaluate_model


# --- 1. Sharpness-Aware Minimization (SAM) Optimizer Implementation ---
class SAM(torch.optim.Optimizer):
    """
    Implements the Sharpness-Aware Minimization (SAM) optimizer.
    SAM seeks parameters in flat loss regions for better generalization.
    It wraps a base optimizer (e.g., AdamW) and performs a two-step update.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        # Instantiate the base optimizer with the provided parameter groups and kwargs
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step of SAM: find the worst-case weights in the neighborhood.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Store original parameter data
                self.state[p]["old_p"] = p.data.clone()
                # Calculate perturbation e_w
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                # Ascend to the worst-case weights
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Second step of SAM: update based on the gradients at the worst-case weights.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore original parameters
                p.data = self.state[p]["old_p"]
        # Perform the actual update using the base optimizer
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """
        Calculate the gradient norm, supporting adaptive SAM.
        """
        # Use the device of the first parameter as a shared device
        shared_device = self.param_groups[0]["params"][0].device
        # Stack all gradient norms to compute a final L2 norm
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    # Override step and load_state_dict for proper integration
    def step(self, closure=None):
        raise NotImplementedError(
            "SAM doesn't support closure for step. Use first_step and second_step."
        )

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# --- 2. Main Training Function (Refactored without Estimator) ---
def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Main training pipeline.
    This version removes the 'estimator' and calculates metrics directly.
    """
    device = cfg.base.device

    # Separate parameters for backbone and head for different learning rates
    backbone_params = [
        p
        for n, p in model.named_parameters()
        if "cnn_backbone" in n and p.requires_grad
    ]
    head_params = [
        p
        for n, p in model.named_parameters()
        if "cnn_backbone" not in n and p.requires_grad
    ]
    param_groups = [
        {"params": backbone_params, "lr": cfg.solver.backbone_lr},
        {"params": head_params, "lr": cfg.solver.head_lr},
    ]

    # Initialize optimizer, loss, and data loaders
    optimizer = initialize_optimizer(cfg, param_groups)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer)

    sam_start_epoch = cfg.train.sam_start_epoch
    best_val_acc = 0.0

    # History dictionary to store metrics for plotting
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("--- Starting Training (Refactored Logic without Estimator) ---")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()  # Set model to training mode
        lr = adjust_learning_rate(cfg, optimizer, epoch)

        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # --- Training Phase ---
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        progress = (
            tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}", leave=False
            )
            if cfg.base.progress
            else train_loader
        )

        for step, train_data in enumerate(progress):
            # --- Data Loading and Preprocessing ---
            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = train_data
                key_states, value_states = (
                    key_states.to(device),
                    value_states.to(device),
                )
                key_states, value_states = (
                    key_states.transpose(0, 1),
                    value_states.transpose(0, 1),
                )
            else:
                X_lpm, X_side, y = train_data
                X_lpm = X_lpm.to(device)
                with torch.no_grad():
                    _, key_states, value_states = frozen_encoder(
                        X_lpm, interpolate_pos_encoding=True
                    )

            X_side, y = X_side.to(device), y.to(device)
            y_true = select_target_type(y, cfg.train.criterion)

            # --- Forward Pass and Loss Calculation ---
            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y_true)

            # --- Optimizer Step (SAM or Standard) ---
            if epoch >= sam_start_epoch:
                # Two-step SAM update
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # Second forward/backward pass for SAM
                loss_function(
                    model(X_side, key_states, value_states), y_true
                ).backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard optimizer update
                optimizer.zero_grad()
                loss.backward()
                optimizer.base_optimizer.step()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # --- In-loop Metric Calculation ---
            total_train_loss += loss.item() * X_side.size(0)  # Weighted by batch size

            # Calculate accuracy for this batch
            _, predicted_labels = torch.max(y_pred, 1)
            correct_train += (predicted_labels == y_true).sum().item()
            total_train_samples += X_side.size(0)

            if cfg.base.progress:
                current_avg_loss = total_train_loss / total_train_samples
                progress.set_postfix(Loss=f"{current_avg_loss:.4f}", LR=f"{lr:.2e}")

        # --- End of Epoch: Calculate and Record Training Metrics ---
        avg_train_loss = total_train_loss / total_train_samples
        train_accuracy = correct_train / total_train_samples
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        # --- Validation Phase ---
        val_loss, val_accuracy = evaluate_model(
            cfg,
            frozen_encoder,
            model,
            val_loader,
            loss_function,
            device,
            just_loss_acc=True,
        )

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        print(
            f"\nEpoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch + 1} | Val Loss:   {val_loss:.4f}, Val Acc:   {val_accuracy:.4f}"
        )

        # --- Model Checkpointing ---
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            print(
                f"✨ New best validation accuracy: {best_val_acc:.4f}. Saving model..."
            )
            save_weights(cfg, model, "best_validation_weights.pt")

    save_weights(cfg, model, "final_weights.pt")
    print("--- Training finished. Final model saved. ---")

    # Generate and save performance plots
    save_plots(history, cfg.dataset.save_path)

    return loss_function


# --- 4. Helper Functions ---


def initialize_optimizer(cfg, params):
    base_optimizer_choice = torch.optim.AdamW
    print(
        f"--- Initializing SAM with base optimizer: {base_optimizer_choice.__name__} ---"
    )
    optimizer = SAM(
        params,
        base_optimizer_choice,
        rho=cfg.solver.rho,
        adaptive=True,
        betas=cfg.solver.betas,
        weight_decay=cfg.solver.weight_decay,
    )
    return optimizer


def initialize_loss(cfg, train_dataset):
    criterion = cfg.train.criterion

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.train.loss_weight
    if criterion == "cross_entropy":
        if loss_weight == "balance":
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == "dynamic":
            loss_weight_scheduler = LossWeightsScheduler(
                train_dataset, cfg.train.loss_weight_decay_rate
            )
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(
                loss_weight, dtype=torch.float32, device=cfg.base.device
            )
        loss = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=cfg.train.label_smoothing
        )
    elif criterion == "mean_square_error":
        loss = nn.MSELoss()
    elif criterion == "mean_absolute_error":
        loss = nn.L1Loss()
    elif criterion == "smooth_L1":
        loss = nn.SmoothL1Loss()
    else:
        raise NotImplementedError("Not implemented loss function.")

    return loss, loss_weight_scheduler


def adjust_learning_rate(cfg, optimizer, epoch):
    warmup_epochs = cfg.train.warmup_epochs
    total_epochs = cfg.train.epochs
    if epoch < warmup_epochs:
        # Linear warmup
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for i, param_group in enumerate(optimizer.param_groups):
        base_lr = (
            cfg.solver.head_lr
            if "head" in param_group.get("name", "")
            else cfg.solver.backbone_lr
        )
        param_group["lr"] = base_lr * lr_scale

    return optimizer.param_groups[1]["lr"]  # Return head LR for logging


def save_plots(history, save_path):
    print("--- Generating and saving performance plots... ---")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Training Accuracy")
    plt.plot(epochs, history["val_acc"], "r-o", label="Validation Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_path, "performance_plots.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Performance plots saved to {plot_path}")


def initialize_dataloader(cfg, train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    return train_loader, val_loader


def save_weights(cfg, model, save_name):
    if not os.path.exists(cfg.dataset.save_path):
        os.makedirs(cfg.dataset.save_path)
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved at: {save_path}")


def resume(cfg, model, optimizer):
    checkpoint_path = cfg.base.checkpoint
    if os.path.exists(checkpoint_path):
        print_msg(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        start_epoch = checkpoint.get("epoch", -1) + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print_msg(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        return start_epoch
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
