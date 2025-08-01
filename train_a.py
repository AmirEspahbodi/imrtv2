import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from src.utils.func import print_msg
from src.scheduler import LossWeightsScheduler
from evaluate_model import evaluate_model, prepare_batch


def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    """
    Main training function with corrected metric calculation, checkpointing, and SWA logic.
    """
    device = torch.device(
        cfg.base.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    if frozen_encoder:
        frozen_encoder.to(device)
        frozen_encoder.eval()

    # --- Initialization ---
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
        {"params": backbone_params, "lr": cfg.solver.get("backbone_lr", 1e-4)},
        {"params": head_params, "lr": cfg.solver.get("head_lr", 1e-3)},
    ]

    optimizer = initialize_optimizer(cfg, param_groups)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    main_scheduler = OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in param_groups],
        total_steps=len(train_loader) * cfg.train.get("epochs", 10),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    swa_start_epoch = cfg.train.get("swa_start_epoch", 5)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg.solver.get("swa_lr", 5e-4))

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # --- Comprehensive State Tracking for Resuming ---
    start_epoch = 0
    max_indicator = 0.0
    epochs_no_improve = 0

    if cfg.base.get("checkpoint"):
        start_epoch, max_indicator, epochs_no_improve = resume(
            cfg, model, optimizer, main_scheduler, swa_model, swa_scheduler
        )

    print("--- Starting Training ---")

    # --- Main Training Loop ---
    for epoch in range(start_epoch, cfg.train.get("epochs", 10)):
        model.train()
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        running_loss, correct_predictions, total_samples = 0.0, 0, 0
        progress = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.train.get('epochs', 10)}",
                leave=False,
            )
            if cfg.base.get("progress", True)
            else train_loader
        )

        for batch in progress:
            X_side, key_states, value_states, y_true, _ = prepare_batch(
                batch, cfg, frozen_encoder, device
            )

            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if epoch < swa_start_epoch:
                main_scheduler.step()

            running_loss += loss.item() * X_side.size(0)
            _, predicted_labels = torch.max(y_pred, 1)
            correct_predictions += (predicted_labels == y_true).sum().item()
            total_samples += X_side.size(0)

            if cfg.base.get("progress", True):
                progress.set_postfix(
                    Loss=f"{running_loss / total_samples:.4f}",
                    Acc=f"{correct_predictions / total_samples:.4f}",
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        avg_train_loss = running_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        eval_model = swa_model if epoch >= swa_start_epoch else model
        val_loss, val_accuracy = evaluate_model(
            cfg, frozen_encoder, eval_model, val_loader, loss_function, device, just_loss_acc=True
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_accuracy)

        print(
            f"\nEpoch {epoch + 1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
        )
        print(
            f"Epoch {epoch + 1} Val Loss:   {val_loss:.4f}, Val Acc:   {val_accuracy:.4f}"
        )

        # --- Corrected Early Stopping and Checkpointing ---
        indicator = val_accuracy
        if indicator > max_indicator:
            max_indicator = indicator
            epochs_no_improve = 0
            # **FIX:** Save the model that was evaluated (`eval_model`)
            save_weights(cfg, eval_model, "best_validation_weights.pt")
        else:
            epochs_no_improve += 1

        # **FIX:** Save a comprehensive checkpoint at the end of each epoch
        save_checkpoint(
            cfg,
            epoch,
            model,
            optimizer,
            main_scheduler,
            swa_model,
            swa_scheduler,
            max_indicator,
            epochs_no_improve,
            "latest_checkpoint.pt",
        )

        if epochs_no_improve >= cfg.train.get("early_stopping_patience", 10):
            print(
                f"--- Early stopping triggered after {epochs_no_improve} epochs with no improvement. ---"
            )
            save_weights(cfg, eval_model, "final_weights.pt")
            break

    # --- Finalize SWA Model ---
    print("\n--- Training finished. Updating SWA model batch norm statistics. ---")
    update_swa_batchnorm(cfg, swa_model, train_loader, frozen_encoder, device)
    save_weights(cfg, swa_model, "final_weights.pt")

    save_plots(cfg, history)
    return loss_function


def update_swa_batchnorm(cfg, swa_model, train_loader, frozen_encoder, device):
    swa_model.train()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            X_side, key_states, value_states, _, _ = prepare_batch(
                batch, cfg, frozen_encoder, device
            )
            swa_model(X_side, key_states, value_states)


# --- Helper and Initialization Functions ---


def initialize_optimizer(cfg, params):
    # (No changes from previous version)
    solver = cfg.solver.get("optimizer", "ADAMW")
    lr = cfg.solver.get("head_lr", 1e-3)
    betas = cfg.solver.get("betas", (0.9, 0.999))
    weight_decay = cfg.solver.get("weight_decay", 1e-2)
    momentum = cfg.solver.get("momentum", 0.9)

    if solver == "SGD":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif solver == "ADAM":
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif solver == "ADAMW":
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {solver} not implemented.")


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


def initialize_dataloader(cfg, train_dataset, val_dataset):
    # (No changes from previous version)
    batch_size = cfg.train.get("batch_size", 32)
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


def save_checkpoint(
    cfg,
    epoch,
    model,
    optimizer,
    main_scheduler,
    swa_model,
    swa_scheduler,
    max_indicator,
    epochs_no_improve,
    save_name,
):
    """Saves a comprehensive checkpoint."""
    save_path_dir = cfg.dataset.get("save_path", "./models")
    os.makedirs(save_path_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "main_scheduler_state_dict": main_scheduler.state_dict(),
        "swa_model_state_dict": swa_model.state_dict(),
        "swa_scheduler_state_dict": swa_scheduler.state_dict(),
        "max_indicator": max_indicator,
        "epochs_no_improve": epochs_no_improve,
    }
    torch.save(checkpoint, os.path.join(save_path_dir, save_name))


def save_weights(cfg, model, save_name):
    """Saves only the model weights (state_dict)."""
    save_path_dir = cfg.dataset.get("save_path", "./models")
    os.makedirs(save_path_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path_dir, save_name))
    print_msg(f"Model weights saved to {os.path.join(save_path_dir, save_name)}")


def resume(cfg, model, optimizer, main_scheduler, swa_model, swa_scheduler):
    """Resumes training from a comprehensive checkpoint."""
    checkpoint_path = cfg.base.get("checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print_msg(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        main_scheduler.load_state_dict(checkpoint["main_scheduler_state_dict"])
        swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
        swa_scheduler.load_state_dict(checkpoint["swa_scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        max_indicator = checkpoint.get("max_indicator", 0.0)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)

        print_msg(
            f"Resumed from epoch {checkpoint.get('epoch', 0)}. Starting at epoch {start_epoch}."
        )
        return start_epoch, max_indicator, epochs_no_improve
    else:
        print_msg(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return 0, 0.0, 0


def save_plots(cfg, history):
    # (No changes from previous version)
    save_path_dir = cfg.dataset.get("save_path", "./plots")
    os.makedirs(save_path_dir, exist_ok=True)
    model_name = cfg.network.get("model", "model")
    print("--- Generating and saving performance plots... ---")

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
    plot_path = os.path.join(save_path_dir, f"{model_name}_performance_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Performance plots saved to {plot_path}")
