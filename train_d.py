import os
import math

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
import matplotlib.pyplot as plt

from src.utils.func import select_target_type, print_msg
from src.loss import KappaLoss, FocalLoss, WarpedLoss
from src.scheduler import LossWeightsScheduler


def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
    device = cfg.base.device
    optimizer = initialize_optimizer(cfg, model)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

    # resume if checkpoint exists
    start_epoch = 0
    if cfg.base.checkpoint:
        start_epoch = resume(cfg, model, optimizer)

    model.train()
    max_indicator = float("-inf")
    history_train_loss = []
    history_train_accuracy = []
    history_validation_loss = []
    history_validation_accuracy = []

    for epoch in range(start_epoch, cfg.train.epochs):
        # update dynamic loss weights
        if loss_weight_scheduler:
            w = loss_weight_scheduler.step()
            loss_function.weight = w.to(device)

        epoch_loss = 0.0
        estimator.reset()

        loader = train_loader
        if cfg.base.progress:
            loader = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.train.epochs}",
                total=len(train_loader),
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(loader):
            scheduler_step = epoch + step / len(train_loader)
            lr = adjust_learning_rate(cfg, optimizer, scheduler_step)

            # unpack batch (with or without preloaded encoder states)
            if cfg.dataset.preload_path:
                X_side, key_states, value_states, y = batch
                key_states = key_states.to(device).transpose(0, 1)
                value_states = value_states.to(device).transpose(0, 1)
            else:
                X_lpm, X_side, y = batch
                X_lpm = X_lpm.to(device)
                with torch.no_grad():
                    _, key_states, value_states = frozen_encoder(
                        X_lpm, interpolate_pos_encoding=True
                    )
                key_states = key_states.to(device).transpose(0, 1)
                value_states = value_states.to(device).transpose(0, 1)

            X_side = X_side.to(device)
            y = select_target_type(y.to(device), cfg.train.criterion)

            # forward + loss
            y_pred = model(X_side, key_states, value_states)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)

            if cfg.base.progress:
                loader.set_postfix({"Loss": f"{avg_loss:.6f}", "LR": f"{lr:.4f}"})

        if cfg.base.progress:
            loader.close()

        # compute & log training metrics
        train_scores = estimator.get_scores(digits=4)
        train_acc = train_scores.get("acc", None)
        print(
            "Training metrics:", ", ".join(f"{k}: {v}" for k, v in train_scores.items())
        )

        # save periodic checkpoint
        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(cfg, model, epoch, optimizer, "checkpoint.pt")

        # validation
        if epoch % cfg.train.eval_interval == 0:
            val_loss = eval(
                cfg, frozen_encoder, model, val_loader, estimator, loss_function, device
            )
            val_scores = estimator.get_scores(digits=6)
            print_msg(
                "Validation metrics:", [f"{k}: {v}" for k, v in val_scores.items()]
            )

            # save best model
            indicator = val_scores[cfg.train.indicator]
            if indicator > max_indicator:
                save_weights(cfg, model, "best_validation_weights.pt")
                max_indicator = indicator

        # record history
        history_train_loss.append(avg_loss)
        history_train_accuracy.append(train_acc)
        history_validation_loss.append(val_loss)
        history_validation_accuracy.append(val_scores.get("acc", None))

    # plot and save performance curves
    plot_training_history(
        history_train_loss,
        history_train_accuracy,
        history_validation_loss,
        history_validation_accuracy,
        os.path.join(cfg.dataset.save_path, "performance_plots.png"),
    )

    # save final model
    save_weights(cfg, model, "final_weights.pt")
    return loss_function


def eval(cfg, frozen_encoder, model, dataloader, estimator, loss_function, device):
    model.eval()
    torch.set_grad_enabled(False)

    total_loss = 0.0
    total_batches = 0

    estimator.reset()
    for test_data in dataloader:
        if cfg.dataset.preload_path:
            X_side, key_states, value_states, y = test_data
            key_states = key_states.to(device).transpose(0, 1)
            value_states = value_states.to(device).transpose(0, 1)
        else:
            X_lpm, X_side, y = test_data
            X_lpm = X_lpm.to(device)
            with torch.no_grad():
                _, key_states, value_states = frozen_encoder(
                    X_lpm, interpolate_pos_encoding=True
                )

        X_side, y = X_side.to(device), y.to(device)
        y = select_target_type(y, cfg.train.criterion)

        y_pred = model(X_side, key_states, value_states)
        loss = loss_function(y_pred, y)

        total_loss += loss.item()
        total_batches += 1

        estimator.update(y_pred, y)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

    model.train()
    torch.set_grad_enabled(True)

    return avg_loss


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
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


# define loss and loss weights scheduler
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
        loss = nn.CrossEntropyLoss(weight=weight)
    elif criterion == "mean_square_error":
        loss = nn.MSELoss()
    elif criterion == "mean_absolute_error":
        loss = nn.L1Loss()
    elif criterion == "smooth_L1":
        loss = nn.SmoothL1Loss()
    elif criterion == "kappa_loss":
        loss = KappaLoss()
    elif criterion == "focal_loss":
        loss = FocalLoss()
    else:
        raise NotImplementedError("Not implemented loss function.")

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    parameters = model.parameters()
    solver = cfg.solver.optimizer
    if solver == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=cfg.dataset.learning_rate,
            momentum=cfg.solver.momentum,
            nesterov=cfg.solver.momentum,
            weight_decay=cfg.solver.weight_decay,
        )
    elif solver == "ADAM":
        optimizer = torch.optim.Adam(
            parameters,
            lr=cfg.dataset.learning_rate,
            betas=cfg.solver.betas,
            weight_decay=cfg.solver.weight_decay,
        )
    elif solver == "ADAMW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=cfg.dataset.learning_rate,
            betas=cfg.solver.betas,
            weight_decay=cfg.solver.weight_decay,
        )
    else:
        raise NotImplementedError("Not implemented optimizer.")

    return optimizer


def adjust_learning_rate(cfg, optimizer, epoch):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.dataset.learning_rate * epoch / cfg.train.warmup_epochs
    else:
        lr = (
            cfg.dataset.learning_rate
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - cfg.train.warmup_epochs)
                    / (cfg.train.epochs - cfg.train.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(cfg, model, epoch, optimizer, save_name):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(checkpoint, checkpoint_path)


def save_weights(cfg, model, save_name):
    save_path = os.path.join(cfg.dataset.save_path, save_name)
    torch.save(model.state_dict(), save_path)
    print_msg("Model saved at {}".format(save_path))


def resume(cfg, model, optimizer):
    checkpoint = cfg.base.checkpoint
    if os.path.exists(checkpoint):
        print_msg("Loading checkpoint {}".format(checkpoint))

        checkpoint = torch.load(checkpoint, map_location="cpu")
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print_msg(
            "Loaded checkpoint {} from epoch {}".format(checkpoint, checkpoint["epoch"])
        )
        return start_epoch
    else:
        print_msg("No checkpoint found at {}".format(checkpoint))
        raise FileNotFoundError("No checkpoint found at {}".format(checkpoint))


def plot_training_history(
    history_train_loss: List[float],
    history_train_accuracy: List[float],
    history_validation_loss: List[float],
    history_validation_accuracy: List[float],
    save_path: str,
) -> None:
    # Determine number of epochs
    epochs = list(range(1, len(history_train_loss) + 1))

    # Create a figure with two subplots: loss and accuracy
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    ax_loss.plot(epochs, history_train_loss, label="Training Loss", marker="o")
    ax_loss.plot(epochs, history_validation_loss, label="Validation Loss", marker="o")
    ax_loss.set_title("Loss over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    # Plot accuracy
    ax_acc.plot(epochs, history_train_accuracy, label="Training Accuracy", marker="o")
    ax_acc.plot(
        epochs, history_validation_accuracy, label="Validation Accuracy", marker="o"
    )
    ax_acc.set_title("Accuracy over Epochs")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    # Improve layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
