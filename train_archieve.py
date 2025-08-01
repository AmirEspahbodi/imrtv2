# import os
# import math

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import OneCycleLR

# from src.utils.func import *
# from src.loss import *
# from src.scheduler import *

# def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
#     device = cfg.base.device

#     # Differential Learning Rates Setup
#     backbone_params = [p for n, p in model.named_parameters() if 'cnn_backbone' in n and p.requires_grad]
#     head_params = [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and p.requires_grad]
#     param_groups = [
#         {'params': backbone_params, 'lr': cfg.solver.backbone_lr},
#         {'params': head_params, 'lr': cfg.solver.head_lr}
#     ]

#     optimizer = initialize_optimizer(cfg, param_groups)
#     loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
#     train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

#     # Schedulers: OneCycleLR for main training
#     main_scheduler = OneCycleLR(
#         optimizer,
#         max_lr=[g['lr'] for g in param_groups],
#         total_steps=len(train_loader) * cfg.train.epochs,
#         pct_start=0.1,
#         anneal_strategy='cos'
#     )

#     # SWA Setup
#     swa_start_epoch = cfg.train.swa_start_epoch
#     swa_model = AveragedModel(model)
#     swa_scheduler = SWALR(optimizer, swa_lr=cfg.solver.swa_lr)

#     # Early Stopping & History Tracking
#     max_indicator = 0
#     epochs_no_improve = 0
#     patience = cfg.train.early_stopping_patience
#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

#     start_epoch = 0
#     if cfg.base.checkpoint:
#         start_epoch = resume(cfg, model, optimizer)

#     print("--- Starting Training with Plan A (SWA) ---")
#     model.train()
#     for epoch in range(start_epoch, cfg.train.epochs):
#         if loss_weight_scheduler:
#             weight = loss_weight_scheduler.step()
#             loss_function.weight = weight.to(device)

#         epoch_loss = 0
#         estimator.reset()
#         progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', leave=False) if cfg.base.progress else train_loader

#         for step, train_data in enumerate(progress):
#             if cfg.dataset.preload_path:
#                 X_side, key_states, value_states, y = train_data
#                 key_states, value_states = key_states.to(device), value_states.to(device)
#                 key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
#             else:
#                 X_lpm, X_side, y = train_data
#                 X_lpm = X_lpm.to(device)
#                 with torch.no_grad():
#                     _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

#             X_side, y = X_side.to(device), y.to(device)
#             y = select_target_type(y, cfg.train.criterion)

#             y_pred = model(X_side, key_states, value_states)
#             loss = loss_function(y_pred, y)

#             optimizer.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()

#             if epoch < swa_start_epoch:
#                 main_scheduler.step()

#             epoch_loss += loss.item()
#             estimator.update(y_pred, y)
#             if cfg.base.progress:
#                 progress.set_postfix(Loss=f'{epoch_loss/(step+1):.4f}', LR=f"{optimizer.param_groups[0]['lr']:.2e}")

#         # SWA Scheduler Step
#         if epoch >= swa_start_epoch:
#             swa_model.update_parameters(model)
#             swa_scheduler.step()

#         # Record training metrics
#         avg_train_loss = epoch_loss / len(train_loader)
#         train_scores = estimator.get_scores(4)
#         history['train_loss'].append(avg_train_loss)
#         history['train_acc'].append(train_scores.get(cfg.train.indicator, 0))

#         # Validation
#         eval_model = swa_model if epoch >= swa_start_epoch else model
#         val_loss, val_scores = eval(cfg, frozen_encoder, eval_model, val_loader, estimator, device, loss_function)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_scores.get(cfg.train.indicator, 0))

#         print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_scores.get(cfg.train.indicator, 0):.4f}")
#         print(f"Epoch {epoch+1} Val Loss:   {val_loss:.4f}, Val Acc:   {val_scores.get(cfg.train.indicator, 0):.4f}")

#         indicator = val_scores[cfg.train.indicator]
#         if indicator > max_indicator:
#             max_indicator = indicator
#             epochs_no_improve = 0
#             save_weights(cfg, model, 'best_validation_weights.pt')
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
#             break

#     # Finalize SWA Model
#     print("\n--- Training finished. Updating SWA model batch norm statistics. ---")
#     swa_model.train()
#     with torch.no_grad():
#         for i, train_data in enumerate(train_loader):
#             if i >= 100: break
#             if cfg.dataset.preload_path:
#                 X_side, key_states, value_states, y = train_data
#                 key_states, value_states = key_states.to(device), value_states.to(device)
#                 key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
#             else:
#                 X_lpm, X_side, y = train_data
#                 X_lpm = X_lpm.to(device)
#                 _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)
#             X_side = X_side.to(device)
#             swa_model(X_side, key_states, value_states)

#     save_weights(cfg, swa_model, 'swa_model_final_weights.pt')

#     # Save plots
#     save_plots(cfg, history, cfg.dataset.save_path)

#     return swa_model


# def save_plots(cfg, history, save_path):
#     """Saves plots for training/validation loss and accuracy."""
#     print("--- Generating and saving performance plots... ---")
#     epochs = range(1, len(history['train_loss']) + 1)

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
#     plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path, f'{cfg.network.model}_plana_loss_plot.png'))
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
#     plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path, f'{cfg.network.model}_plana_accuracy_plot.png'))
#     plt.close()
#     print(f"Performance plots saved to {save_path}")


# def eval(cfg, frozen_encoder, model, dataloader, estimator, device, loss_function):
#     model.eval()
#     torch.set_grad_enabled(False)
#     estimator.reset()
#     total_loss = 0.0

#     for test_data in dataloader:
#         if cfg.dataset.preload_path:
#             X_side, key_states, value_states, y = test_data
#             key_states, value_states = key_states.to(device), value_states.to(device)
#             key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
#         else:
#             X_lpm, X_side, y = test_data
#             X_lpm = X_lpm.to(device)
#             with torch.no_grad():
#                 _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

#         X_side, y = X_side.to(device), y.to(device)
#         y_true = select_target_type(y, cfg.train.criterion)
#         y_pred = model(X_side, key_states, value_states)

#         loss = loss_function(y_pred, y_true)
#         total_loss += loss.item()
#         estimator.update(y_pred, y_true)

#     model.train()
#     torch.set_grad_enabled(True)
#     avg_loss = total_loss / len(dataloader)
#     scores = estimator.get_scores(6)
#     return avg_loss, scores

# def initialize_optimizer(cfg, params): # Now accepts params directly
#     solver = cfg.solver.optimizer
#     if solver == 'SGD':
#         optimizer = torch.optim.SGD(params, lr=cfg.solver.head_lr, momentum=cfg.solver.momentum, weight_decay=cfg.solver.weight_decay)
#     elif solver == 'ADAM':
#         optimizer = torch.optim.Adam(params, lr=cfg.solver.head_lr, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
#     elif solver == 'ADAMW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.solver.head_lr, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
#     else:
#         raise NotImplementedError('Not implemented optimizer.')
#     return optimizer


# def initialize_loss(cfg, train_dataset):
#     criterion = cfg.train.criterion
#     weight = None
#     loss_weight_scheduler = None
#     loss_weight = cfg.train.loss_weight
#     if criterion == 'cross_entropy':
#         if loss_weight == 'balance':
#             loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
#         elif loss_weight == 'dynamic':
#             loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.train.loss_weight_decay_rate)
#         elif isinstance(loss_weight, list):
#             assert len(loss_weight) == len(train_dataset.classes)
#             weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.base.device)
#         loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)
#     elif criterion == 'mean_square_error':
#         loss = nn.MSELoss()
#     elif criterion == 'mean_absolute_error':
#         loss = nn.L1Loss()
#     elif criterion == 'smooth_L1':
#         loss = nn.SmoothL1Loss()
#     elif criterion == 'kappa_loss':
#         loss = KappaLoss()
#     elif criterion == 'focal_loss':
#         loss = FocalLoss()
#     else:
#         raise NotImplementedError('Not implemented loss function.')

#     # loss_function = WarpedLoss(loss, criterion)
#     return loss, loss_weight_scheduler


# def adjust_learning_rate(cfg, optimizer, epoch):
#     """Decays the learning rate with half-cycle cosine after warmup"""
#     if epoch < cfg.train.warmup_epochs:
#         lr = cfg.dataset.learning_rate * epoch / cfg.train.warmup_epochs
#     else:
#         lr = cfg.dataset.learning_rate * 0.5 * (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


# def save_checkpoint(cfg, model, epoch, optimizer, save_name):
#     checkpoint = {
#         'epoch': epoch,
#         'state_dict': model.state_dict(),
#         'optimizer' : optimizer.state_dict()
#     }
#     checkpoint_path = os.path.join(cfg.dataset.save_path, save_name)
#     torch.save(checkpoint, checkpoint_path)


# def save_weights(cfg, model, save_name):
#     save_path = os.path.join(cfg.dataset.save_path, save_name)
#     torch.save(model.state_dict(), save_path)
#     print_msg('Model saved at {}'.format(save_path))


# def resume(cfg, model, optimizer):
#     checkpoint = cfg.base.checkpoint
#     if os.path.exists(checkpoint):
#         print_msg('Loading checkpoint {}'.format(checkpoint))

#         checkpoint = torch.load(checkpoint, map_location='cpu')
#         start_epoch = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])

#         print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint, checkpoint['epoch']))
#         return start_epoch
#     else:
#         print_msg('No checkpoint found at {}'.format(checkpoint))
#         raise FileNotFoundError('No checkpoint found at {}'.format(checkpoint))

# def initialize_dataloader(cfg, train_dataset, val_dataset):
#     batch_size = cfg.train.batch_size
#     num_workers = cfg.train.num_workers
#     pin_memory = cfg.train.pin_memory
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         drop_last=True,
#         pin_memory=pin_memory
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         drop_last=False,
#         pin_memory=pin_memory
#     )

#     return train_loader, val_loader


# def resume(cfg, model, optimizer):
#     checkpoint = cfg.base.checkpoint
#     if os.path.exists(checkpoint):
#         print_msg('Loading checkpoint {}'.format(checkpoint))

#         checkpoint = torch.load(checkpoint, map_location='cpu')
#         start_epoch = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])

#         print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint, checkpoint['epoch']))
#         return start_epoch
#     else:
#         print_msg('No checkpoint found at {}'.format(checkpoint))
#         raise FileNotFoundError('No checkpoint found at {}'.format(checkpoint))

##################################################################################################################################################

# import os
# import math

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

# # Assuming these are your custom utility/loss imports
# from src.utils.func import *
# from src.loss import *
# from src.scheduler import *

# # --- 1. Sharpness-Aware Minimization (SAM) Optimizer Implementation ---
# class SAM(torch.optim.Optimizer):
#     """
#     Implements the Sharpness-Aware Minimization (SAM) optimizer.
#     SAM seeks parameters in flat loss regions for better generalization.
#     It wraps a base optimizer (e.g., AdamW) and performs a two-step update.
#     """
#     def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
#         assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
#         defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
#         super(SAM, self).__init__(params, defaults)

#         self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
#         self.param_groups = self.base_optimizer.param_groups
#         self.defaults.update(self.base_optimizer.defaults)

#     @torch.no_grad()
#     def first_step(self, zero_grad=False):
#         grad_norm = self._grad_norm()
#         for group in self.param_groups:
#             scale = group["rho"] / (grad_norm + 1e-12)
#             for p in group["params"]:
#                 if p.grad is None: continue
#                 self.state[p]["old_p"] = p.data.clone()
#                 e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
#                 p.add_(e_w)
#         if zero_grad: self.zero_grad()

#     @torch.no_grad()
#     def second_step(self, zero_grad=False):
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None: continue
#                 p.data = self.state[p]["old_p"]
#         self.base_optimizer.step()
#         if zero_grad: self.zero_grad()

#     def _grad_norm(self):
#         shared_device = self.param_groups[0]["params"][0].device
#         norm = torch.norm(
#             torch.stack([
#                 ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
#                 for group in self.param_groups for p in group["params"]
#                 if p.grad is not None
#             ]), p=2
#         )
#         return norm

#     def step(self, closure=None):
#         raise NotImplementedError("SAM doesn't support closure for step.")

#     def load_state_dict(self, state_dict):
#         super().load_state_dict(state_dict)
#         self.base_optimizer.param_groups = self.param_groups

# # --- 2. Main Training Function (Plan A with Plotting & Estimator Fix) ---
# def train(cfg, frozen_encoder, model, train_dataset, val_dataset, estimator):
#     device = cfg.base.device

#     backbone_params = [p for n, p in model.named_parameters() if 'cnn_backbone' in n and p.requires_grad]
#     head_params = [p for n, p in model.named_parameters() if 'cnn_backbone' not in n and p.requires_grad]
#     param_groups = [
#         {'params': backbone_params, 'lr': cfg.solver.backbone_lr},
#         {'params': head_params, 'lr': cfg.solver.head_lr}
#     ]

#     optimizer = initialize_optimizer(cfg, param_groups)
#     loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
#     train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset)

#     start_epoch = 0
#     if cfg.base.checkpoint:
#         start_epoch = resume(cfg, model, optimizer)

#     sam_start_epoch = cfg.train.sam_start_epoch
#     max_indicator = 0

#     history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

#     print("--- Starting Training with Plan B (SAM Optimizer) ---")
#     model.train()
#     for epoch in range(start_epoch, cfg.train.epochs):
#         lr = adjust_learning_rate(cfg, optimizer, epoch)

#         if loss_weight_scheduler:
#             weight = loss_weight_scheduler.step()
#             loss_function.weight = weight.to(device)

#         # --- Training Phase ---
#         epoch_loss = 0
#         # ✨ FIX: Reset estimator state before starting training for the epoch
#         estimator.reset()
#         progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.train.epochs}', leave=False) if cfg.base.progress else train_loader

#         for step, train_data in enumerate(progress):
#             if cfg.dataset.preload_path:
#                 X_side, key_states, value_states, y = train_data
#                 key_states, value_states = key_states.to(device), value_states.to(device)
#                 key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
#             else:
#                 X_lpm, X_side, y = train_data
#                 X_lpm = X_lpm.to(device)
#                 with torch.no_grad():
#                     _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

#             X_side, y = X_side.to(device), y.to(device)
#             y = select_target_type(y, cfg.train.criterion)

#             y_pred = model(X_side, key_states, value_states)
#             loss = loss_function(y_pred, y)

#             if epoch >= sam_start_epoch:
#                 loss.backward()
#                 optimizer.first_step(zero_grad=True)
#                 loss_function(model(X_side, key_states, value_states), y).backward()
#                 optimizer.second_step(zero_grad=True)
#             else:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.base_optimizer.step()

#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#             epoch_loss += loss.item()
#             estimator.update(y_pred, y)
#             if cfg.base.progress:
#                 progress.set_postfix(Loss=f'{epoch_loss/(step+1):.4f}', LR=f"{lr:.2e}")

#         # Record training metrics for the epoch
#         avg_train_loss = epoch_loss / len(train_loader)
#         train_scores = estimator.get_scores(4)
#         history['train_loss'].append(avg_train_loss)
#         history['train_acc'].append(train_scores.get(cfg.train.indicator, 0))

#         # --- Validation Phase ---
#         # ✨ FIX: Reset estimator state again before validation for a clean calculation
#         estimator.reset()
#         val_loss, val_scores = eval(cfg, frozen_encoder, model, val_loader, estimator, device, loss_function)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_scores.get(cfg.train.indicator, 0))

#         print(f"\nEpoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_scores.get(cfg.train.indicator, 0):.4f}")
#         print(f"Epoch {epoch+1} Val Loss:   {val_loss:.4f}, Val Acc:   {val_scores.get(cfg.train.indicator, 0):.4f}")

#         indicator = val_scores[cfg.train.indicator]
#         if indicator > max_indicator:
#             max_indicator = indicator
#             save_weights(cfg, model, 'best_validation_weights.pt')

#     save_weights(cfg, model, 'final_weights.pt')
#     print("--- Training finished. Final model saved. ---")

#     save_plots(history, cfg.dataset.save_path)

#     return model

# # --- 3. Helper Function Modifications ---

# def initialize_optimizer(cfg, params):
#     base_optimizer_choice = torch.optim.AdamW
#     print(f"--- Initializing SAM with base optimizer: {base_optimizer_choice.__name__} ---")
#     optimizer = SAM(params, base_optimizer_choice, rho=cfg.solver.rho, adaptive=True, betas=cfg.solver.betas, weight_decay=cfg.solver.weight_decay)
#     return optimizer

# def initialize_loss(cfg, train_dataset):
#     criterion = cfg.train.criterion
#     weight = None
#     loss_weight_scheduler = None
#     if criterion == 'cross_entropy':
#         print(f"--- Using CrossEntropyLoss with label smoothing: {cfg.train.label_smoothing} ---")
#         loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.train.label_smoothing)
#     else:
#         raise NotImplementedError('Not implemented loss function.')
#     return loss, loss_weight_scheduler

# def adjust_learning_rate(cfg, optimizer, epoch):
#     main_lr = cfg.solver.head_lr
#     warmup_epochs = cfg.train.warmup_epochs
#     total_epochs = cfg.train.epochs
#     if epoch < warmup_epochs:
#         lr_scale = (epoch + 1) / warmup_epochs
#     else:
#         progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
#         lr_scale = 0.5 * (1. + math.cos(math.pi * progress))
#     for i, param_group in enumerate(optimizer.param_groups):
#         base_lr = cfg.solver.head_lr if i == 1 else cfg.solver.backbone_lr
#         param_group['lr'] = base_lr * lr_scale
#     return optimizer.param_groups[1]['lr']

# def save_plots(history, save_path):
#     print("--- Generating and saving performance plots... ---")
#     epochs = range(1, len(history['train_loss']) + 1)

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
#     plt.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path, 'loss_plot.png'))
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
#     plt.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
#     plt.close()
#     print(f"Performance plots saved to {save_path}")

# # --- 4. Unchanged & Modified Helper Functions ---

# def initialize_dataloader(cfg, train_dataset, val_dataset):
#     train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
#     return train_loader, val_loader

# def eval(cfg, frozen_encoder, model, dataloader, estimator, device, loss_function):
#     """
#     Evaluates the model on a given dataloader.
#     ✨ FIX: This function no longer resets the estimator. The caller is responsible.
#     """
#     model.eval()
#     torch.set_grad_enabled(False)
#     total_loss = 0.0

#     for test_data in dataloader:
#         if cfg.dataset.preload_path:
#             X_side, key_states, value_states, y = test_data
#             key_states, value_states = key_states.to(device), value_states.to(device)
#             key_states, value_states = key_states.transpose(0, 1), value_states.transpose(0, 1)
#         else:
#             X_lpm, X_side, y = test_data
#             X_lpm = X_lpm.to(device)
#             with torch.no_grad():
#                 _, key_states, value_states = frozen_encoder(X_lpm, interpolate_pos_encoding=True)

#         X_side, y = X_side.to(device), y.to(device)
#         y_true = select_target_type(y, cfg.train.criterion)
#         y_pred = model(X_side, key_states, value_states)

#         loss = loss_function(y_pred, y_true)
#         total_loss += loss.item()
#         estimator.update(y_pred, y_true)

#     model.train()
#     torch.set_grad_enabled(True)
#     avg_loss = total_loss / len(dataloader)
#     scores = estimator.get_scores(6)
#     return avg_loss, scores

# def save_weights(cfg, model, save_name):
#     save_path = os.path.join(cfg.dataset.save_path, save_name)
#     torch.save(model.state_dict(), save_path)
#     print(f'Model weights saved at: {save_path}')


# def resume(cfg, model, optimizer):
#     checkpoint = cfg.base.checkpoint
#     if os.path.exists(checkpoint):
#         print_msg('Loading checkpoint {}'.format(checkpoint))

#         checkpoint = torch.load(checkpoint, map_location='cpu')
#         start_epoch = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])

#         print_msg('Loaded checkpoint {} from epoch {}'.format(checkpoint, checkpoint['epoch']))
#         return start_epoch
#     else:
#         print_msg('No checkpoint found at {}'.format(checkpoint))
#         raise FileNotFoundError('No checkpoint found at {}'.format(checkpoint))
