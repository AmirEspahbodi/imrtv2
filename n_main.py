# image_retrieval/train.py

import torch
import os
import sys
from pathlib import Path
import hydra
import argparse

# --- Import project components ---
from src.config import cfg
from src.data.dataset import CarAccessoriesDataset, get_transforms
from src.model.models import CoAtNetSideViTClassifier_4
from src.losses.proxy_anchor import ProxyAnchorLoss
from src.engine.core import Trainer
from src.model.builder import generate_model
from data.builder import generate_dataset

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(hydra_cfg):
    """Main function to orchestrate the training process."""
    
    # --- Setup ---
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for saving checkpoints
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # --- Data Pipeline ---
    train_transform = get_transforms(cfg.dataset.image_size)
    val_transform = get_transforms(cfg.dataset.image_size)
    
    # Initialize datasets (you'll need to provide key_value_generator)
    train_dataset = CarAccessoriesDataset(
        hydra_cfg.dataset.preload_path,
        data_path='/home/amirh/work/reza_imrt/imrtV2/car_accessories_dataset/train', 
        transform=train_transform
    )
    val_dataset = CarAccessoriesDataset(
        hydra_cfg.dataset.preload_path,
        data_path='/home/amirh/work/reza_imrt/imrtV2/car_accessories_dataset/validation', 
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True
    )

    # --- Model, Loss, Optimizer ---
    frozen_encoder, side_vit_model1 = generate_model(hydra_cfg)
    frozen_encoder2, side_vit_model2 = generate_model(hydra_cfg)
    del frozen_encoder2
    
    classifier_with_side_vits = CoAtNetSideViTClassifier_4(
        side_vit1=side_vit_model1,
        side_vit2=side_vit_model2,
        cfg=hydra_cfg,
    ).to(hydra_cfg.base.device)
    loss_fn = ProxyAnchorLoss(
        num_classes=cfg.loss.num_classes,
        embedding_dim=cfg.loss.embedding_dim,
        margin=cfg.loss.margin,
        alpha=cfg.loss.alpha
    ).to(device)
    
    
    loss_fn = ProxyAnchorLoss(
        num_classes=cfg.loss.num_classes,
        embedding_dim=cfg.loss.embedding_dim,  # Now 192
        margin=cfg.loss.margin,
        alpha=cfg.loss.alpha
    ).to(device)
    
    # Combine model and loss parameters for the optimizer
    params = list(classifier_with_side_vits.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params, 
        lr=cfg.train.learning_rate, 
        weight_decay=cfg.train.weight_decay
    )
    
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * cfg.train.epochs
    )
    scaler = torch.cuda.amp.GradScaler()

    # --- Training ---
    trainer = Trainer(
        model=classifier_with_side_vits,
        frozen_model=frozen_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        cfg=cfg,
        scheduler=scheduler,
        scaler=scaler
    )
    
    best_accuracy = 0.0
    for epoch in range(1, cfg.train.epochs + 1):
        print(f"\n--- Epoch {epoch}/{cfg.train.epochs} ---")
        
        train_loss = trainer.train_one_epoch()
        eval_metrics = trainer.evaluate()
        
        print(f"Epoch {epoch} Summary: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {eval_metrics['val_loss']:.4f}, "
              f"Proxy Accuracy: {eval_metrics['proxy_accuracy']:.4f}")
              
        # --- Checkpointing ---
        if eval_metrics['proxy_accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['proxy_accuracy']
            print(f"New best accuracy! Saving model to {checkpoint_dir}/best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_with_side_vits.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'accuracy': best_accuracy
            }, checkpoint_dir / "best_model.pth")

if __name__ == "__main__":
    import os
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    parser = argparse.ArgumentParser(
        description="Run training with custom arguments for Hydra."
    )
    parser.add_argument(
        "--btl",
        "--backbone_trainable_layers",
        type=int,
        nargs="*",
        help="List of backbone layers to train (e.g., 1 2 3 4).",
    )
    parser.add_argument(
        "--v1fs",
        "--vit1_feature_strame",
        type=int,
        nargs="+",
        help="Two numbers for ViT1 feature stride, each between 1 and 4 (e.g., 1 4).",
    )
    parser.add_argument(
        "--v2fs",
        "--vit2_feature_strame",
        type=int,
        nargs="+",
        help="Two numbers for ViT2 feature stride, each between 1 and 4 (e.g., 2 3).",
    )

    args, unknown_args = parser.parse_known_args()

    hydra_overrides = []
    if args.btl is not None:
        btl_str = "[" + ",".join(map(str, args.btl)) + "]"
        hydra_overrides.append(f"+network.backbone_trainable_layers={btl_str}")

    if args.v1fs is not None:
        v1fs_str = "[" + ",".join(map(str, args.v1fs)) + "]"
        hydra_overrides.append(f"+network.vit1_feature_strame={v1fs_str}")

    if args.v2fs is not None:
        v2fs_str = "[" + ",".join(map(str, args.v2fs)) + "]"
        hydra_overrides.append(f"+network.vit2_feature_strame={v2fs_str}")


    model_id = f"btl{''.join(map(str, args.btl))}_v1fs{''.join(map(str, args.v1fs))}_v2fs{''.join(map(str, args.v2fs))}"
    hydra_overrides.append(f"+network.model_id={model_id}")

    sys.argv = [sys.argv[0]] + hydra_overrides + unknown_args

    main()
