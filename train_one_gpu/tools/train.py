import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
import timm
import numpy as np
from mmengine.config import Config
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from src.dataset.dataset_builder import build_datasets, build_dataloaders
import src.training_procedure_mixed as trainers
from src.models.contrastive import FeatHead
from src.loss_utils import BCEWithLogitsLoss_LS, FocalLoss, BinaryFocalLoss, SupConLoss
from src.init_utils import set_seed, init_wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.py",
    )
    return parser.parse_args()

def train_main(config):
    set_seed(config.dataset.get("val_seed", 42))

    train_ds, val_ds = build_datasets(config)
    train_loader, val_loader = build_dataloaders(
        config, train_ds, val_ds
    )

    model = FeatHead(
        backbone_name=config.model["model_name"],
        pretrained=config.model.get("pretrained", True),
        proj_dim=config.loss.get("supcon_proj_dim", 128)
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(config.device)

    optimizer_cls = getattr(torch.optim, config.optimizer.pop("type"))
    steps       = len(train_loader) * config.trainer["epochs"]
    warmup_steps= int(0.05 * steps)
    optimizer   = optimizer_cls(
        model.parameters(),
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"],
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=steps,
        cycle_mult=1,
        max_lr=config.scheduler["max_lr"],
        min_lr=config.scheduler["min_lr"],
        warmup_steps=warmup_steps,
        gamma=config.scheduler["gamma"],
    )

    # loss
    if config.loss["type"] == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.loss["type"] == "binary_focal":
        criterion = BinaryFocalLoss(
            alpha=config.loss.get("alpha", 0.5),
            gamma=config.loss.get("gamma", 2.0)
        )
    elif config.loss["type"] == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    else:
        criterion = BCEWithLogitsLoss_LS()

    # wandb init
    if config.wandb.get("use", False):
        init_wandb(config=config)

    TrainerCls = getattr(trainers, config.trainer["type"])
    trainer = TrainerCls(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        schedulers=[scheduler],
        config=config,
        use_multihead=config.get("train_multihead", False),
        device=config.device,
        supcon_weight=config.loss.get("supcon_weight", 0.1),
        epochs=config.trainer["epochs"],
        output_name=config.wandb["experiment_name"],
        weights_save_folder=config.trainer["weights_save_folder"],
        mixed_precision= True,
    )

    trainer.run(train_loader, val_loader)

if __name__ == "__main__":
    args   = parse_args()
    config = Config.fromfile(args.config_path)
    train_main(config)