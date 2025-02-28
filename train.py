#!/usr/bin/env python
import json
import os
import random
from typing import List, Tuple, Union, Dict, Optional, Any
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping
# Local imports
from data import RegionalDataset
from model import LightningDualUNet

# Path to config
config_json = "config.json"


def main():
    # Load config
    with open("config.json") as f:
        config = json.load(f)

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config['train']['early_stop_callback']['patience'],
        mode='min'
    )

    # Training
    trainer = Trainer(
        max_epochs=config['train']['max_epochs'],
        accelerator=config['train']['accelerator'],
        callbacks=[early_stop_callback]
    )

    # Dataset paths
    train = config['dataset']['train']
    val = config['dataset']['val']
    images = os.path.join(config['dataset']['all'], config['dataset']['images'])
    masks = os.path.join(config['dataset']['all'], config['dataset']['masks'])
    annotation_suffix = '.json'
    batch_size = config['train']['batch_size']
    transforms = config['dataset']['transforms']
    compressed = config['dataset']['compressed']
    if compressed:
        image_suffix = '.npz'
    else:
        image_suffix = '.npy'

    # Dataset instances
    train_dataset = RegionalDataset(images, masks, train + annotation_suffix, transforms=transforms,
                                    target_size=config['dataset']['crop_target_size'])
    val_dataset = RegionalDataset(images, masks, val + annotation_suffix, transforms=transforms,
                                  target_size=config['dataset']['crop_target_size'])

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

    # Load model with custom config parameters
    unet_params = config['model']['unet']
    model = LightningDualUNet(**unet_params)

    # Trainer
    trainer = Trainer(
        enable_progress_bar=True,
        max_epochs=config['train']['max_epochs'],
        accelerator=config['train']['accelerator'],
        callbacks=[early_stop_callback]
    )

    # Find the best learning rate
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, train_loader)
    suggested_lr = lr_finder.suggestion()
    model.lr = suggested_lr
    print(f"Set learning rate: {suggested_lr}")


    # Training start
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
