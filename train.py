#!/usr/bin/env python
import json
import os
import random
from typing import List, Tuple, Union, Dict, Optional, Any
import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from data import RegionalDataset
# Local imports
from model import LightningUNet

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
    transforms = config['dataset']['transforms']
    compressed = config['dataset']['compressed']
    if compressed:
        image_suffix = '.npz'
    else:
        image_suffix = '.npy'

    # Dataset instances
    train_dataset = RegionalDataset(images, masks, train + annotation_suffix, transforms=transforms)
    val_dataset = RegionalDataset(images, masks, val + annotation_suffix, transforms=transforms)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=1)

    # Load model with custom config parameters
    unet_params = config['model']['unet']
    model = LightningUNet(**unet_params)

    # Trainer
    trainer = Trainer(
        enable_progress_bar=True,
        max_epochs=config['train']['max_epochs'],
        accelerator=config['train']['accelerator'],
        callbacks=[early_stop_callback]
    )

    # Training start
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()
