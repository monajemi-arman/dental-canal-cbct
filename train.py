#!/usr/bin/env python
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
import json
# Local imports
from model import LightningUNet
from data import Dataset

# Load config
config_json = "config.json"
with open("config.json") as f:
    config = json.load(f)

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min'
)

# Training
trainer = Trainer(
    max_epochs=100,
    accelerator='gpu',
    callbacks=[early_stop_callback]
)

# Dataset
train_dataset = Dataset(config['dataset']['train'])
val_dataset = Dataset(config['dataset']['val'])

# Dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Load model with custom config parameters
unet_params = config['model']['unet']
model = LightningUNet(**unet_params)

# Trainer
trainer = Trainer(
    max_epochs=config['train']['max_epochs'],
    accelerator=config['train']['accelerator'],
    callbacks=[early_stop_callback]
)

# Training start
trainer.fit(model, train_loader, val_loader)