#!/usr/bin/env python

import json
import torch
from monai.losses import DiceCELoss
from lightning import LightningModule
from monai.networks.nets import UNet
from torch.optim import Adam

# Path to config file
config_json = 'config.json'


class LightningUNet(LightningModule):
    def __init__(self, **unet_params):
        super().__init__()
        self.model = UNet(**unet_params).to("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = DiceCELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_hat = self.model(x)
        train_loss = self.loss_fn(y_hat, y)
        self.log('train_loss', train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        y_hat = self.model(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


def main():
    # Load config from file
    with open(config_json) as f:
        config = json.load(f)

    # Load model with custom config parameters
    unet_params = config['model']['unet']
    model = LightningUNet(**unet_params)


if __name__ == '__main__':
    main()
