#!/usr/bin/env python
from lightning import LightningModule
from monai.networks.nets import UNet
from torch.nn import CrossEntropyLoss
import torch
import json
from torch.optim import Adam

# Path to config file
config_json = 'config.json'


class LightningUNet(LightningModule):
    def __init__(self, **unet_params):
        super().__init__()
        self.model = UNet(**unet_params).to("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long()  # Remove channel dimension and convert to LongTensor
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze(1).long()  # Remove channel dimension and convert to LongTensor
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
