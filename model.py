#!/usr/bin/env python
from lightning import LightningModule
from monai.networks.nets import UNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class LightningUNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet(dimensions=3, in_channels=1, out_channels=2)
        self.loss_fn = CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)


