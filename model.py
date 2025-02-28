#!/usr/bin/env python
import json
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from lightning import LightningModule
from monai.networks.nets import UNet
from torch.optim import Adam

config_json = 'config.json'


class LightningDualUNet(LightningModule):
    def __init__(self, consistency_weight=0.1, **unet_params):
        super().__init__()
        self.model1 = UNet(**unet_params)
        self.model2 = UNet(**unet_params)
        self.loss_fn = DiceCELoss()
        self.consistency_weight = consistency_weight
        self.lr = 0.0001

    def forward(self, x):
        y1 = self.model1(x)
        y2 = self.model2(x)
        return (y1 + y2) / 2

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat1 = self.model1(x)
        y_hat2 = self.model2(x)

        # Supervised loss (only for labeled data)
        if y is not None:
            y = y.long()
            loss1 = self.loss_fn(y_hat1, y)
            loss2 = self.loss_fn(y_hat2, y)
            supervised_loss = loss1 + loss2
        else:
            supervised_loss = 0

        # Consistency loss (for both labeled and unlabeled data)
        y_hat1_soft = torch.softmax(y_hat1, dim=1)
        y_hat2_soft = torch.softmax(y_hat2, dim=1)
        consistency_loss = F.mse_loss(y_hat1_soft, y_hat2_soft)

        # Total loss
        total_loss = supervised_loss + self.consistency_weight * consistency_loss

        self.log_dict({
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'total_train_loss': total_loss
        }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        y_hat1 = self.model1(x)
        y_hat2 = self.model2(x)

        val_loss1 = self.loss_fn(y_hat1, y)
        val_loss2 = self.loss_fn(y_hat2, y)
        avg_val_loss = (val_loss1 + val_loss2) / 2

        self.log_dict({
            'val_loss1': val_loss1,
            'val_loss2': val_loss2,
            'val_loss': avg_val_loss
        }, prog_bar=True)

        return avg_val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def main():
    with open(config_json) as f:
        config = json.load(f)

    unet_params = config['model']['unet']
    consistency_weight = config['model'].get('consistency_weight', 0.1)

    model = LightningDualUNet(
        consistency_weight=consistency_weight,
        **unet_params
    )


if __name__ == '__main__':
    main()
