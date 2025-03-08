#!/usr/bin/env python
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.losses import DiceCELoss
from lightning import LightningModule
from monai.networks.nets import UNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

config_json = 'config.json'


class LightningDualUNet(LightningModule):
    def __init__(self, consistency_weight=0.1, learning_rate=0.0001, consistency_rampup=200, **unet_params):
        super().__init__()
        self.model1 = UNet(**unet_params)
        self.model2 = UNet(**unet_params)
        self.loss_fn = DiceCELoss()
        self.consistency_weight = consistency_weight
        self.lr = learning_rate
        self.consistency_rampup = consistency_rampup
        self.use_cpu_offload = False  # Flag to enable CPU offloading if GPU memory is insufficient

        # Initialize model weights
        self.model1 = kaiming_normal_init_weight(self.model1)
        self.model2 = xavier_normal_init_weight(self.model2)

    def forward(self, x):
        if self.use_cpu_offload:
            x1 = x.to("cuda")
            x2 = x.to("cpu")
            y1 = self.model1(x1)
            y2 = self.model2(x2).to("cuda")
            return (y1 + y2) / 2
        else:
            y1 = self.model1(x)
            y2 = self.model2(x)
            return (y1 + y2) / 2

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.use_cpu_offload:
            x1 = x.to("cuda")
            x2 = x.to("cpu")
            y_hat1 = self.model1(x1)
            y_hat2 = self.model2(x2).to("cuda")
        else:
            y_hat1 = self.model1(x)
            y_hat2 = self.model2(x)

        y_hat1_soft = torch.softmax(y_hat1, dim=-1)
        y_hat2_soft = torch.softmax(y_hat2, dim=-1)

        y = y.long().to("cuda" if not self.use_cpu_offload else y_hat1.device)

        # Supervised loss (CE + Dice)
        loss1 = 0.5 * (self.ce_loss(y_hat1, y.to(torch.float)) + self.dice_loss(y_hat1_soft, y.unsqueeze(1)))
        loss2 = 0.5 * (self.ce_loss(y_hat2, y.to(torch.float)) + self.dice_loss(y_hat2_soft, y.unsqueeze(1)))
        supervised_loss = loss1 + loss2

        # Pseudo-supervision for consistency
        pseudo_outputs1 = torch.argmax(y_hat1_soft.detach(), dim=-1, keepdim=False)
        pseudo_outputs2 = torch.argmax(y_hat2_soft.detach(), dim=-1, keepdim=False)

        pseudo_supervision1 = self.ce_loss(y_hat1, pseudo_outputs2)
        pseudo_supervision2 = self.ce_loss(y_hat2, pseudo_outputs1)

        # Consistency loss
        consistency_loss = pseudo_supervision1 + pseudo_supervision2

        # Dynamic consistency weight
        self.consistency_weight = self.get_current_consistency_weight(self.current_epoch)

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

        if self.use_cpu_offload:
            x1 = x.to("cuda")
            x2 = x.to("cpu")
            y_hat1 = self.model1(x1)
            y_hat2 = self.model2(x2).to("cuda")
        else:
            y_hat1 = self.model1(x)
            y_hat2 = self.model2(x)

        y = y.long().to("cuda" if not self.use_cpu_offload else y_hat1.device)
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
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def on_train_start(self):
        # Check GPU memory and enable CPU offloading if necessary
        try:
            # Test memory usage with a dummy tensor
            dummy = torch.randn(2, 1, 64, 64, 64).to("cuda")
            _ = self.model1(dummy)
            _ = self.model2(dummy)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                self.use_cpu_offload = True
                self.model2 = self.model2.to("cpu")
                print("GPU memory insufficient. Enabled CPU offloading for model2.")
            else:
                raise e

    def ce_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def dice_loss(self, pred, target):
        smooth = 1.
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - (
                (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()

    def get_current_consistency_weight(self, epoch):
        return self.consistency_weight * sigmoid_rampup(epoch, self.consistency_rampup)


def main():
    with open(config_json) as f:
        config = json.load(f)

    unet_params = config['model']['unet']
    consistency_weight = config['model'].get('consistency_weight', 0.1)

    model = LightningDualUNet(
        consistency_weight=consistency_weight,
        **unet_params
    )


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


if __name__ == '__main__':
    main()
