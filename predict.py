#!/usr/bin/env python
import torch
import torch.nn.functional as F
from model import LightningUNet
import json
import numpy as np
import argparse
import os
# Local imports
from data import RegionalDataset

# Path to config json
config_json = "config.json"


class UnetPredictor():
    def __init__(self, config):
        unet_params = config["model"]["unet"]
        checkpoint = config["predict"]["checkpoint"]["unet"]

        # Load model from checkpoint and set to evaluation mode
        self.model = LightningUNet.load_from_checkpoint(checkpoint, **unet_params)
        self.model.eval()

    def predict(self, image):
        """
        Prediction function
        :param image: Grey 3D input image in format [D, H, W]
        :return: Returns model prediction mask
        """
        image = self.resize_3d_tensor(image)
        image = image.unsqueeze(0)
        return self.model(image)

    @staticmethod
    def resize_3d_tensor(input_tensor, target_size=(256, 256, 256)):
        # Ensure input is a 3D tensor
        if input_tensor.dim() != 3:
            raise ValueError(f"Input tensor must be 3D, got {input_tensor.dim()}D tensor")

        # Add batch and channel dimensions for interpolation
        # Shape becomes [1, 1, depth, height, width]
        x = input_tensor.unsqueeze(0).unsqueeze(0)

        # Resize using trilinear interpolation
        resized = F.interpolate(
            x,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

        # Remove batch and channel dimensions
        # Resulting shape is [target_depth, target_height, target_width]
        return resized.squeeze()


def main():
    # Load config
    with open(config_json) as f:
        config = json.load(f)

    # Get from config
    device = config["train"]["device"]
    train = config['dataset']['train']
    images = os.path.join(config['dataset']['all'], config['dataset']['images'])
    masks = os.path.join(config['dataset']['all'], config['dataset']['masks'])
    annotation_suffix = '.json'
    transforms = config['dataset']['transforms']

    train_dataset = RegionalDataset(images, masks, train + annotation_suffix, transforms=transforms)
    image, mask = train_dataset.__getitem__(0)
    image = torch.as_tensor(image, device=device)

    # Predictor
    predictor = UnetPredictor(config)
    predictions = predictor.predict(image)
    print(predictions)


if __name__ == '__main__':
    main()
