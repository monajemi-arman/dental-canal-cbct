#!/usr/bin/env python
import torch
import torch.nn.functional as F
from model import LightningUNet
import json
import numpy as np
from skimage.transform import resize
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import RadioButtons
from monai.inferers import sliding_window_inference
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

    def predict(self, image, revert_size=False):
        """
        Prediction function
        :param image: Grey 3D input image in format [D, H, W]
        :return: Returns model prediction mask
        """
        orig_shape = image.shape
        # Resize to expected size of model
        image = self.resize_3d_tensor(image)
        image = image.unsqueeze(0)
        # Pass to model
        predictions = self.model(image)
        pred_mask = self.process_model_output(predictions)
        if revert_size:
            # Revert back to original size
            pred_mask = self.resize_3d_tensor(pred_mask, target_size=orig_shape, mode='nearest')
        return pred_mask

    @staticmethod
    def process_model_output(predictions, threshold=0.5):
        binary_mask = (predictions > threshold).float().detach().cpu()
        binary_mask = binary_mask.squeeze(0).squeeze(0)
        return binary_mask

    @staticmethod
    def resize_3d_tensor(input_tensor, target_size=(128, 128, 128), mode='trilinear'):
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
            mode=mode,
            align_corners=False if mode in ['trilinear', 'bilinear'] else None
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

    train_dataset = RegionalDataset(images, masks, train + annotation_suffix, transforms=transforms,
                                    target_size=config['dataset']['crop_target_size'])
    image, mask = train_dataset.__getitem__(0)
    image = torch.as_tensor(image, device=device)

    # Predictor
    predictor = UnetPredictor(config)
    predictions = predictor.predict(image)
    display_most_important_layer(mask, predictions)


def display_most_important_layer(mask, pred_mask):
    """
    Display the most important layer of a 3D grey mask along with its predicted mask,
    with a toggle button to switch between the most important layer in pred_mask itself
    or the layer matching the most important layer in the original mask.

    Parameters:
        mask (torch.Tensor or np.ndarray): A 3D torch tensor or numpy array (shape: depth x height x width) with values 0 and 1.
        pred_mask (torch.Tensor or np.ndarray): A 3D torch tensor or numpy array (shape: depth x height x width) with predicted mask values.

    Returns:
        None
    """
    def validate_and_convert(input_mask):
        if isinstance(input_mask, np.ndarray):
            if input_mask.ndim != 3:
                raise ValueError("Input must be a 3D numpy array.")
            return input_mask
        elif isinstance(input_mask, torch.Tensor):
            if input_mask.dim() != 3:
                raise ValueError("Input must be a 3D torch tensor.")
            return input_mask.cpu().numpy()
        else:
            raise TypeError("Input must be either a torch.Tensor or a numpy.ndarray.")

    mask = validate_and_convert(mask)
    pred_mask = validate_and_convert(pred_mask)

    # Resize smaller mask to match the larger mask
    if mask.shape != pred_mask.shape:
        if mask.shape > pred_mask.shape:
            pred_mask = np.array([resize(layer, mask.shape[1:], preserve_range=True) for layer in pred_mask])
        else:
            mask = np.array([resize(layer, pred_mask.shape[1:], preserve_range=True) for layer in mask])

    # Calculate importance for each layer
    mask_layer_importance = mask.sum(axis=(1, 2))
    pred_mask_layer_importance = pred_mask.sum(axis=(1, 2))

    most_important_layer_idx_mask = mask_layer_importance.argmax()
    most_important_layer_idx_pred_mask = pred_mask_layer_importance.argmax()

    most_important_layer = mask[most_important_layer_idx_mask]
    most_important_pred_layer = pred_mask[most_important_layer_idx_pred_mask]
    matching_pred_layer = pred_mask[most_important_layer_idx_mask]

    # Interactive toggle functionality
    def update_display(label):
        if label == 'Predicted (Self) Importance':
            im_pred.set_data(most_important_pred_layer)
            ax_pred.set_title(f"Predicted Mask\nMost Important Layer (Index: {most_important_layer_idx_pred_mask})")
        elif label == 'Predicted (Matching Original)':
            im_pred.set_data(matching_pred_layer)
            ax_pred.set_title(f"Predicted Mask\nLayer Matching Original Mask (Index: {most_important_layer_idx_mask})")
        fig.canvas.draw_idle()

    # Create the plot
    fig, (ax_orig, ax_pred) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.3)

    im_orig = ax_orig.imshow(most_important_layer, cmap='gray')
    ax_orig.set_title(f"Original Mask\nMost Important Layer (Index: {most_important_layer_idx_mask})")
    ax_orig.axis('off')

    im_pred = ax_pred.imshow(most_important_pred_layer, cmap='gray')
    ax_pred.set_title(f"Predicted Mask\nMost Important Layer (Index: {most_important_layer_idx_pred_mask})")
    ax_pred.axis('off')

    # Add radio buttons
    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, ('Predicted (Self) Importance', 'Predicted (Matching Original)'))

    radio.on_clicked(update_display)
    plt.show()


if __name__ == '__main__':
    main()
