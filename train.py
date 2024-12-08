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
        patience=10,
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=1, collate_fn=mosaic_collate)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=1, collate_fn=mosaic_collate)

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

def mosaic_collate(
        batch: List[Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        padding_value: float = 0.0,
        crop_strategy: str = 'random',
        maintain_distribution: bool = True,
        verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Advanced mosaic collate function for 3D grayscale images and masks.

    Supports various crop strategies, handles different input types,
    and provides flexible volume processing.

    Args:
        batch (List[Tuple]): List of (image, mask) pairs
        target_shape (Tuple[int, int, int]): Desired output volume size
        padding_value (float): Value used for padding/filling empty spaces
        crop_strategy (str): Cropping method ('random', 'center', 'adaptive')
        maintain_distribution (bool): Try to preserve original data distribution
        verbose (bool): Print additional processing information

    Returns:
        Tuple of batched images and masks, each of shape (N, 1, D, H, W)

    Raises:
        ValueError: For incompatible input shapes or types
    """
    # Input validation and preprocessing
    processed_images, processed_masks = [], []

    # Statistical tracking for adaptive strategies
    original_volumes_stats: Dict[str, List[int]] = {
        'depths': [],
        'heights': [],
        'widths': []
    }

    # Validate crop strategy
    valid_strategies = ['random', 'center', 'adaptive']
    if crop_strategy not in valid_strategies:
        raise ValueError(f"Invalid crop strategy. Must be one of {valid_strategies}")

    def safe_convert_to_tensor(data: Union[torch.Tensor, np.ndarray, Any]) -> torch.Tensor:
        """
        Safely convert input to a torch tensor with float32 dtype

        Args:
            data: Input data to convert

        Returns:
            Converted torch tensor
        """
        if isinstance(data, torch.Tensor):
            return data.float()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            try:
                return torch.tensor(data, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Cannot convert input to tensor: {e}")

    def compute_crop_bounds(
            original_size: int,
            target_size: int,
            strategy: str = 'random',
            stats: Optional[List[int]] = None
    ) -> Tuple[int, int]:
        """
        Compute crop start and end indices based on strategy.

        Args:
            original_size: Size of the original dimension
            target_size: Desired size of the dimension
            strategy: Cropping strategy
            stats: Optional statistics for adaptive strategy

        Returns:
            Tuple of (start, end) indices for cropping
        """
        if strategy == 'center':
            start = max(0, (original_size - target_size) // 2)
            end = start + target_size
        elif strategy == 'random':
            start = random.randint(0, max(0, original_size - target_size))
            end = start + target_size
        elif strategy == 'adaptive' and stats is not None:
            # More intelligent cropping based on volume distribution
            mean_size = np.mean(stats)
            std_size = np.std(stats)

            potential_start = max(0, int(mean_size - std_size / 2))
            start = random.randint(potential_start, min(original_size - target_size, potential_start + int(std_size)))
            end = start + target_size
        else:
            raise ValueError(f"Unknown crop strategy or missing statistics: {strategy}")

        return start, end

    # First pass: Validate inputs and collect statistics
    for idx, (image, mask) in enumerate(batch):
        image = safe_convert_to_tensor(image)
        mask = safe_convert_to_tensor(mask)

        # For grayscale 3D volumes, ensure single-channel input
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Validate tensor dimensions
        if image.ndim != 4 or mask.ndim != 4:
            raise ValueError(f"Volume {idx} must be 4D: (1, D, H, W)")

        # Ensure single channel
        if image.shape[0] != 1 or mask.shape[0] != 1:
            raise ValueError(f"Volume {idx} must be single-channel")

        # Collect statistics
        original_volumes_stats['depths'].append(image.shape[1])
        original_volumes_stats['heights'].append(image.shape[2])
        original_volumes_stats['widths'].append(image.shape[3])

        if verbose:
            print(f"Volume {idx}: Shape {image.shape}, Mask Shape {mask.shape}")

    # Second pass: Process volumes
    for image, mask in batch:
        # Convert to torch tensors
        image = safe_convert_to_tensor(image)
        mask = safe_convert_to_tensor(mask)

        # Ensure single-channel and 4D tensors
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)

        # Initialize target tensors (single channel)
        padded_image = torch.full((1, *target_shape),
                                  padding_value, dtype=torch.float32)
        padded_mask = torch.full((1, *target_shape),
                                 padding_value, dtype=torch.float32)

        # Compute bounds for each dimension
        bounds = [
            compute_crop_bounds(
                dim,
                target_size,
                crop_strategy,
                original_volumes_stats[f"{['depths', 'heights', 'widths'][i]}"]
            )
            for i, (dim, target_size) in enumerate(zip(image.shape[1:], target_shape))
        ]

        # Crop and insert
        d_start, d_end = bounds[0]
        h_start, h_end = bounds[1]
        w_start, w_end = bounds[2]

        # Determine actual crop sizes
        crop_d = min(d_end - d_start, image.shape[1])
        crop_h = min(h_end - h_start, image.shape[2])
        crop_w = min(w_end - w_start, image.shape[3])

        # Insert cropped region
        padded_image[0, :crop_d, :crop_h, :crop_w] = image[
                                                     0, d_start:d_start + crop_d,
                                                     h_start:h_start + crop_h,
                                                     w_start:w_start + crop_w
                                                     ]
        padded_mask[0, :crop_d, :crop_h, :crop_w] = mask[
                                                    0, d_start:d_start + crop_d,
                                                    h_start:h_start + crop_h,
                                                    w_start:w_start + crop_w
                                                    ]

        processed_images.append(padded_image)
        processed_masks.append(padded_mask)

    # Final stacking
    return torch.stack(processed_images), torch.stack(processed_masks)

if __name__ == '__main__':
    main()
