#!/usr/bin/env python
import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

from data import BaseDataset, RegionalDataset


def reconstruct_volumes(image_dir, mask_dir, annotation_file, output_root):
    """
    Reconstruct 3D volumes from patches and save them as NIfTI files.
    """
    dataset = RegionalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        annotation_file=annotation_file,
        image_suffix=".npy",  # Adjust as needed
        transforms=None,  # Adjust as needed
    )

    os.makedirs(output_root, exist_ok=True)

    # Process each image_id
    for image_id in range(len(dataset.images)):
        key = dataset.image_id_to_key(image_id)
        print(f"Processing {key}...")

        # Load the original image to get its dimensions
        image_path = os.path.join(dataset.image_dir, key + dataset.image_suffix)
        original_image = dataset.read_image(image_path)
        original_shape = original_image.shape

        image_volume = np.zeros(original_shape, dtype=np.float32)
        mask_volume = np.zeros(original_shape, dtype=np.uint8)

        # Get all box IDs for this image_id
        box_ids = [
            box_id for box_id, img_id in dataset.box_id_to_image_id.items() if img_id == image_id
        ]

        # Process each patch
        for box_id in box_ids:
            image_patch, mask_patch = dataset[box_id]
            image_patch = image_patch.squeeze().numpy()
            mask_patch = mask_patch.squeeze().numpy().astype(np.uint8)

            # Get the bounding box and cast to integers
            bbox = dataset.boxes[box_id]
            x, y, z, w, h, depth = [int(val) for val in bbox]  # Fix: Cast to integers
            original_crop_size = (depth, h, w)

            # Resize the patch back to its original size
            target_d, target_h, target_w = dataset.target_size
            scale_factors = (
                original_crop_size[0] / target_d,
                original_crop_size[1] / target_h,
                original_crop_size[2] / target_w,
            )

            resized_image = zoom(image_patch, scale_factors, order=3)
            resized_mask = zoom(mask_patch.astype(float), scale_factors, order=0)
            resized_mask = resized_mask.round().astype(np.uint8)

            # Calculate the end coordinates for the patch
            z_end = z + resized_image.shape[0]
            y_end = y + resized_image.shape[1]
            x_end = x + resized_image.shape[2]

            # Ensure the patch fits within the original volume
            if z_end > original_shape[0]:
                z_end = original_shape[0]
            if y_end > original_shape[1]:
                y_end = original_shape[1]
            if x_end > original_shape[2]:
                x_end = original_shape[2]

            # Trim the resized patch if necessary
            current_depth = z_end - z
            current_h = y_end - y
            current_w = x_end - x

            if (current_depth, current_h, current_w) != resized_image.shape:
                resized_image = resized_image[:current_depth, :current_h, :current_w]
                resized_mask = resized_mask[:current_depth, :current_h, :current_w]

            # Place the patch into the volume
            image_volume[z:z_end, y:y_end, x:x_end] = resized_image
            mask_volume[z:z_end, y:y_end, x:x_end] = resized_mask

        # Save the reconstructed volumes as NIfTI files
        output_dir = os.path.join(output_root, key)
        os.makedirs(output_dir, exist_ok=True)

        affine = np.eye(4)
        img_nii = nib.Nifti1Image(image_volume, affine)
        mask_nii = nib.Nifti1Image(mask_volume, affine)

        img_path = os.path.join(output_dir, f"{key}_flare.nii")
        mask_path = os.path.join(output_dir, f"{key}_seg.nii")

        nib.save(img_nii, img_path)
        nib.save(mask_nii, mask_path)
        print(f"Saved NIfTI files for {key} in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D volumes from patches.")
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Directory containing image patches.")
    parser.add_argument("-m", "--mask_dir", type=str, required=True, help="Directory containing mask patches.")
    parser.add_argument("-a", "--annotation_file", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("-o", "--output_root", type=str, required=True, help="Root directory to save reconstructed volumes.")

    args = parser.parse_args()

    reconstruct_volumes(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        annotation_file=args.annotation_file,
        output_root=args.output_root,
    )