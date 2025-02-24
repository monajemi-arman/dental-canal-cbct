#!/usr/bin/env python
import os
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from data import BaseDataset, RegionalDataset


def get_image_suffix(image_dir):
    for file in os.listdir(image_dir):
        if file.endswith(".npy"):
            return ".npy"
        elif file.endswith(".npz"):
            return ".npz"
    raise ValueError("No .npy or .npz files found in the image directory.")


def reconstruct_volumes(image_dir, mask_dir, annotation_file, output_root, train_ratio=0.8):
    dataset = RegionalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        annotation_file=annotation_file,
        image_suffix=get_image_suffix(image_dir),
        transforms=None,
    )

    os.makedirs(output_root, exist_ok=True)
    image_keys = [dataset.image_id_to_key(image_id) for image_id in range(len(dataset.images))]
    processed_keys = []

    for key in image_keys:
        output_dir = os.path.join(output_root, key)
        img_path = os.path.join(output_dir, f"{key}_flare.nii")
        mask_path = os.path.join(output_dir, f"{key}_seg.nii")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            print(f"Skipping {key}, already processed.")
            processed_keys.append(key)
            continue

        print(f"Processing {key}...")
        image_path = os.path.join(dataset.image_dir, key + dataset.image_suffix)
        original_image = dataset.read_image(image_path)
        original_shape = original_image.shape

        image_volume = np.zeros(original_shape, dtype=np.float32)
        mask_volume = np.zeros(original_shape, dtype=np.uint8)

        box_ids = [box_id for box_id, img_id in dataset.box_id_to_image_id.items() if
                   dataset.image_id_to_key(img_id) == key]

        for box_id in box_ids:
            image_patch, mask_patch = dataset[box_id]
            image_patch = image_patch.squeeze().numpy()
            mask_patch = mask_patch.squeeze().numpy().astype(np.uint8)

            bbox = dataset.boxes[box_id]
            x, y, z, w, h, depth = [int(val) for val in bbox]
            original_crop_size = (depth, h, w)

            target_d, target_h, target_w = dataset.target_size
            scale_factors = (
                original_crop_size[0] / target_d,
                original_crop_size[1] / target_h,
                original_crop_size[2] / target_w,
            )

            resized_image = zoom(image_patch, scale_factors, order=3)
            resized_mask = zoom(mask_patch.astype(float), scale_factors, order=0)
            resized_mask = resized_mask.round().astype(np.uint8)

            z_end, y_end, x_end = z + resized_image.shape[0], y + resized_image.shape[1], x + resized_image.shape[2]
            z_end, y_end, x_end = min(z_end, original_shape[0]), min(y_end, original_shape[1]), min(x_end,
                                                                                                    original_shape[2])

            current_depth, current_h, current_w = z_end - z, y_end - y, x_end - x
            resized_image, resized_mask = resized_image[:current_depth, :current_h, :current_w], resized_mask[
                                                                                                 :current_depth,
                                                                                                 :current_h, :current_w]

            image_volume[z:z_end, y:y_end, x:x_end] = resized_image
            mask_volume[z:z_end, y:y_end, x:x_end] = resized_mask

        os.makedirs(output_dir, exist_ok=True)

        affine = np.eye(4)
        nib.save(nib.Nifti1Image(image_volume, affine), img_path)
        nib.save(nib.Nifti1Image(mask_volume, affine), mask_path)
        print(f"Saved NIfTI files for {key} in {output_dir}")
        processed_keys.append(key)

    np.random.shuffle(processed_keys)
    split_idx = int(len(processed_keys) * train_ratio)
    train_keys, val_keys = processed_keys[:split_idx], processed_keys[split_idx:]

    with open(os.path.join(output_root, "train.txt"), "w") as f:
        f.writelines(f"{key}\n" for key in train_keys)

    with open(os.path.join(output_root, "val.txt"), "w") as f:
        f.writelines(f"{key}\n" for key in val_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct 3D volumes from patches.")
    parser.add_argument("-i", "--image_dir", type=str, required=True, help="Directory containing image patches.")
    parser.add_argument("-m", "--mask_dir", type=str, required=True, help="Directory containing mask patches.")
    parser.add_argument("-a", "--annotation_file", type=str, required=True, help="Path to the annotation file.")
    parser.add_argument("-o", "--output_root", type=str, required=True,
                        help="Root directory to save reconstructed volumes.")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.8,
                        help="Train-validation split ratio (default: 0.8).")

    args = parser.parse_args()

    reconstruct_volumes(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        annotation_file=args.annotation_file,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
    )
