#!/usr/bin/env python
import json
import os
import re
import numpy as np
from collections import Counter
from monai.transforms import LoadImage, Compose
from scipy.ndimage import zoom

# Path to config
config_json = 'config.json'


def main():
    # Load config
    with open(config_json) as f:
        config = json.load(f)

    # Load parameters from config
    compressed = config['dataset']['compressed']
    dataset = config['dataset']['raw']
    output_all = config['dataset']['all']
    output_images = os.path.join(config['dataset']['all'], 'images')
    output_masks = os.path.join(config['dataset']['all'], 'masks')
    output_coco = config['dataset']['all'] + '.json'

    # Directories should exist
    for directory in output_all, output_images, output_masks:
        os.makedirs(directory, exist_ok=True)

    # Go through the raw dataset folder
    masks = []
    i = 0
    for first in os.listdir(dataset):
        i += 1
        first = os.path.join(dataset, first)
        for second in os.listdir(first):
            second = os.path.join(first, second)

            # Save image
            if second.endswith(config['dataset']['dcm_directory_suffix']):
                image_data = process_dcm_directory(second, config['dataset']['dcm_file_pattern'])
                save_np(os.path.join(output_images, str(i)), image_data, compressed)

            # Get file suffix
            second_parts = os.path.splitext(second)

            # Gather mask if seen
            if second_parts[-1] == config['dataset']['mask_suffix']:
                masks.append(
                    read_image(second)
                )

        # Save mask
        masks = sanitize_masks(masks)
        combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
        save_np(os.path.join(output_masks, str(i)), combined_mask, compressed)


def process_dcm_directory(directory, pattern):
    dcm_files_dict = {}
    for file in os.listdir(directory):
        found = re.search(pattern, file)
        if found:
            idx = int(found.group(1))
            dcm_files_dict[idx] = file

    dcm_data = []
    dcm_keys = list(dcm_files_dict.keys())
    dcm_keys.sort()
    for dcm_key in dcm_keys:
        dcm_path = os.path.join(directory, dcm_files_dict[dcm_key])
        image_array = read_image(dcm_path)
        dcm_data.append(image_array)
    return np.asarray(dcm_data)


def read_image(dcm_path):
    transforms = Compose([LoadImage(image_only=True)])
    data = transforms(dcm_path)
    data = np.asarray(data)
    if data.shape[-1] == 1:
        data = data.squeeze(-1)
    return data


def save_np(output_path, obj, compressed=False):
    if compressed:
        np.savez_compressed(output_path + '.npz', obj)
    else:
        np.save(output_path + '.npy', obj)


def sanitize_masks(masks):
    shapes = [x.shape for x in masks]
    majority_shape = Counter(shapes).most_common(1)[0][0]
    new_masks = []
    for mask in masks:
        if mask.shape != majority_shape:
            new_masks.append(
                resize_mask(mask)
            )
    return new_masks


def resize_mask(mask, target_shape):
    zoom_factors = [t / s for s, t in zip(mask.shape, target_shape)]
    return zoom(mask, zoom_factors, order=0)


if __name__ == '__main__':
    main()
