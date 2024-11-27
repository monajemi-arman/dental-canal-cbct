#!/usr/bin/env python
import json
import os
import re
from math import ceil
from collections import Counter
import numpy as np
from monai.transforms import LoadImage
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
    output_train = config['dataset']['train']
    output_val = config['dataset']['val']
    output_test = config['dataset']['test']
    output_images = os.path.join(config['dataset']['all'], config['dataset']['images'])
    output_masks = os.path.join(config['dataset']['all'], config['dataset']['masks'])
    output_json = config['dataset']['all'] + '.json'

    # Directories should exist
    for directory in output_all, output_images, output_masks:
        os.makedirs(directory, exist_ok=True)

    # Go through the raw dataset folder
    masks = []
    output_json_data = {}
    image_id = 0
    for first in os.listdir(dataset):
        image_id += 1
        bboxes = []
        first = os.path.join(dataset, first)
        for second in os.listdir(first):
            second = os.path.join(first, second)

            # Save image
            if second.endswith(config['dataset']['dcm_directory_suffix']):
                image_data, spacing = process_dcm_directory(second, config['dataset']['dcm_file_pattern'], spacing=True)
                temp_path = os.path.join(output_images, str(image_id))
                if not os.path.exists(temp_path):
                    save_np(temp_path, image_data, compressed)

            # Get file suffix
            second_parts = os.path.splitext(second)

            # Gather mask if seen
            if second_parts[-1] == config['dataset']['mask_suffix']:
                masks.append(
                    read_image(second)
                )

            # Process ROI JSON
            if second_parts[-1].lower() == '.json':
                with open(second) as f:
                    json_data = json.load(f)
                # Get bounding box
                for markup in json_data['markups']:
                    size = markup['size']
                    center = markup['center']
                    bbox = [0, 0, 0, 0, 0, 0]
                    indices = len(size)
                    for i in range(indices):
                        bbox[i] = (center[i] - size[i] / 2)
                    for i in range(indices):
                        j = indices + i
                        bbox[j] = size[i]
                    # Save bbox in image bboxes
                    bboxes.append(bbox)

        # Save mask
        masks = sanitize_masks(masks)
        combined_mask = np.logical_or.reduce(masks).astype(np.uint8)
        temp_path = os.path.join(output_masks, str(image_id))
        if not os.path.exists(temp_path):
            save_np(temp_path, combined_mask, compressed)

        # Bounding box processing; Convert mm to pixel
        new_bboxes = []
        for bbox in bboxes:
            for i in range(len(bbox)):
                if i >= len(spacing):
                    spacing_i = i - len(spacing)
                else:
                    spacing_i = i
                # Apply ratio
                bbox[i] /= spacing[spacing_i]
                # Normal float
                bbox[i] = float(bbox[i])
            new_bboxes.append(bbox)
        bboxes = new_bboxes
        output_json_data.update({
            str(image_id): bboxes
        })
    # Save bbox JSON
    with open(output_json, 'w') as f:
        json.dump(output_json_data, f)

    # Now split JSON into train, test, split
    ratio = config['dataset']['split']
    split_json(output_json, config['dataset']['train'], config['dataset']['val'], config['dataset']['test'],
               ratio=[ratio['train'], ratio['val'], ratio['test']])


def process_dcm_directory(directory, pattern, spacing=False):
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
        image_array, metadata = read_image(dcm_path, image_only=False)
        dcm_data.append(image_array)

    if spacing:
        return np.asarray(dcm_data), metadata['spacing']
    else:
        return np.asarray(dcm_data)


def read_image(dcm_path, image_only=True):
    loader = LoadImage(image_only=image_only)
    loaded = loader(dcm_path)

    if image_only:
        data = np.asarray(loaded)
        if data.shape[-1] == 1:
            data = data.squeeze(-1)
        return data
    else:
        data, metadata = np.asarray(loaded[0]), loaded[1]
        if data.shape[-1] == 1:
            data = data.squeeze(-1)
        return data, metadata


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
            mask = resize_mask(mask, majority_shape)
        new_masks.append(mask)
    return new_masks


def resize_mask(mask, target_shape):
    zoom_factors = [t / s for s, t in zip(mask.shape, target_shape)]
    return zoom(mask, zoom_factors, order=0)


def split_json(all_json, train, val, test, ratio):
    with open(all_json) as f:
        all_data = json.load(f)

    all_keys = list(all_data.keys())
    all_count = len(all_keys)

    val_count = ceil(all_count * ratio[1])
    test_count = ceil(all_count * ratio[2])
    train_count = all_count - (val_count + test_count)

    train_indices, val_indices, test_indices = [], [], []

    idx = 0
    # Fill val
    for count in range(val_count):
        val_indices.append(idx)
        idx += 1
    # Fill test
    for count in range(test_count):
        if idx < all_count:
            test_indices.append(idx)
            idx += 1
    # Fill train
    for count in range(train_count):
        if idx < all_count:
            train_indices.append(idx)
            idx += 1

    # Split into JSONs
    for output_json, indices in (
            (train, train_indices),
            (val, val_indices),
            (test, test_indices)
):
        # Gather data
        output_data = {}
        for idx in indices:
            key = all_keys[idx]
            output_data.update({key: all_data[key]})
        # Save to file
        with open(output_json + '.json', 'w') as f:
            json.dump(output_data, f)


if __name__ == '__main__':
    main()
