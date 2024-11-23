#!/usr/bin/env python
import json
import os
from locale import normalize

import numpy as np
from torch.utils.data import Dataset

# Path to config file
config_json = 'config.json'


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, annotation_file, image_suffix=".npy", transforms=None):
        self.images, self.annotations = self.load_annotations(annotation_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_suffix = image_suffix
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        :param item: Index integer of image
        :return: Returns image, mask, bbox
        """
        # item is index int, converting + 1 to string that is image name
        item = str(item + 1)
        if item in self.images:
            # Paths to image and mask
            image_path = os.path.join(
                self.image_dir, item + self.image_suffix
            )
            mask_path = os.path.join(
                self.mask_dir, item + self.image_suffix
            )

            # Read image and mask
            image = self.read_image(image_path)
            mask = self.read_image(mask_path)

            # Apply transforms if enabled
            image = self.transform(image)

            # Bounding box from JSON
            bbox = self.annotations[item]

            return image, mask, bbox

    def load_annotations(self, annotation_file):
        with open(annotation_file) as f:
            data = json.load(f)
        return data.keys(), data

    def read_image(self, image_path):
        return np.load(image_path)

    def transform(self, image):
        transform_func = {
            'normalize': self.normalize
        }

        if self.transforms:
            for transform in self.transforms:
                image = transform_func[transform](image)

        return image

    def normalize(self, image):
        min_val = np.min(image)
        max_val = np.max(image)

        # Prevent devision by zero
        if min_val == max_val:
            return np.zeros_like(image)

        normalized = (image - min_val) / (max_val - min_val)
        return normalized


def main():
    # Load config from file
    with open(config_json) as f:
        config = json.load(f)

    # Load params from config
    all_dir = config['dataset']['all']
    transforms = config['dataset']['transforms']
    if config['dataset']['compressed']:
        image_suffix = '.npz'
    else:
        image_suffix = '.npy'

    dataset = CustomDataset(image_dir=os.path.join(all_dir, "images"), mask_dir=os.path.join(all_dir, "masks"),
                            annotation_file=all_dir + '.json', image_suffix=image_suffix, transforms=transforms)

if __name__ == '__main__':
    main()
