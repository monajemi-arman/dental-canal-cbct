#!/usr/bin/env python
import json
import os
from locale import normalize

import numpy as np
from torch.utils.data import Dataset

# Path to config file
config_json = 'config.json'


class BaseDataset(Dataset):
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
        return self.read_item(item)

    def read_item(self, item):
        # index to key
        item = self.images[item]
        if item:
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
        keys = list(data.keys())
        return keys, data

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


class RegionalDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, annotation_file, image_suffix=".npy", transforms=None):
        super().__init__(image_dir, mask_dir, annotation_file, image_suffix=".npy", transforms=None)

        # Queue: Images with multiple bounding boxes will create a queue of bboxes before moving to next image
        self.queue = []  # Queue of bboxes of an image
        self.offset = 0  # This will be subtracted from requested index in __getitem__

    def __getitem__(self, item):
        # Mind the queue
        item -= self.offset
        if item < 0:
            item = 0

        # Read data
        image, mask, bboxes = self.read_item(item)

        # Number of bounding boxes
        bboxes_shape = np.array(bboxes).shape
        if len(bboxes_shape) == 1:
            bbox_count = 1
        elif len(bboxes_shape) == 2:
            bbox_count = bboxes_shape[0]

        if bbox_count > 1:
            for bbox in bboxes:
                self.queue.append([image, mask, bbox])

        # Getting the item, taking into account the queue
        if len(self.queue) > 0:
            ret_image, ret_mask, ret_bbox = self.queue.pop()
            self.offset += 1
        else:
            # Single bbox image
            ret_image, ret_mask, ret_bbox = image, mask, bboxes

        # Crop to region
        cropped_image, cropped_mask = crop_image_and_mask(image, mask, bbox)

        return cropped_image, cropped_mask


def crop_image_and_mask(image, mask, bbox):
    bbox = [int(x) for x in bbox]
    x, y, z, w, h, length_in_z = bbox
    cropped_image = image[z:z + length_in_z, y:y + h, x:x + w]

    if mask.ndim == 3:
        cropped_mask = mask[z:z + length_in_z, y:y + h, x:x + w]
    else:
        raise ValueError("Mask must be 3D array")

    return cropped_image, cropped_mask


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

    dataset = BaseDataset(image_dir=os.path.join(all_dir, "images"), mask_dir=os.path.join(all_dir, "masks"),
                          annotation_file=all_dir + '.json', image_suffix=image_suffix, transforms=transforms)


if __name__ == '__main__':
    main()
