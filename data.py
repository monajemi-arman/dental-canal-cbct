#!/usr/bin/env python
import json
import os
import numpy as np
import torch
from scipy.ndimage import zoom
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

    def read_item(self, idx):
        # index to key
        key = self.image_id_to_key(idx)
        if key:
            # Paths to image and mask
            image_path = os.path.join(
                self.image_dir, key + self.image_suffix
            )
            mask_path = os.path.join(
                self.mask_dir, key + self.image_suffix
            )

            # Read image and mask
            image = self.read_image(image_path)
            mask = self.read_image(mask_path)

            # Apply transforms if enabled
            image = self.transform(image)

            # Bounding box from JSON
            bbox = self.annotations[key]

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

    def image_id_to_key(self, idx):
        return self.images[idx]

    def key_to_image_id(self, key):
        return self.images.index(key)

    @staticmethod
    def resize(image, target_size=(100, 50, 50), is_mask=False):
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            raise TypeError("Input image must be a numpy array or a tensor.")

        image_is_tensor = torch.is_tensor(image)
        if image_is_tensor:
            image = image.detach().cpu().numpy()

        if len(image.shape) != 3:
            raise ValueError("Input image must have 3 dimensions (d, h, w).")

        scale_factors = [t / s for t, s in zip(target_size, image.shape)]
        resized = zoom(image, scale_factors, order=0 if is_mask else 3)

        return torch.from_numpy(resized) if image_is_tensor else resized


class RegionalDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, annotation_file, image_suffix=".npy", transforms=None, target_size=[100, 50, 50]):
        super().__init__(image_dir, mask_dir, annotation_file, image_suffix=".npy", transforms=transforms)

        self.target_size = target_size

        # Populate boxes
        self.boxes, self.box_id_to_image_id = self.get_boxes()

    def __getitem__(self, box_id):
        image_id = self.box_id_to_image_id[box_id]

        image, mask, _ = self.read_item(image_id)
        bbox = self.boxes[box_id]

        # Crop to region
        cropped_image, cropped_mask = crop_image_and_mask(image, mask, bbox)

        # Fix order of dimensions (x,y,z to z,y,x)
        cropped_image = np.transpose(cropped_image, (2, 1, 0))
        cropped_mask = np.transpose(cropped_mask, (2, 1, 0))

        # Apply transforms if set
        if self.transforms:
            cropped_image = self.transform(cropped_image)

        # Final resize
        cropped_image = self.resize(cropped_image)
        cropped_mask = self.resize(cropped_mask, target_size=self.target_size , is_mask=True)

        # To tensor
        cropped_image = torch.as_tensor(cropped_image).unsqueeze(0)
        cropped_mask = torch.as_tensor(cropped_mask).unsqueeze(0)

        return cropped_image, cropped_mask

    def get_boxes(self):
        all_boxes = []
        box_id_to_image_id = {}
        box_id = 0
        for key in self.annotations.keys():
            # Get image id
            image_id = self.key_to_image_id(key)
            # Save boxes and corresponding image id
            boxes = self.annotations[key]
            for box in boxes:
                all_boxes.append(box)
                box_id_to_image_id[box_id] = image_id
                box_id += 1

        return all_boxes, box_id_to_image_id


def crop_image_and_mask(image, mask, bbox):
    bbox = [int(x) for x in bbox]
    x, y, z, w, h, length_in_z = bbox
    cropped_image = image[x: x + w, y: y + h, z: z + length_in_z]

    if mask.ndim == 3:
        cropped_mask = mask[x: x + w, y: y + h, z: z + length_in_z]
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

    dataset = RegionalDataset(image_dir=os.path.join(all_dir, "images"), mask_dir=os.path.join(all_dir, "masks"),
                              annotation_file=all_dir + '.json', image_suffix=image_suffix, transforms=transforms)

    first = dataset.__getitem__(0)
    print(first)


if __name__ == '__main__':
    main()
