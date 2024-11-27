# Dental Canal Cbct
Teeth canal detection from CBCT 3D images using deep learning

# Purpose
This project aims to first take raw images in DICOM (.dcm) and masks in NIFTI (.nii) format made by 3D Slicer program and create a dataloader for them. Secondly it uses these images, masks, and given ROI JSONs to train a deep learning model for segmentation and object detection.

# Features
* **Models** supported
  * UNet
* Dataset types
  * BaseDataset
    * Returns image, mask, bbox (bbox in format z, x, y, w, h, Z)
  * RegionalDataset
    * Converts your dataset to segmented ROI boxes for your model to learn segmenting indside the boxes.

# Requirements
1. Clone this repository.
```bash
git clone https://github.com/monajemi-arman/dental-canal-cbct
cd dental-canal-cbct/ 
```
2. Make sure you have cuda toolkit and NVIDIA drivers installed.
3. Install the required python modules.
```bash
pip install -r requirements.txt
```
# Usage
* **Dataset**  
Your dataset must be put in 'dataset' folder in project directory and follow this order:  
```
dataset/  
    -- first_image/
        -- roi_1_first_image.json
        -- roi_2_first_image.json
        -- ...
        -- first_image_dcm/
            -- first_image_001.dcm
            -- first_image_002.dcm
            -- ...
    ... ... ...
```
* Once your dataset is in this format, run the convert script to prepare numpy arrays of your images and masks along with a concise JSON of bounding boxes:
```bash
python convert.py
```

* **Configs**  
All the configs related to training, dataset conversion and split, transforms, and paths are saved in config.json. You may change these parameters as you wish, make sure to remove "all/" directory, all.json, train.json, val.json, and test.json, then re-run convert.py with the new config.

* **Training**
```bash
python train.py 
```