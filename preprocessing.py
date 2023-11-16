import os
import torch
from torchvision import transforms, datasets
from torchvision.transforms import CenterCrop
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
import pandas as pd

device = torch.device("cuda")

def create_transformed_datasets(in_dir, out_dir, crop_size):
    # Define transformation for creating upfront datasets
    upfront_transforms = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    # Load the dataset
    image_dataset = datasets.ImageFolder(in_dir, upfront_transforms)

    # Get class names to create folders
    class_names = image_dataset.classes

    # Save the transformed dataset in separate directories
    os.makedirs(out_dir, exist_ok=True)

    for i, (image, label) in enumerate(image_dataset):
        # Get the filename of the current image without extension
        fullpath = image_dataset.imgs[i][0]
        filebase = os.path.basename(fullpath)
        parent_dir_path = os.path.dirname(fullpath)
        focal_class = os.path.basename(parent_dir_path)
        label_dir = os.path.join(out_dir, focal_class)
        os.makedirs(label_dir, exist_ok=True)
        out_file = os.path.join(label_dir, filebase.replace('tif','png'))

        # Save the cropped image
        Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy()).save(out_file)

crop_size = 39

raw_train_dir = '/home/kdoherty/spurge/data_release/data/images/train'
raw_val_dir = '/home/kdoherty/spurge/data_release/data/images/test_1'

train_dir = f'/home/kdoherty/spurge/data_release/data/crop_{crop_size}/train'
val_dir = f'/home/kdoherty/spurge/data_release/data/crop_{crop_size}/val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

create_transformed_datasets(raw_train_dir, train_dir, crop_size)
create_transformed_datasets(raw_val_dir, val_dir, crop_size)