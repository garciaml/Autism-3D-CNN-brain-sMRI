
# Use example: python run_transforms_on_masks.py output_highresnet_train1 output_highresnet_train1_transformed 

# Import libraries
import os
import shutil
import tempfile
from glob import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torchio as tio
from monai.data import DataLoader
import monai
from helpers import makedir
import nibabel as nib
import torch.nn as nn
#print_config()


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('masks_dir', help='The directory with the input dataset.')
parser.add_argument('output_dir', default='output_masks_transformed', help='The directory where the output masks should be stored.')

args = parser.parse_args()

# BIDS data
masks_dir = args.masks_dir
output_dir = args.output_dir
makedir(output_dir)

# Find good files
subject_files = glob(os.path.join(masks_dir, "*nii.gz"))

### 1- Preprocessing - for all masks 
# Spatial normalization with Resample
# crop or pad 256*256*256 or EnsureShapeMultiple(32)
# Reorder data to be closest to canonical (RAS+) orientation
transform = tio.Compose([
    tio.Resample(1.5),
    tio.CropOrPad(256),
    tio.ToCanonical()
    ])

def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img), affine]

subids = list(map(lambda s: ((s.split("/")[-1]).split("_")[0]).split("sub-")[-1], subject_files))
for i, filename in enumerate(subject_files):
    subject = tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader))
    transformed_subject = transform(subject)
    img = nib.Nifti1Image(transformed_subject['image']['data'].cpu().detach().numpy(), transformed_subject['image']['affine'])
    nib.save(img, os.path.join(output_dir, "sub-" + subids[i] + "_transf.nii.gz"))

