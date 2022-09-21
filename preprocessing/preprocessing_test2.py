#!/bin/usr/python3

# Use example 1: python preprocessing_test2.py /home/melanie/data_test data_test_information.csv /home/melanie/data_test/transforms 
# Use example 2: python preprocessing_test2.py data/train4/BIDS_data_brain Test2_algo_train1.csv test2_algo_train1_preprocessed

## Import libraries
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
from monai.config import print_config
from monai.transforms import (
    AddChanneld,
    SqueezeDimd,
    Compose,
    LoadImaged,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    SpatialPadd,
    Orientationd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandShiftIntensityd,
    RandBiasFieldd,
    RandHistogramShiftd,
    HistogramNormalized,
    SaveImaged,
)
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from helpers import makedir
import time
import nibabel as nib
#print_config()


## Set the seed for reproducibility
set_determinism(seed=0)

## Create a parser
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('subjects_csv', help="The csv file containing information about the subjects to be preprocessed.")
parser.add_argument('outdir', help="The name of the output directory in the BIDS directory.")

args = parser.parse_args()

## Parse data
bids_dir = args.bids_dir
outdir = args.outdir

## Create output directory
makedir(os.path.join(bids_dir, outdir))

## Find good files
subids_to_preprocess = pd.read_csv(args.subjects_csv, dtype=str).SUB_ID.astype(str).tolist()
subject_dirs = glob(os.path.join(bids_dir, "sub-*"))
subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
subjects_to_analyze = [sub for sub in subjects_to_analyze if sub in subids_to_preprocess]
filenames = []
for subject_label in subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, 
                                "sub-%s"%subject_label,
                                "anat", 
                                "*_T1w.nii*")) + glob(os.path.join(bids_dir,
                                "sub-%s"%subject_label,
                                "ses-*","anat", 
                                "*_T1w.nii*"))
    filenames = filenames + sub_filenames
    if len(sub_filenames) < 1:
        print(subject_label)

#print(len(subjects_to_analyze))

## Preprocessing steps detailed
# Spatial normalization with Resample
# Intensity normalization with RescaleIntensity(percentiles=(0.5, 99.5)) and ZNormalization
# crop or pad 256*256*256 or EnsureShapeMultiple(32)
# Reorder data to be closest to canonical (RAS+) orientation
transform = tio.Compose([
    tio.Resample(1.5),
    tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
    tio.ZNormalization(),
    #tio.EnsureShapeMultiple(256),
    tio.CropOrPad(256),
    tio.ToCanonical()
    ]) 

## Create a subjects dataset including the transformations
## and run the operations
def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img), affine]

for i, subid in enumerate(subjects_to_analyze):
    subject = tio.Subject(image = tio.ScalarImage(filenames[i], reader=nib_reader))
    transformed_subject = transform(subject)
    img = nib.Nifti1Image(transformed_subject['image']['data'].cpu().detach().numpy(), transformed_subject['image']['affine'])
    nib.save(img, os.path.join(bids_dir, outdir, subid + "_prep_2.nii.gz"))
