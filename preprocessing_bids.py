#!/bin/usr/python3

# Use example: python preprocessing_bids.py /home/melanie/BIDS_data_test /home/melanie/BIDS_data_test/transforms 

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
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from utils.helpers import makedir
from utils.log import create_logger
from torchsummary import summary
import time
import nibabel as nib
#print_config()


## Set the seed for reproducibility
set_determinism(seed=0)


# TODO: change outdir, change in function of new folder utils 
## Create a parser to let the user give instructions
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('outdir', default='/bids_dir/preprocessed', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
args = parser.parse_args()

## Parse data
bids_dir = args.bids_dir
outdir = args.outdir

## Create output directories
makedir(outdir)
makedir(os.path.join(outdir, "train"))
makedir(os.path.join(outdir, "val"))
makedir(os.path.join(outdir, "test"))

## Find good files
subject_dirs = glob(os.path.join(bids_dir, "sub-*"))
subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
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

## Get labels from participants.tsv
# 1- read tsv file with pd
df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)
# 2- make subjects_to_analyze match subids
labels = []
datasets = []
common_subjects_to_analyze = []
common_filenames = []
for i, subid in enumerate(subjects_to_analyze):
    if subid in df_participants.participant_id.tolist():
        common_subjects_to_analyze.append(subid)
        common_filenames.append(filenames[i])
        # 3- get labels from column "label"
        labels.append(int(df_participants[df_participants.participant_id == subid]["label"].item()))
        # 4- get dataset from column "dataset"
        datasets.append(df_participants[df_participants.participant_id == subid]["dataset"].item())
## Create the dictionary
# create train, val, test dictionaries with keys "image", "label" and "dataset" , adding if condition to get correct dataset
train_files = []
validation_files = []
test_files = []
for i, subid in enumerate(common_subjects_to_analyze):
    if subid in df_participants.participant_id.tolist():
        if "train" in datasets[i]:
            train_files.append({"image": common_filenames[i], 'label': labels[i]})
        elif "val" in datasets[i]:
            validation_files.append({"image": common_filenames[i], 'label': labels[i]})
        elif "test" in datasets[i]:
            test_files.append({"image": common_filenames[i], 'label': labels[i]})

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

## Create training, validation, testing subjects datasets including the transorming operations
## and run the transforms
def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img), affine]

for i, subid in enumerate(common_subjects_to_analyze):
    if subid in df_participants.participant_id.tolist():
        if "train" in datasets[i]:
            subject = tio.Subject(image = tio.ScalarImage(common_filenames[i], reader=nib_reader), label=torch.tensor(labels[i]))
            transformed_subject = transform(subject)
            img = nib.Nifti1Image(transformed_subject['image']['data'].cpu().detach().numpy(), transformed_subject['image']['affine'])
            nib.save(img, os.path.join(outdir, "train", subid + "_prep.nii.gz"))
        elif "val" in datasets[i]:
            subject = tio.Subject(image = tio.ScalarImage(common_filenames[i], reader=nib_reader), label=torch.tensor(labels[i]))
            transformed_subject = transform(subject)
            img = nib.Nifti1Image(transformed_subject['image']['data'].cpu().detach().numpy(), transformed_subject['image']['affine'])
            nib.save(img, os.path.join(outdir, "val", subid + "_prep.nii.gz"))
        elif "test" in datasets[i]:
            subject = tio.Subject(image = tio.ScalarImage(common_filenames[i], reader=nib_reader), label=torch.tensor(labels[i]))
            transformed_subject = transform(subject)
            img = nib.Nifti1Image(transformed_subject['image']['data'].cpu().detach().numpy(), transformed_subject['image']['affine'])
            nib.save(img, os.path.join(outdir, "test", subid + "_prep.nii.gz"))

