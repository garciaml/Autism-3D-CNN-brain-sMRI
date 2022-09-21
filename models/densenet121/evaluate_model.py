
# python evaluate_model.py ../train1/BIDS_data_brain ../models/lr_1e-5/model_1_epochs.pth ./output_test/model_1_epochs/

import os
import shutil
import tempfile
from glob import glob
import argparse
from log import create_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

import torch

import monai

from monai.apps import download_and_extract
#from monai.data import ImageDataset
from monai.data import Dataset, DataLoader
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
#from torch.utils.data import DataLoader
from monai.handlers import CheckpointLoader, CheckpointSaver

from monai.config import print_config
from monai.metrics import compute_roc_auc
from monai.networks.nets import densenet121, NetAdapter
from monai.transforms import (
    AddChanneld,
    SqueezeDimd,
    Compose,
    LoadImaged,
    SaveImaged,
    EnsureTyped,
    Activations,
    AsDiscrete,
    EnsureType,
)
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from helpers import makedir
import time
#print_config()
from monai.data import CSVSaver
from medcam import medcam

# Set the seed for reproducibility
set_determinism(seed=0)

# Load parameters
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('pretrain_path', help='The path where the pretrained model is stored.')
parser.add_argument('output_dir', help='The directory where the predictions should be stored.')
parser.add_argument('--n_classes', default='2', help='Integer; Number of classes for the classification model.')

args = parser.parse_args()


# BIDS data
bids_dir = args.bids_dir
pretrain_path = args.pretrain_path
output_dir = args.output_dir
makedir(output_dir)
makedir(os.path.join(output_dir, "train"))
makedir(os.path.join(output_dir, "validation"))
makedir(os.path.join(output_dir, "test"))
n_classes = int(args.n_classes)
# Create log file

# Get labels from participants.tsv
# 1- read tsv file with pd
df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)
# Get subjects, labels and datasets
train_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/train", "sub-*_T1w"))
train_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in train_subject_dirs]
validation_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/validation", "sub-*_T1w"))
validation_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in validation_subject_dirs]
test_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/test", "sub-*_T1w"))
test_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in test_subject_dirs]

# 2- make subjects_to_analyze match subids
# 3- get labels from column "label"
train_filenames = []
train_labels = []
for subject_label in train_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/train", "sub-%s*_T1w"%subject_label, "*.nii*"))
    train_labels = train_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    train_filenames = train_filenames + sub_filenames
test_filenames = []
test_labels = []
for subject_label in test_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/test", "sub-%s*_T1w"%subject_label, "*.nii*"))
    test_labels = test_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    test_filenames = test_filenames + sub_filenames
validation_filenames = []
validation_labels = []
for subject_label in validation_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/validation", "sub-%s*_T1w"%subject_label, "*.nii*"))
    validation_labels = validation_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    validation_filenames = validation_filenames + sub_filenames

# 5- create train, val, test dictionaries with keys "image", "label" and "dataset" , adding if condition to get correct dataset
train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_filenames, train_labels)]
validation_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(validation_filenames, validation_labels)]
test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_filenames, test_labels)]

# Define transforms
transforms = Compose([LoadImaged(keys=['image']),
                      AddChanneld(keys=['image']),
                      EnsureTyped(keys=["image"]),
                      ])

post_pred = Compose([EnsureType(), Activations(softmax=True)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])

# Define image dataset, data loader
#check_ds = Dataset(train_files, transform=transforms)
#check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
#check_data = monai.utils.misc.first(check_loader)

# create a training data loader
train_ds = Dataset(train_files, transform=transforms)
val_ds = Dataset(validation_files, transform=transforms)
test_ds = Dataset(test_files, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=1, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=1, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_ds, batch_size=1, pin_memory=torch.cuda.is_available())

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)
net_dict = model.state_dict()
#pretrain_path = '/home/melanie/sMRI_ASD/models_brain_train1_densenet121_lr_1e-5/model_19_epochs.pth'
pretrain = torch.load(pretrain_path)
pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)
#model.load_state_dict(torch.load("best_metric_model_classification3d_dict.pth"))
#model = medcam.inject(model, output_dir="attention_maps", save_maps=True, backend='ggcam')
model.eval()
with torch.no_grad():
    ## TRAIN
    #num_correct = 0.0
    #metric_count = 0
    ##saver = CSVSaver(output_dir="./output_test")
    #saver = CSVSaver(output_dir=os.path.join(output_dir, "train"))
    #for val_data in train_loader:
    #    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    #    val_outputs_raw = model(val_images)
    #    val_outputs = val_outputs_raw.argmax(dim=1)
    #    value = torch.eq(val_outputs, val_labels)
    #    metric_count += len(value)
    #    num_correct += value.sum().item()
    #    #saver.save_batch(val_outputs, val_data["image_meta_dict"])
    #    saver.save_batch(val_outputs_raw, val_data["image_meta_dict"])
    #metric = num_correct / metric_count
    #print("evaluation metric:", metric)
    #saver.finalize()
    ## VALIDATION
    #num_correct = 0.0
    #metric_count = 0
    ##saver = CSVSaver(output_dir="./output_test")
    #saver = CSVSaver(output_dir=os.path.join(output_dir, "validation"))
    #for val_data in val_loader:
    #    val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
    #    val_outputs_raw = model(val_images)
    #    val_outputs = val_outputs_raw.argmax(dim=1)
    #    value = torch.eq(val_outputs, val_labels)
    #    metric_count += len(value)
    #    num_correct += value.sum().item()
    #    #saver.save_batch(val_outputs, val_data["image_meta_dict"])
    #    saver.save_batch(val_outputs_raw, val_data["image_meta_dict"])
    #metric = num_correct / metric_count
    #print("evaluation metric:", metric)
    #saver.finalize()
    # TEST
    num_correct = 0.0
    metric_count = 0
    #saver = CSVSaver(output_dir="./output_test")
    saver = CSVSaver(output_dir=os.path.join(output_dir, "test"))
    for val_data in test_loader:
        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        val_outputs_raw = model(val_images)
        val_outputs = val_outputs_raw.argmax(dim=1)
        value = torch.eq(val_outputs, val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
        #saver.save_batch(val_outputs, val_data["image_meta_dict"])
        saver.save_batch(val_outputs_raw, val_data["image_meta_dict"])
    metric = num_correct / metric_count
    print("evaluation metric:", metric)
    saver.finalize()
