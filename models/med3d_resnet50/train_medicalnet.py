#!/bin/usr/python3

# Use example: python preprocessing.py /home/melanie/BIDS_data_test /home/melanie/BIDS_data_test/transforms 

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
from models.resnet2 import resnet50
from log import create_logger
from torchsummary import summary
import time
import nibabel as nib
import torch.nn as nn
#print_config()


# Set the seed for reproducibility
set_determinism(seed=0)


parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', default='/out_dir', help='The directory where the models '
                    'should be stored.')
parser.add_argument('--n_classes', default='2', help='Integer; Number of classes for the classification model.')

args = parser.parse_args()

# BIDS data
bids_dir = args.bids_dir
output_dir = args.output_dir
makedir(output_dir)
n_classes = args.n_classes 
# Create log file
log, logclose = create_logger(log_filename=os.path.join(output_dir, 'training.log'))

# Find good files
# Get labels from participants.tsv
# 1- read tsv file with pd
df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)
# 2- make subjects_to_analyze match subids
def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img, dtype=torch.float32), affine]

common_subjects_to_analyze = df_participants.participant_id.tolist()
datasets = df_participants.dataset.tolist() 
labels = df_participants.label.astype(float).tolist()

train_subjects = []
validation_subjects = []
#test_subjects = []
for i, subid in enumerate(common_subjects_to_analyze):
    if "train" in datasets[i]:
        #common_filenames = glob(os.path.join(bids_dir, "preprocessed_2", "train", "%s_prep_2.nii.gz"%subid))
        filename = os.path.join(bids_dir, "preprocessed_2", "train", subid + "_prep_2.nii.gz")
        train_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))
    elif "val" in datasets[i]:
        #common_filenames = glob(os.path.join(bids_dir, "preprocessed_2", "val", "%s_prep_2.nii.gz"%subid))
        validation_subjects.append(tio.Subject(image = tio.ScalarImage(os.path.join(bids_dir, "preprocessed_2", "val", subid + "_prep_2.nii.gz"), reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))

train_subjects_dataset = tio.SubjectsDataset(train_subjects)
validation_subjects_dataset = tio.SubjectsDataset(validation_subjects)

# Dataloader
train_loader = DataLoader(train_subjects_dataset, batch_size=2, pin_memory=torch.cuda.is_available(), shuffle=True)
val_loader = DataLoader(validation_subjects_dataset, batch_size=2, pin_memory=torch.cuda.is_available())


# Load model, initialize CrossEntropyLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(sample_input_D=256, sample_input_H=256, sample_input_W=256, num_seg_classes=n_classes)
net_dict = model.state_dict()
pretrain_path = '/home/melanie/sMRI_ASD/net2/MedicalNet/pretrain/resnet_50.pth'
log('loading pretrained model {}'.format(pretrain_path))
pretrain = torch.load(pretrain_path)
pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)
for name, param in model.named_parameters():
    #if param.requires_grad:
    #    print(name)
    if name.split(".")[0] in ["layer4"]:
        param.requires_grad = True
    else:
        param.requires_grad = False
#set_parameter_requires_grad(model, True)
model = nn.Sequential(model, nn.AvgPool3d(32), nn.Flatten(), nn.Linear(2048, 2))
model.to(device)
#print(summary(model, (1, 256, 256, 256)))

loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

#CUDA_LAUNCH_BLOCKING=1
# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
for epoch in range(10):
    log("-" * 10)
    log(f"epoch {epoch + 1}")
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()
    for batch_data in train_loader:
        step += 1
        #print(batch_data['image']['data'].shape)
        inputs, labels = batch_data["image"]["data"].to(device), batch_data["label"].long().to(device)
        #print(inputs.size())
        #print(labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs.size())
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_subjects_dataset) // train_loader.batch_size
        log(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        #writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    elapsed_time = time.time() - start_time
    log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    log("Epoch time duration: " + str(elapsed_time))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            num_correct = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                metric_count += len(value)
                num_correct += value.sum().item()
            metric = num_correct / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch+1
            log(
                "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "model_" + str(epoch+1) + "_epochs.pth"))
log(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
logclose()




