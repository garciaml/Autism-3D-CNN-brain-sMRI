#!/bin/usr/python3

# Use example: python predict_medicalnet.py /home/melanie/sMRI_ASD/data/train1/BIDS_data_brain /home/melanie/sMRI_ASD/net2/MedicalNet/train1_models_lr_1e-3_resnet50_new_preproc_from_10ep/model_42_epochs.pth /home/melanie/sMRI_ASD/net2/MedicalNet/train1_models_lr_1e-3_resnet50_new_preproc_from_10ep/predictions/model_42_epochs 

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
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from helpers import makedir
from models.resnet2 import resnet50
from model import generate_model
from log import create_logger
from torchsummary import summary
import time
import nibabel as nib
import torch.nn as nn
#print_config()
from monai.data import CSVSaver
from medcam import medcam


# Set the seed for reproducibility
set_determinism(seed=0)


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
makedir(os.path.join(output_dir, "attention_maps"))
n_classes = int(args.n_classes) 

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
test_subjects = []
train_meta_data = {"SUB_ID": [], "filename": [], "label": []}
validation_meta_data = {"SUB_ID": [], "filename": [], "label": []}
test_meta_data = {"SUB_ID": [], "filename": [], "label": []}
for i, subid in enumerate(common_subjects_to_analyze):
    if "train" in datasets[i]:
        filename = os.path.join(bids_dir, "preprocessed_2", "train", subid + "_prep_2.nii.gz")
        train_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32), SUB_ID = subid))
        train_meta_data['SUB_ID'].append(subid)
        train_meta_data['filename'].append(filename)
        train_meta_data['label'].append(labels[i])
    elif "val" in datasets[i]:
        filename = os.path.join(bids_dir, "preprocessed_2", "val", subid + "_prep_2.nii.gz")
        validation_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32), SUB_ID = subid))
        validation_meta_data['SUB_ID'].append(subid)
        validation_meta_data['filename'].append(filename)
        validation_meta_data['label'].append(labels[i])
    elif "test" in datasets[i]:
        filename = os.path.join(bids_dir, "preprocessed_2", "test", subid + "_prep_2.nii.gz")
        test_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32), SUB_ID = subid))
        test_meta_data['SUB_ID'].append(subid)
        test_meta_data['filename'].append(filename)
        test_meta_data['label'].append(labels[i])

#train_subjects_dataset = tio.SubjectsDataset(train_subjects)
#validation_subjects_dataset = tio.SubjectsDataset(validation_subjects)
#test_subjects_dataset = tio.SubjectsDataset(test_subjects)

all_subjects = train_subjects + validation_subjects + test_subjects
subjects_dataset = tio.SubjectsDataset(all_subjects)

# Save meta data
pd.DataFrame(train_meta_data).to_csv(os.path.join(output_dir, "train", "train_meta_data.csv"))
pd.DataFrame(validation_meta_data).to_csv(os.path.join(output_dir, "validation", "validation_meta_data.csv"))
pd.DataFrame(test_meta_data).to_csv(os.path.join(output_dir, "test", "test_meta_data.csv"))

# Dataloader
#train_loader = DataLoader(train_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available(), shuffle=True)
#val_loader = DataLoader(validation_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())
#test_loader = DataLoader(test_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())

#ds_loader = DataLoader(subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())
ds_loader = DataLoader(subjects_dataset, batch_size=1)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Load model, initialize CrossEntropyLoss and Adam optimizer
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = resnet50(sample_input_D=256, sample_input_H=256, sample_input_W=256, num_seg_classes=n_classes)
model = nn.Sequential(model, nn.AvgPool3d(32), nn.Flatten(), nn.Linear(2048, 2))
net_dict = model.state_dict()
pretrain = torch.load(pretrain_path)
pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)
model = nn.Sequential(model, nn.Softmax(1))
#set_parameter_requires_grad(model, True)
model = medcam.inject(model, output_dir=os.path.join(output_dir, "attention_maps"), save_maps=True, backend='ggcam')
model.to(device).eval()
#print(summary(model, (1, 256, 256, 256)))

#with torch.inference_mode():
with torch.no_grad():
    # ALL
    num_correct = 0.0
    metric_count = 0
    saver = CSVSaver(output_dir=output_dir)
    start_time = time.time()
    for val_data in ds_loader:
        #print(val_data)
        val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
        val_outputs_raw = model(val_images)
        val_outputs = val_outputs_raw.argmax(dim=1)
        value = torch.eq(val_outputs, val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
        saver.save_batch(val_outputs_raw)
        #print(val_data["SUB_ID"])

    metric = num_correct / metric_count
    print("evaluation metric:", metric)
    saver.finalize()
    elapsed_time = time.time() - start_time
    print(elapsed_time)
#    ## TRAIN
#    #num_correct = 0.0
#    #metric_count = 0
#    ##saver = CSVSaver(output_dir="./output_test")
#    #saver = CSVSaver(output_dir=os.path.join(output_dir, "train"))
#    #for val_data in train_loader:
#    #    val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
#    #    val_outputs_raw = model(val_images)
#    #    val_outputs = val_outputs_raw.argmax(dim=1)
#    #    value = torch.eq(val_outputs, val_labels)
#    #    metric_count += len(value)
#    #    num_correct += value.sum().item()
#    #    #saver.save_batch(val_outputs, val_data["image_meta_dict"])
#    #    saver.save_batch(val_outputs_raw)
#    #    break
#    #metric = num_correct / metric_count
#    #print("evaluation metric:", metric)
#    #saver.finalize()
#    ## VALIDATION
#    #num_correct = 0.0
#    #metric_count = 0
#    ##saver = CSVSaver(output_dir="./output_test")
#    #saver = CSVSaver(output_dir=os.path.join(output_dir, "validation"))
#    #for val_data in val_loader:
#    #    val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
#    #    #val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
#    #    val_outputs_raw = model(val_images)
#    #    val_outputs = val_outputs_raw.argmax(dim=1)
#    #    value = torch.eq(val_outputs, val_labels)
#    #    metric_count += len(value)
#    #    num_correct += value.sum().item()
#    #    #saver.save_batch(val_outputs, val_data["image_meta_dict"])
#    #    saver.save_batch(val_outputs_raw)
#    #metric = num_correct / metric_count
#    #print("evaluation metric:", metric)
#    #saver.finalize()
#    ## TEST
#    #num_correct = 0.0
#    #metric_count = 0
#    ##saver = CSVSaver(output_dir="./output_test")
#    #saver = CSVSaver(output_dir=os.path.join(output_dir, "test"))
#    #for val_data in test_loader:
#    #    val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
#    #    #val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
#    #    val_outputs_raw = model(val_images)
#    #    val_outputs = val_outputs_raw.argmax(dim=1)
#    #    value = torch.eq(val_outputs, val_labels)
#    #    metric_count += len(value)
#    #    num_correct += value.sum().item()
#    #    #saver.save_batch(val_outputs, val_data["image_meta_dict"])
#    #    saver.save_batch(val_outputs_raw)
#    #metric = num_correct / metric_count
#    #print("evaluation metric:", metric)
#    #saver.finalize()
#
