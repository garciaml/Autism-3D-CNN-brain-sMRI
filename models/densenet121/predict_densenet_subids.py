#!/bin/usr/python3

# Use example: python predict_densenet_subids.py /home/melanie/sMRI_ASD/data/train4/BIDS_data_brain/test2_algo_train1_preprocessed/ /home/melanie/sMRI_ASD/data/Test2_algo_train1.csv /home/melanie/sMRI_ASD/net2/MedicalNet/good_exps/train1_models_lr_1e-3_densenet121_new_preproc_from_10ep/model_40_epochs.pth /home/melanie/sMRI_ASD/net2/MedicalNet/good_exps/train1_models_lr_1e-3_densenet121_new_preproc_from_10ep/predictions/model_40_epochs_test2 

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
parser.add_argument('indir', default='/indir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('subjects_csv', help='The file conatining information on the subjects (like the label for instance).')
parser.add_argument('pretrain_path', help='The path where the pretrained model is stored.')
parser.add_argument('output_dir', help='The directory where the predictions should be stored.')
parser.add_argument('--n_classes', default='2', help='Integer; Number of classes for the classification model.')

args = parser.parse_args()

# BIDS data
indir = args.indir
pretrain_path = args.pretrain_path
output_dir = args.output_dir
makedir(output_dir)
#makedir(os.path.join(output_dir, "train"))
#makedir(os.path.join(output_dir, "validation"))
#makedir(os.path.join(output_dir, "test"))
makedir(os.path.join(output_dir, "attention_maps"))
n_classes = int(args.n_classes) 
# Create log file
log, logclose = create_logger(log_filename=os.path.join(output_dir, 'training.log'))

# Find good files
# Get labels from participants.tsv
# 1- read tsv file with pd
#df_participants = pd.read_csv(os.path.join(indir, "participants.tsv"), sep="\t", dtype=str)
df_participants = pd.read_csv(os.path.join(args.subjects_csv), dtype=str)
# 2- make subjects_to_analyze match subids
def nib_reader(path):
    load_img = nib.load(path)
    img = np.squeeze(load_img.get_fdata())
    img = np.expand_dims(img, axis=0)
    affine = load_img.affine
    load_img.uncache()
    return [torch.tensor(img, dtype=torch.float32), affine]

#common_subjects_to_analyze = df_participants.participant_id.tolist()
common_subjects_to_analyze = df_participants.SUB_ID.tolist()
datasets = df_participants.dataset.tolist() 
labels = df_participants.label.astype(float).tolist()

#train_subjects = []
#validation_subjects = []
#test_subjects = []
#train_meta_data = {"SUB_ID": [], "filename": [], "label": []}
#validation_meta_data = {"SUB_ID": [], "filename": [], "label": []}
#test_meta_data = {"SUB_ID": [], "filename": [], "label": []}
#for i, subid in enumerate(common_subjects_to_analyze):
#    if "train" in datasets[i]:
#        filename = os.path.join(indir, "preprocessed_2", "train", subid + "_prep_2.nii.gz")
#        train_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))
#        train_meta_data['SUB_ID'].append(subid)
#        train_meta_data['filename'].append(filename)
#        train_meta_data['label'].append(labels[i])
#    elif "val" in datasets[i]:
#        filename = os.path.join(indir, "preprocessed_2", "val", subid + "_prep_2.nii.gz")
#        validation_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))
#        validation_meta_data['SUB_ID'].append(subid)
#        validation_meta_data['filename'].append(filename)
#        validation_meta_data['label'].append(labels[i])
#    elif "test" in datasets[i]:
#        filename = os.path.join(indir, "preprocessed_2", "test", subid + "_prep_2.nii.gz")
#        test_subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))
#        test_meta_data['SUB_ID'].append(subid)
#        test_meta_data['filename'].append(filename)
#        test_meta_data['label'].append(labels[i])
#
#subjects_dataset = tio.SubjectsDataset(train_subjects + validation_subjects + test_subjects)

subjects = []
meta_data = {"SUB_ID": [], "filename": [], "label": []}
for i, subid in enumerate(common_subjects_to_analyze):
    try:
        filename = os.path.join(indir, subid + "_prep_2.nii.gz")
        subjects.append(tio.Subject(image = tio.ScalarImage(filename, reader=nib_reader), label=torch.tensor(labels[i], dtype=torch.float32)))
        meta_data['SUB_ID'].append(subid)
        meta_data['filename'].append(filename)
        meta_data['label'].append(labels[i])
    except:
        print(subid)

subjects_dataset = tio.SubjectsDataset(subjects)

# Save meta data
#pd.DataFrame(train_meta_data).to_csv(os.path.join(output_dir, "train", "train_meta_data.csv"))
#pd.DataFrame(validation_meta_data).to_csv(os.path.join(output_dir, "validation", "validation_meta_data.csv"))
#pd.DataFrame(test_meta_data).to_csv(os.path.join(output_dir, "test", "test_meta_data.csv"))
pd.DataFrame(meta_data).to_csv(os.path.join(output_dir, "meta_data.csv"))

# Dataloader
#train_loader = DataLoader(train_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available(), shuffle=True)
#val_loader = DataLoader(validation_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())
#test_loader = DataLoader(test_subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())
ds_loader = DataLoader(subjects_dataset, batch_size=1, pin_memory=torch.cuda.is_available())


# Load model, initialize CrossEntropyLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)
net_dict = model.state_dict()
#pretrain_path = '/home/melanie/sMRI_ASD/net2/MedicalNet/pretrain/resnet_50.pth'
#log('loading pretrained model {}'.format(pretrain_path))
pretrain = torch.load(pretrain_path)
pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)
model = nn.Sequential(model, nn.Softmax(1))
model = medcam.inject(model, output_dir=os.path.join(output_dir, "attention_maps"), save_maps=True, backend='ggcam')
model.to(device).eval()
#print(summary(model, (1, 256, 256, 256)))

with torch.no_grad():
    # ALL
    num_correct = 0.0
    metric_count = 0
    saver = CSVSaver(output_dir=output_dir)
    for val_data in ds_loader:
        val_images, val_labels = val_data["image"]['data'].to(device), val_data["label"].long().to(device)
        val_outputs_raw = model(val_images)
        val_outputs = val_outputs_raw.argmax(dim=1)
        value = torch.eq(val_outputs, val_labels)
        metric_count += len(value)
        num_correct += value.sum().item()
        saver.save_batch(val_outputs_raw)
    metric = num_correct / metric_count
    print("evaluation metric:", metric)
    saver.finalize()

