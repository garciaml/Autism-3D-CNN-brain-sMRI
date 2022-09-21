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


# Set the seed for reproducibility
set_determinism(seed=0)

# Load parameters
parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('model_dir', default='/out_dir', help='The directory where the models '
                    'should be stored.')
parser.add_argument('--n_classes', default='2', help='Integer; Number of classes for the classification model.')

args = parser.parse_args()


# BIDS data
#bids_dir = "/home/melanie/BIDS_data_test/"
#model_dir = os.path.join(bids_dir, "models_dct")
bids_dir = args.bids_dir
model_dir = args.model_dir
makedir(model_dir)
n_classes = int(args.n_classes)
# Create log file
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'training.log'))

# Get labels from participants.tsv
# 1- read tsv file with pd
df_participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)
# Get subjects, labels and datasets
#subject_dirs = glob(os.path.join(bids_dir, "sub-*"))
#subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]
train_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/train", "sub-*"))
train_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in train_subject_dirs]
validation_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/validation", "sub-*"))
validation_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in validation_subject_dirs]
test_subject_dirs = glob(os.path.join(bids_dir, "preprocessed/test", "sub-*"))
test_subjects_to_analyze = [(subject_dir.split("-")[-1]).split("_T1w")[0] for subject_dir in test_subject_dirs]

# 2- make subjects_to_analyze match subids
# 3- get labels from column "label"
train_filenames = []
train_labels = []
for subject_label in train_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/train", "sub-%s*"%subject_label, "*.nii*"))
    train_labels = train_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    train_filenames = train_filenames + sub_filenames
    #break
#print(train_filenames)
test_filenames = []
test_labels = []
for subject_label in test_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/test", "sub-%s*"%subject_label, "*.nii*"))
    test_labels = test_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    test_filenames = test_filenames + sub_filenames
validation_filenames = []
validation_labels = []
for subject_label in validation_subjects_to_analyze:
    sub_filenames = glob(os.path.join(bids_dir, "preprocessed/validation", "sub-%s*"%subject_label, "*.nii*"))
    validation_labels = validation_labels + [float(df_participants[df_participants.participant_id == subject_label]["label"].item())]*len(sub_filenames)
    validation_filenames = validation_filenames + sub_filenames

#train_labels = [float(df_participants[df_participants.participant_id == subid]["label"].item()) for subid in train_subjects_to_analyze]
#test_labels = [float(df_participants[df_participants.participant_id == subid]["label"].item()) for subid in test_subjects_to_analyze]
#validation_labels = [float(df_participants[df_participants.participant_id == subid]["label"].item()) for subid in validation_subjects_to_analyze]
# 4- get dataset from column "dataset"
#datasets = [df_participants[df_participants.participant_id == subid]["dataset"].item() for subid in subjects_to_analyze]

# 5- create train, val, test dictionaries with keys "image", "label" and "dataset" , adding if condition to get correct dataset
train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_filenames, train_labels)]
validation_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(validation_filenames, validation_labels)]
test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_filenames, test_labels)]

#print(test_files[0:5])
# Define transforms
transforms = Compose([LoadImaged(keys=['image']),
                      AddChanneld(keys=['image']),
                      EnsureTyped(keys=["image"]),
                      ])

post_pred = Compose([EnsureType(), Activations(softmax=True)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, num_classes=2)])

# Define image dataset, data loader
check_ds = Dataset(train_files, transform=transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available())
check_data = monai.utils.misc.first(check_loader)
#print(type(im), im.shape, label)
#print(check_data["image"].shape, check_data["label"])

# create a training data loader
train_ds = Dataset(train_files, transform=transforms)
val_ds = Dataset(validation_files, transform=transforms)
#train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(train_ds, batch_size=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=2, pin_memory=torch.cuda.is_available())

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=n_classes).to(device)
net_dict = model.state_dict()
pretrain_path = '/home/melanie/sMRI_ASD/models_brain_train1_densenet121_lr_1e-5/model_19_epochs.pth'
log('loading pretrained model {}'.format(pretrain_path))
pretrain = torch.load(pretrain_path)
pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
net_dict.update(pretrain_dict)
model.load_state_dict(net_dict)


loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
#optimizer = torch.optim.Adam(model.parameters(), 1e-3)
auc_metric = ROCAUCMetric()

#print(model)

#CUDA_LAUNCH_BLOCKING=1
# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter(model_dir)
for epoch in range(20, 100):
    log("-" * 10)
    log(f"epoch {epoch + 1}")
    model.train()
    epoch_loss = 0
    step = 0
    start_time = time.time()
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].long().to(device)
        #print(inputs.size())
        #print(labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(outputs.size())
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        log(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    elapsed_time = time.time() - start_time
    log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    log("Epoch time duration: " + str(elapsed_time))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)

            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
            #num_correct = 0.0
            #metric_count = 0
            #for val_data in val_loader:
            #    val_images, val_labels = val_data["image"].to(device), val_data["label"].long().to(device)
            #    val_outputs = model(val_images)
            #    value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            #    metric_count += len(value)
            #    num_correct += value.sum().item()
            #metric = num_correct / metric_count
            #metric_values.append(metric)
            #if metric > best_metric:
            #    best_metric = metric
            #    best_metric_epoch = epoch + 1
            #    torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            #    print("saved new best metric model")
            #log(
            #    "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
            #        epoch + 1, metric, best_metric, best_metric_epoch
            #    )
            #)
            log(
                 "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                     epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                 )
            )
            #writer.add_scalar("val_accuracy", metric, epoch + 1)
            writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
            log("val_accuracy: " + str(acc_metric) + str(epoch + 1))
    if (epoch + 1) % 2 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, "model_" + str(epoch) + "_epochs.pth"))
log(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
logclose()
