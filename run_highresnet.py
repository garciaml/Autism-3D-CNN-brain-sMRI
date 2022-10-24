#!/bin/usr/python3


# Use example: python run_highresnet.py ../data/train1/BIDS_data_brain output_highresnet_train1 


## Import libraries
import time
import datetime
from pathlib import Path
import os
from glob import glob
import argparse

import nibabel as nib
import numpy as np
import pandas as pd
from utils.helpers import makedir

import torch
torch.set_grad_enabled(False);
import numpy as np
import torchio as tio
from torchvision.datasets.utils import download_and_extract_archive

# Set the seed for reproducibility
torch.manual_seed(20202021)
#print('TorchIO version:', tio.__version__)
#print('Last run:', datetime.date.today())


## Define functions useful to run highres3dnet
def get_lr_remapping(table_path):
    df = pd.read_csv(table_path, sep=' ', names=['Label', 'Name', *'RGBA'])
    mapping = {}
    for row in df.itertuples():
        if 'Left' in row.Name:
            mapping[row.Label] = df[df.Name == f'Right-{row.Name[5:]}'].Label.values[0]
        elif 'Right' in row.Name:
            mapping[row.Label] = df[df.Name == f'Left-{row.Name[6:]}'].Label.values[0]
    return mapping


def run_highresnet(mri_path_list, output_names, output_dir="."):
    # Pretrained model
    repo = 'fepegar/highresnet'
    model_name = 'highres3dnet'
    model = torch.hub.load(repo, model_name, pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    model.to(device).eval()
    # Preprocessing
    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        #tio.Crop((0, 0, 10, 30, 40, 40)),
    ]
    transform = tio.Compose(transforms)
    # Load subject
    for i, mri_path in enumerate(mri_path_list):
        subject = tio.Subject(t1=tio.ScalarImage(mri_path))
        preprocessed = transform(subject)
        # Run model
        patch_overlap = 4
        patch_size = 64
        grid_sampler = tio.inference.GridSampler(
            preprocessed,
            patch_size,
            patch_overlap,
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler)
        aggregator = tio.inference.GridAggregator(grid_sampler)
        preprocessed.clear_history()  # so that image is not padded at the end
        for patches_batch in patch_loader:
            input_tensor = patches_batch['t1'][tio.DATA].to(device)
            locations = patches_batch[tio.LOCATION]
            with torch.cuda.amp.autocast():
                logits = model(input_tensor)
            labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
            aggregator.add_batch(labels, locations)
        patchwise_output_tensor = aggregator.get_output_tensor()
        # Save mask
        mask = patchwise_output_tensor.cpu().detach().numpy().squeeze()
        nib_mask = nib.Nifti1Image(mask, np.eye(4))
        nib.save(nib_mask, os.path.join(output_dir, output_names[i]))


def run_highresnet_tta(mri_path_list, output_names, output_dir="."):
    # Load subject
    #subject_oasis = tio.Subject(t1=tio.ScalarImage(mri_path))
    #subject = subject_oasis
    # Pretrained model
    repo = 'fepegar/highresnet'
    model_name = 'highres3dnet'
    model = torch.hub.load(repo, model_name, pretrained=True)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    model.to(device).eval()

    # Preprocessing
    transforms = [
        tio.ToCanonical(),
        tio.Resample(1),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
        #tio.Crop((0, 0, 10, 30, 40, 40)),
    ]
    transform = tio.Compose(transforms)
    for i, mri_path in enumerate(mri_path_list):
        subject = tio.Subject(t1=tio.ScalarImage(mri_path))
        preprocessed = transform(subject)
        ## Pretrained model
        #repo = 'fepegar/highresnet'
        #model_name = 'highres3dnet'
        #model = torch.hub.load(repo, model_name, pretrained=True)
        #device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        #print('Device:', device)
        #model.to(device).eval()
        # Augmentation
        num_augmentations = 20
        results = []
        remapping = get_lr_remapping('GIFNiftyNet.ctbl')
        flip = tio.Compose((
            tio.RandomFlip(axes=['LR'], flip_probability=1),
            tio.RemapLabels(remapping),
            ),
            p=0.5,
        )
        resample = tio.OneOf({
            tio.RandomAffine(image_interpolation='nearest'): 0.75,
            tio.RandomElasticDeformation(image_interpolation='nearest'): 0.25,
        })
        augment = tio.Compose((flip, resample))
        for _ in range(num_augmentations):
            augmented = augment(preprocessed)
            # Run model
            patch_overlap = 4
            patch_size = 64
            grid_sampler = tio.inference.GridSampler(
                augmented,
                patch_size,
                patch_overlap,
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler)
            aggregator = tio.inference.GridAggregator(grid_sampler)
            preprocessed.clear_history()  # so that image is not padded at the end
            for patches_batch in patch_loader:
                input_tensor = patches_batch['t1'][tio.DATA].to(device)
                locations = patches_batch[tio.LOCATION]
                with torch.cuda.amp.autocast():
                    logits = model(input_tensor)
                labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
                aggregator.add_batch(labels, locations)
            patchwise_output_tensor = aggregator.get_output_tensor()
            lm_temp = tio.LabelMap(tensor=torch.rand(1,1,1,1), affine=augmented.t1.affine)
            augmented.add_image(lm_temp, 'label')
            augmented.label.set_data(patchwise_output_tensor)
            back = augmented.apply_inverse_transform(warn=True)
            results.append(back.label.data)
        result = torch.stack(results).long()
        tta_result_tensor = result.mode(dim=0).values
        # Save mask
        #mask = patchwise_output_tensor.cpu().detach().numpy().squeeze()
        #nib_mask = nib.Nifti1Image(mask, np.eye(4))
        nib_mask = nib.Nifti1Image(tta_result_tensor.cpu().detach().numpy().squeeze(), np.eye(4))
        nib.save(nib_mask, os.path.join(output_dir, output_names[i]))



if __name__=='__main__':
    # Load parameters
    parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
    parser.add_argument('bids_dir', default='/bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the predictions should be stored.')
    args = parser.parse_args()
    # BIDS data
    bids_dir = args.bids_dir
    output_dir = args.output_dir
    makedir(output_dir)
    # Get filenames 
    filenames = glob(os.path.join(bids_dir, "sub-*", "anat", "*.nii*"))
    #filenames = glob(os.path.join("../data", "raw","*.nii*"))
    participants = pd.read_csv(os.path.join(bids_dir, "participants.tsv"), sep="\t", dtype=str)["participant_id"].tolist()
    already_seg = pd.read_csv("test2_already_segmented.csv", header=None)[0].tolist()
    already_seg = list(map(lambda x: x.split("_")[0], already_seg))
    filenames_participants = []
    output_names = []
    for k, f in enumerate(filenames):
        subid = (((f.split("/")[-1]).split(".nii.gz")[0]).split("sub-")[-1]).split("_")[0]
        #subid = (((f.split("/")[-1]).split(".nii.gz")[0]).split("-00")[-1]).split("_")[0]
        #print(subid)
        #subid = (f.split("/")[-1]).split(".nii.gz")[0]
        if subid in participants:
            if subid not in already_seg:
                filenames_participants.append(f)
                output_names.append(subid + "_tta_full_seg.nii.gz")
    #print(len(output_names))
    
    ## Run highresnet in tta mode
    run_highresnet_tta(filenames_participants, output_names, output_dir)

    ## Version for participants when test is available
    #subids_test_available = pd.read_csv("./subids_test_available.csv", index_col=None, header=None, dtype=str)[0].tolist()
    #print(subids_test_available)
    #filenames_test_available = []
    #filenames_no_test_available = []
    #for k, f in enumerate(filenames):
        #subid = (((f.split("/")[-1]).split(".nii.gz")[0]).split("sub-")[-1]).split("_")[0]
        ##subid = (((f.split("/")[-1]).split(".nii.gz")[0]).split("-00")[-1]).split("_")[0]
        ##print(subid)
        ##subid = (f.split("/")[-1]).split(".nii.gz")[0]
        #if (subid.split("_")[0]).split("-")[-1] in subids_test_available:
        #    #output_name = subid + "_tta_full_seg.nii.gz"
        #    #output_names.append(output_name)
        #    filenames_test_available.append(f)
        #else:
        #    output_name = subid + "_tta_full_seg.nii.gz"
        #    output_names.append(output_name)
        #    filenames_no_test_available.append(f)
    #run_highresnet(filenames_test_available, output_names, output_dir)
    #run_highresnet(filenames_no_test_available, output_names, output_dir)

    ## Simple path
    #mri_path = "../data/train1/BIDS_data_brain/sub-28863/anat/sub-28863_T1w.nii.gz"
    #run_highresnet(mri_path)
