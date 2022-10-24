# Autism-3D-CNN-brain-sMRI

This project aimed at training 3D-CNN models to predict Autism on structural brain MRI data.

We also used the Guided Grad-CAM and HighRes3DNet algorithms in order to build a method of interpretation of the trained models. 

The abstract *Towards building an interpretable predictive tool for ASD with 3D Convolutional Neural Networks*, Mélanie Garcia, Clare Kelly was introduced at the conference the Organization for Human Brain Mapping 2022 held in Glasgow in June 2022.

Preprint available: https://doi.org/10.1101/2022.10.18.22281196 

We used Monai (https://github.com/Project-MONAI/MONAI) to build the DenseNet121 network and MedicalNet to keep on training a pre-trained ResNet50 model (https://github.com/Tencent/MedicalNet). 

### Instructions
1- Start the project by running the preprocessing script. Consider to stucture your data following the BIDS organization (https://bids.neuroimaging.io/).
If your data is BIDS-organized, you may run:
```
python ...
```

### Citation
When using our code, please include the following citation:

***Towards 3D Deep Learning for neuropsychiatry: predicting Autism diagnosis using an interpretable Deep Learning pipeline applied to minimally processed structural MRI data*, Melanie Garcia, Clare Kelly**. medRxiv 2022.10.18.22281196; doi: https://doi.org/10.1101/2022.10.18.22281196 
