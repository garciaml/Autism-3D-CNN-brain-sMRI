# Autism-3D-CNN-brain-sMRI

This project aimed at training 3D-CNN models to predict Autism on structural brain MRI data.

We also used the Guided Grad-CAM and HighRes3DNet algorithms in order to build a method of interpretation of the trained models. 

When using our code or pipeline, please include the following citation:

***Towards 3D Deep Learning for neuropsychiatry: predicting Autism diagnosis using an interpretable Deep Learning pipeline applied to minimally processed structural MRI data*, Melanie Garcia, Clare Kelly**. medRxiv 2022.10.18.22281196; doi: https://doi.org/10.1101/2022.10.18.22281196 

We used Monai (https://github.com/Project-MONAI/MONAI) to build the DenseNet121 network and MedicalNet to keep on training a pre-trained ResNet50 model (https://github.com/Tencent/MedicalNet). 



### Instructions

1- Consider to stucture your data following the BIDS organization (https://bids.neuroimaging.io/).
Your participants.tsv file should contain: 
- a column called *participant_id* corresponding to the subject id (the same as the folder names in the BIDS dataset: sub-<participant_id>).
- a column called *label* corresponding to the binary target variable (here: 0=no diagnosis, 1=Autism)
- a column called *dataset* corresponding to where (training, validation, testing set) each participant data will be used: the code supports the three modalities *train*, *val*, *test*.


2- Make sure to be into a correct environment to run this code.
```
pip install -r requirements.txt
```

***Training and Testing:***

3- Once that your data is BIDS-organized and that you are in a correct environment, you can start the project by running the preprocessing script:
```
python preprocessing_bids.py <your_bids_dir> <preprocessed_directory>
```
For instance, you can choose the preprocessed directory as being *<your_bids_dir>/preprocessed*. 

The code will create *<preprocessed_directory>* if it does not exist in your system yet, and three subdirectories in it: *train*, *val* and *test*. 

It will preprocess your data and store it in the subdirectories in function of the *dataset* column value.


4- You can launch DenseNet121 training by running:
```
python train_densenet.py <your_bids_dir> <preprocessed_directory> <save_densenet_model_dir>
```
The DenseNet121 models and the training.log will be saved into *<save_densenet_model_dir>*.


5- Before launching Med3d-ResNet50 training, download the pretrained model resnet50.pth [here](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view) or [here](https://share.weiyun.com/55sZyIx).

You can launch Med3d-ResNet50 training by running:
```
python train_medicalnet.py <your_bids_dir> <preprocessed_directory> <save_resnet_model_dir> <pretrain_path>
```
*<pretrain_path>* corresponds to the path where you stored your model resnet50.pth.

The Med3d-ResNet50 models and the training.log will be saved into *<save_resnet_model_dir>*.


6- Launch the predictions:

For the DenseNet121 model:
```
python predict_densenet.py <your_bids_dir> <preprocessed_directory> <saved_densenet_model> <output_dir>
```
*<saved_densenet_model>* is the path of the saved DenseNet121 model you want to use for the predictions.

For the Med3d-ResNet50 model:
```
python predict_medicalnet.py <your_bids_dir> <preprocessed_directory> <saved_resnet_model> <output_dir>
```
*<saved_resnet_model>* is the path of the saved Med3d-ResNet50 model you want to use for the predictions.


*<output_dir>* will consist of :
- a sub-directory *attention_maps* with all the attention maps inside;
- a sub-directory *train* containing a file *train_meta_data.csv* with info on the participants included in the training set;
- a sub-directory *validation* containing a file *validation_meta_data.csv* with info on the participants included in the validation set;
- a sub-directory *test* containing a file *test_meta_data.csv* with info on the participants included in the test set;
- a file *predictions.csv* containing the logits (i.e. the probability scores) returned by the model, the first column being for label 0 and second one for label 1. 

The file *predictions.csv* and the files in *attention_maps* corresponds to the same index numbers. It was generated such as the files correspond to the train, validation and testing set files in this strict order (see code predict_densenet.py). Thus the total number of files N = N_train + N_val + N_test. 


***INFERENCE on a second testing set*** (optional step if all of your testing set was included in your initial BIDS dataset):

7- You can launch preprocessing with:
```
python preprocessing_test2.py <your_bids_dir> <subjects_csv> <preprocessed_directory_test2>
```
*<subjects_csv>* contains at least two columns:
- a column called *SUB_ID* corresponding to the subject id (the same as the folder names in the BIDS dataset: sub-<SUB_ID>);
- a column called *label* corresponding to the binary target variable.

The code will create *<preprocessed_directory_test2>* if it does not exist in your system yet. It will preprocess your data and store it in 
<preprocessed_directory_test2>.

8- You can launch the predictions:

For the DenseNet121 model:
```
python predict_densenet_subids.py <preprocessed_directory_test2> <subjects_csv> <saved_densenet_model> <output_dir_test2>
```
*<saved_densenet_model>* is the path of the saved DenseNet121 model you want to use for the predictions.

For the Med3d-ResNet50 model:
```
python predict_medicalnet_subids.py <preprocessed_directory_test2> <subjects_csv> <saved_resnet_model> <output_dir_test2>
```
*<saved_resnet_model>* is the path of the saved Med3d-ResNet50 model you want to use for the predictions.

*<output_dir_test2>* will consist of : 
- a sub-directory *attention_maps* with all the attention maps inside;
- a file *predictions.csv* containing the logits (i.e. the probability scores) returned by the model, the first column being for label 0 and second one for label 1. 


***BRAIN SEGMENTATION with HighRes3DNet***

We can segment brain scans to be able to interpret the attention maps previously generated.

9- Run HighRes3DNet:
```
python run_highresnet.py <your_bids_dir> <output_dir_seg>
```
The code performs appropriate preprocessing prior running HighRes3DNet on your BIDS dataset, and it stores the resulting segmented brains in *<output_dir_seg>*.


***Matching Attention maps and Segmented brains***

10- Run:
```
python run_transforms.py <output_dir_seg> <output_dir_seg_transformed>
```
This transforms the segmented brains generated by HighRes3DNet to be in the same space as the one of the attention maps. It stores the resulting transformed segmented brains in *<output_dir_seg_transformed>*.

11- Create a csv file containing:
- a column called *SUB_ID* corresponding to the subject id (the same as the folder names in the BIDS dataset: sub-<SUB_ID>); 
- a column called *label* corresponding to the binary target variable (here: 0=no diagnosis, 1=Autism)
- a column called *dataset* corresponding to where (training, validation, testing set) each participant data was be used (The code can work without this column);
- a column called *att_map_filename* corresponding to the path to the attention map file predicted for a given model; 
- a column called *img_seg_filename* corresponding to the path to the transormed segmented brain; 
- a column called *img_seg_original_filename* corresponding to the path to the segmented brain.

You may call this file *subids_filenames_densenet121_32ep.csv* for instance, because it depends on the model used to generate the attention maps. We will need it for the next step.

12- Launch the comparison between the attention maps and the segmented brains:
```
python generate_ROI_results.py <GIF_ROIs_filename> <subids_files_filename> <output_dir_ROI_results>
```
*<GIF_ROIs_filename>* is the path to the file GIFNiftyNet.ctbl.
*<subids_files_filename>* is the csv file you just created at step 11. 

The code will generate all the results for each participant in <output_dir_ROI_results>.


### Other pieces of code

We have more pieces of code available upon request to garciaml@tcd.ie that are usable after performing this whole pipeline. 


### Citation
When using our code or pipeline, please include the following citation:

***Towards 3D Deep Learning for neuropsychiatry: predicting Autism diagnosis using an interpretable Deep Learning pipeline applied to minimally processed structural MRI data*, Melanie Garcia, Clare Kelly**. medRxiv 2022.10.18.22281196; doi: https://doi.org/10.1101/2022.10.18.22281196 

The abstract *Towards building an interpretable predictive tool for ASD with 3D Convolutional Neural Networks*, Mélanie Garcia, Clare Kelly was introduced at the conference the Organization for Human Brain Mapping 2022 held in Glasgow in June 2022.
