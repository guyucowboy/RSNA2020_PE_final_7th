# Kaggle competition:
# [RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection)


Team Yuval Reina:
====================

-   [Yuval Reina](https://www.kaggle.com/yuval6967)

Private Leaderboard Score: 0.157

Private Leaderboard Place: 7

General
=======

This archive holds the code which was used to create and inference
the 7th place solution in “RSNA STR Pulmonary Embolism Detection” competition.

The solution consists of the following components, run consecutively

-   Prepare data and metadata

-   Training features generating neural networks

-   Training transformer neural networks based on the features and metadata

-   Inference

ARCHIVE CONTENTS
================

-   Main - All notebooks needed to prepare and train the models

-   kaggle_kernel - The inference code which run on kaggle's kernel to run on kaggle go to [https://www.kaggle.com/yuval6967/rsna2020-inference-2nd-op-final](https://www.kaggle.com/yuval6967/rsna2020-inference-2nd-op-final)

-   exp - a python library holding files that are automatically created using some of the notebooks

Setup
=====

### HARDWARE: (The following specs were used to create the original solution)

CPU intel i9-9920, RAM 64G, GPU Tesla V100, GPU Titan RTX.


### SOFTWARE (python packages are detailed separately in requirements.txt):

OS: Ubuntu 18.04 TLS

CUDA – 10.2

Kaggle GPU docker with the following added/changed python libraries:

* Pytorch 1.6.0 

* Fire 

* geffnet

* sandesh 

* pretrainedmodels


DATA SETUP
==========

1.  Download train and test data from Kaggle and inflate the files. As the files are very big you can also download only the train data and keep it zipped. 

2. Update config.json file with the data directories locations:

* data - the main data location

* train -  location of unzipped train files

* test -  location of unzipped test files

3. Create and update the name of the following folders

* features  - location for features files

* models - location for models files

* outputs - location for predictions files

Data Processing
===============

Prepare data + metadata
-----------------------

Run the prepare.ipynd notebook - Don't run the full notebook

1. run everything until - Dicom images to pickled numpy

2.  If you don’t want to inflate the zipped training files run the next cell – this will take some time and you’ll still need 500GB of free space

3. You will need to run the cells after unzipping the training data, if you didn’t unzip it, run this notebook on Kaggle kernel with the competition’s dataset and then download the output files ('dicom_data.csv' and 'full_train.csv') and save them to the data folder.

Training Base Models and prepare feature vectors
---------------------
Run ‘basic learner.ipynb’ 

The 2nd cell in this notebook contains the model type and parameters which could be changed.

Beware Running the full notebook can take about 12H or more

If you only want to prepare the feature vectors, skip this step

Prepare feature vectors 
--------------------------
If you used the trained models and skipped the previous step you’ll need to prepare the feature vectors for training the transformer using the notebook ‘calculate features.ipynb’.

The 2nd cell holds the configuration as before.

TrainingTransformer models 
--------------------------
Run ‘transformer learner.ipynb’ 

Here there is some more names to set in the 2nd cell, follow the example

Inference
-------------------------- 
Use Kaggle_kernels/rsna2020-inference-2nd-opinion.ipynb for inference
This notebook was only tested on Kaggle, hence it is better to use [the notebook on Kaggle]( https://www.kaggle.com/yuval6967/rsna2020-inference-2nd-op-final) 

You’ll need to create and upload a dataset with your trained models, Edit the 3th cell and follow this notebooks example.

Full Models' waeights
-------------------------- 
The models' weights are loaded as public dataset on kaggle platform: [https://www.kaggle.com/yuval6967/rsna2020-models](https://www.kaggle.com/yuval6967/rsna2020-models)

