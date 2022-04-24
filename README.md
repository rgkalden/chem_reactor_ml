# Applying Machine Learning to Industrial Chemical Reactors

This repository contains files related to the application of Machine Learning to Industrial Chemical Reactors. Two different POC's have been developed for predicting reactor performance and improving process safety, and the concepts have been developed in Jupyter Notebooks in the `Proof of Concept` folder. The goal of this project is to further advance the work on predicting reactor performance using machine learning, and more background detail is provided below.

**Background Information**

Many products from industrial chemical processes are manufactured in a chemical reactor, the type of equipment used to carry out chemical reactions at a large scale. Feed chemicals are sent to the reactor and then the products are withdrawn, either in a batch or continuous process. If the reaction is endothermic (absorbs heat) or exothermic (releases heat), then energy must be added or removed from the reactor to control the reaction temperature. The chemical reaction that produces the desired product may be joined by side reactions that produce undesired products.

The design of an efficient chemical reactor must take into consideration the feed conditions, energy requirements, side reactions, as well as the desired product quality. The economics of the process may depend on the ability to create a product of a certain quality, for example, an inefficient reactor may require additional equipment to separate out the desired product from the undesired products. A useful performance metric to measure the amount of desired product is the **overall yield (Y)**, defined generally as:

<img src="https://render.githubusercontent.com/render/math?math=\color{white}Y = \frac{amount \space of \space desired \space product \space produced}{amount \space of \space feed \space component \space reacted}">

In other words, the overall yield is a fraction (values from 0 to 1) representing how much product is created for the amount of feed that is sent to the reactor. Once a chemical reactor is in operation, the yield must still be maximized by adapting to changes in process conditions in order to maintain the desired product quality. 

**Why attempt to apply Machine Learning to Industrial Chemical Reactors?**

The purpose of this project is to demonstrate how machine learning can be used to model the relationship between the process conditions and the overall yield, so that a real time prediction for reactor performance can be provided to those involved with operating the reactor. A real time prediction for reactor performance given by a machine learning model can be part of a system that raises a warning if the yield decreases below a target value, and then provides information on which process conditions are causing the decrease so that action can be taken to return to maximum yield.

**Value Proposition**

Chemical reactors involve many complex chemical and physical processes that can make modelling by traditional means difficult, despite strong knowledge of the underlying scientific principles involved, so machine learning is an ideal technique to experiment with by taking advantage of historical operating data to build a statistical model instead of one based on first principles.

The advantage of using a machine learning model to predict the yield over taking samples of the reactor products and calculating the yield directly is that the model can be put into use making real time predictions for the product yield to help maintain the maximum yield while minimizing the requirement for additional sampling in the future. If there is enough suitable historical data for the operating conditions of the reactor, and adequate sampling to determine historical yields, then it is likely possible to gather a dataset suitable for a model to be trained on, and develop a machine learning solution for predicting reactor performance.

**Project Objectives**

This project has the following objectives:
1. Convert POC notebook `Reactor Model Part 1 v5.ipynb` into Python scripts ✔
2. Build a model training pipeline to run locally ✔
3. Create capability to generate new synthetic data to feed to the trained model ✔
4. Develop inference pipeline ✔
5. Build Docker image for the model training pipeline, include inferencing capability, and run a container to generate predictions locally ✔
6. Train model on AWS Sagemaker, deploy to endpoint, and generate predictions. Load new data from AWS S3 and save predictions to S3 ✔
7. Monitor data drift using the model monitor functionality of Sagemaker 🚧

## 1. Getting Started

To get started with this project, it is recommended to take the following steps.

### 1.1 Background Detail for this Project

To become more familiar with the concepts behind this project it is recommended to read the notebook `Reactor Model Part 1 v5.ipynb` in the `Proof of Conept` folder before browsing further. Understanding the concepts in the notebook will make it easier to understand the functions of the python scripts, as they have been adapted from the notebook.

### 1.2 Environment Setup

The Python package requirements are provided in `requirements.txt`. The work in this repository was mainly developed using the VS Code IDE, and instructions for creating a virtual environment with the `venv` package in the terminal are given in `terminal_commands.txt`. 

>NOTE: Permissions for the Powershell terminal may need to be modified in order to activate the virtual environment, see https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.2

>NOTE: when using VSCode, change the Jupyter Extension Notebook File Root setting from `${fileDirname}` to `${workspaceRoot}`
>in order to have relative paths work when importing csv. Setting makes the project folder the root directory,
>not the folder where this notebook is saved

>NOTE: In order to run the pipeline `pipeline.py` it may be required to add paths to PYTHONPATH for the modules saved in their respective folders, with the following code:

    # Check to add module folders to python path, so the python interpreter can find modules to import
    import sys
    sys.path.append(r'<path to project>\chem_reactor_ml\data_generation')
    sys.path.append(r'<path to project>\chem_reactor_ml\data_preparation')
    sys.path.append(r'<path to project>\chem_reactor_ml\model_training')
    print(sys.path)

## 2. Folder Structure

### 2.1 Proof of Concept Reactor Model

This folder contains notebooks developing the proofs of concept of two application of machine learning to chemical reactors. These notebooks should be consulted for the background information and theory behind the applications themselves. Information regarding the generation of a synthetic dataset is also contained in these notebooks. 

This project focuses on further developing the concepts in the notebook for Part 1.

Part 1: Project demonstrating how machine learning can be used to model the relationship between the process conditions and the overall yield of a chemical reactor.

Part 2: Project exploring how machine learning can improve process safety in chemical reactors by predicting if a reactor runaway will occur.

### 2.2 Data Generation

Folder containing python scripts for generating a synthetic dataset.

### 2.3 Data Visualization

Folder containing a jupyter notebook to do some simple visualization and EDA on the synthetic dataset.

### 2.4 Data Preparation

Folder contains a python script for preparing the data, including a train/test split and standaradizing the data.

### 2.5 Model Training

Folder contains a script to train a random forest model.

### 2.6 Model Training Pipeline

In this folder, the `pipeline.py` script can be executed to run the entire model training pipeline.

### 2.7 Docker Implementation

Package code from the previous sections into a docker container.

### 2.8 AWS Implementation

Folder containing files related to the development of the machine learning model on AWS Sagemaker and the implementation of the pipeline on AWS.

## 3. Running the Pipelines

There are four pipelines that have been developed in this project:
1. Local version for model training based on Python modules
2. Local inferencing pipeline based on locally trained model
3. Local Docker implementation
4. Cloud AWS implementation

Instructions for running the pipelines are as follows:

### 3.1 Running the Model Training Pipeline

The model training pipeline can be executed by running `pipeline.py`, from the `model_training_pipeline` folder. The model will be saved as a joblib file by default.

### 3.2 Running the Local Inferencing Pipeline

The model inferencing pipeline can be executed by running `inference_script.py`, from the project root folder.

### 3.3 Local Docker Implementation

Python modules have been copied into the `docker_implementation` folder, then modified as necessary to build a Docker image, using the `Dockerfile`. Instructions to run a container and generate predictions from new data are included in `docker_commands.txt`.

### 3.4 Cloud AWS Implementation

The AWS implementation requires two files. The `train_deploy_infer.ipynb` notebook must be run within an AWS Sagemaker notebook instance, as it relies on the `boto3` and `sagemaker` Python SDK's to interact with AWS resources such as S3. To run the notebook, the `data` folder should be copied into the notebook environment on Sagemaker. The notebook performs the following steps:
1. Creates train and test datasets in csv files
2. Loads train and test csv's to S3
3. Trains model based on training script `train.py`
4. Deploys model to an endpoint
5. Sends data to endpoint for inference, then writes predictions back to S3