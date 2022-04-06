# Applying Machine Learning to Industrial Chemical Reactors

This repository contains files related to the application of Machine Learning to Industrial Chemical Reactors. Two different POC's have been developed for predicting reactor performance and improving process safety, and the concepts have been developed in Jupyter Notebooks in the `Proof of Concept` folder. To become more familiar with the concepts behind this project it is recommended to read the notebook `Reactor Model Part 1 v5.ipynb` before browsing further.

Main objectives of this project are to:
1. Convert POC notebook into Python scripts
2. Build a model training pipeline
3. Create capability to generate new synthetic data to feed to the trained model
4. Develop inference pipeline (batch/real time)
5. Deploy model, monitor data drift

## 1. Getting Started

### 1.1 Environment Setup

The Python package requirements are provided in `requirements.txt`. The work in this repository was mainly developed using the VS Code IDE, and instructions for creating a virtual environment with the `venv` package in the terminal are given in `terminal_commands.txt`.

>NOTE: when using VSCode, change the Jupyter Extension Notebook File Root setting from `${fileDirname}` to `${workspaceRoot}`
>in order to have relative paths work when importing csv. Setting makes the project folder the root directory,
>not the folder where this notebook is saved

>NOTE: In order to run the pipeline `pipeline.py` it may be required to add paths to PYTHONPATH for the modules saved in their >respective folders, with the following code:

    # Check to add module folders to python path, so the python interpreter can find modules to import
    import sys
    sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\data_generation')
    sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\data_preparation')
    sys.path.append(r'C:\Users\rgkal\Documents\chem_reactor_ml\model_training')
    print(sys.path)

### 1.2 Running the Model Training Pipeline

The model training pipeline can be executed by running `pipeline.py`, from the `model_training_pipeline` folder. The model will be saved as a joblib file by default.

## 2. Folder Structure

### 2.1 Proof of Concept Reactor Model

This folder contains notebooks developing the proofs of concept of two application of machine learning to chemical reactors. These notebooks should be consulted for the background information and theory behind the applications themselves. Information regarding the generation of a synthetic dataset is also contained in these notebooks.

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