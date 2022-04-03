# Applying Machine Learning to Industrial Chemical Reactors

This repository contains files related to the application of Machine Learning to Industrial Chemical Reactors. Two different POC's have been developed for predicting reactor performance and improving process safety, and the concepts have been developed in Jupyter Notebooks in the `Proof of Concept` folder.

The main objective of this work is to build a machine learning pipeline for the first application, predicting chemical reactor performance, to further develop the concept into an example of a production ready, deployable machine learning model.

## 1. Getting Started

The Python package requirements are provided in `requirements.txt`. The work in this repository was mainly developed using the VS Code IDE, and instructions for creating a virtual environment with the `venv` package in the terminal are given in `terminal_commands.txt`.

>NOTE: when using VSCode, change the Jupyter Extension Notebook File Root setting from `${fileDirname}` to `${workspaceRoot}`
>in order to have relative paths work when importing csv. Setting makes the project folder the root directory,
>not the folder where this notebook is saved

## 2. Folder Structure

### 2.1 Proof of Concept Reactor Model

Part 1: Project demonstrating how machine learning can be used to model the relationship between the process conditions and the overall yield of a chemical reactor.

Part 2: Project exploring how machine learning can improve process safety in chemical reactors by predicting if a reactor runaway will occur.

### 2.2 Data Generation

### 2.3 Data Visualization

### 2.4 Data Preparation

### 2.5 Model Training