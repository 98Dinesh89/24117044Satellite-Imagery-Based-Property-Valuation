# 24117044Satellite-Imagery-Based-Property-Valuation

Multimodal house price prediction using tabular data and satellite imagery

**Project: Satellite Imagery-Based Property Valuation**

# Overview

    This project builds a multimodal regression pipeline to predict residential property prices by combining tabular housing attributes with satellite imagery–derived visual features.
    The goal is to enhance traditional valuation models by incorporating environmental and neighborhood context captured from satellite images.

# Problem Statement

    Given historical housing data with geographic coordinates, we aim to:
        Predict property prices accurately
        Acquire satellite imagery using latitude and longitude
        Extract meaningful visual features using a CNN
        Fuse tabular and visual features into a single predictive model

# Dataset

    Tabular Data
        train(1).xlsx – training data with prices
        test2.xlsx – test data without prices

        Target: price

    Visual Data
        Satellite images fetched using latitude & longitude
        One image per unique property ID

# Project Structure

    ├── Data/
    │   ├── train_clean.csv             (preprocessed data)
    │   ├── test_clean.csv              (preprocessed data)
    │   ├── train(1).xlsx               (original data)
    │   ├── test2.xlsx                  (original data)
    │   ├── image_embeddings_train.npy
    │   ├── image_embeddings_test.npy
    │   └── images/
    │       ├── train/                  (few sample images)
    │       └── test/                   (few sample image)
    │
    ├── Notebooks/                      (Code Repository)
    │   ├── image_fetching.ipynb        (data_fetcher)
    │   ├── preprocessing.ipynb
    │   └── model_training.ipynb
    │
    ├── 24117044_file.csv
    │
    ├── 24117044_report.pdf
    │
    └── README.md

# Note:

    . All satellite image folders are not included in the repository due to size constraints, only a few sample images are provided for reference.

    . (Optional) Create and activate a virtual environment
    . Install requirements
        numpy
        pandas
        requests
        scikit-learn
        xgboost
        torch
        torchvision
        pillow

    . Precomputed CNN embeddings (image_embeddings_train.npy and image_embeddings_test.npy) are provided, so downloading satellite images is not required for normal execution

    . Image downloading and CNN feature extraction are computationally expensive and time-consuming, so their code sections are commented out

    . Some parts of the code are commented out to:
        prevent unnecessary image downloads,
        avoid long execution times,
        and ensure smooth evaluation using cached features

    . The model_training.ipynb notebook is divided into four labeled sections
        tabular model part
        feature extraction part
        tabular + satellite image model part
        Prediction for test data (Multimodal Prediction)

    . Additional details about the methodology, experiments, and results are provided in 24117044_report.pdf

# Run Code

    Data preprocessing
        run preprocessing.ipynb   (the code for saving preprocessed data is commented out)

    Download satellite images
        run image_fetching.ipynb  (to redownload satellite images, uncomment the last two cells in the notebook before running)

    Train model & generate predictions
        run model_training.ipynb  (prediction on test data is performed within this notebook)

# Tech Stack

    Data: Pandas, NumPy
    ML: Scikit-learn, XGBoost
    Visualization: Matplotlib, Seaborn
    Image Processing: PIL

    API key from https://www.maptiler.com
