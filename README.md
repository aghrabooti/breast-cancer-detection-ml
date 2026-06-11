# breast-cancer-detection-ml

This repository contains a comprehensive machine learning pipeline for breast cancer detection using the Wisconsin Breast Cancer dataset. The objective is to accurately predict whether a tumor is benign or malignant based on cellular characteristics.

## Repository Structure
* `BreastCancer_ML_Project.ipynb` - Core Jupyter Notebook containing EDA, preprocessing, and model training.
* `requirements.txt` - Python dependencies needed to execute the project.

## Dataset
* **Name:** Wisconsin Breast Cancer Dataset (scikit-learn built-in)
* **Samples:** 569 rows
* **Features:** 30 continuous numerical attributes
* **Classes:** 0 = Malignant, 1 = Benign

## Project Pipeline
1. **Data Understanding & EDA:** Investigated feature distributions, outlier checking via boxplots, normality validation, and Pearson correlation matrix analysis.
2. **Preprocessing:** Cleaned missing boundaries, implemented IQR outlier capping, scaled attributes via `StandardScaler`, and performed a stratified 80/20 train/test split.
3. **Model Training:** Built and compared multiple models: Decision Tree, Naive Bayes, Logistic Regression, and unsupervised K-Means clustering.
4. **Evaluation:** Validated model classifications using Confusion Matrices, Accuracy, Precision, Recall, and F1-Score calculations, focusing heavily on reducing False Negatives.

## Requirements
To run this project locally, ensure you have Python installed, then install the dependencies:
```bash
pip install -r requirements.txt
