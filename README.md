# Bank Customer Transaction Prediction (PRCP-1003)

## Overview
A machine learning classification project to predict whether a bank customer will make a specific transaction in the future (`target = 1`) or not (`target = 0`), based on 200 anonymized continuous features (`var_0` to `var_199`).

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
- **Deployment:** Flask, HTML/CSS

## Key Highlights & Models Evaluated
- Handled severe class imbalance (~90:10) using stratified splitting and balanced class weights.
- Evaluated multiple models using **ROC-AUC** as the primary metric:
  - Logistic Regression (Balanced)
  - Decision Tree Classifier
  - **Random Forest Classifier (Tuned via RandomizedSearchCV — ROC-AUC: ~0.825)**

## How to Run Locally
1. Run the Jupyter Notebook (`PRCP-1003_Notebook.ipynb`) to train the model and generate `model.pkl`.
2. Start the Flask web server:
   ```bash
   python deploy.py
