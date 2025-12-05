IFC Investment Prediction – Machine Learning Model
Overview

This project develops a hybrid machine learning system capable of predicting IFC investment amounts (regression) and classifying environmental categories (multiclass classification). The final solution is deployed as a Streamlit application with two tabs: dataset-based predictions and new-entry predictions.

Objectives

Clean and preprocess IFC project data

Engineer meaningful numerical and categorical features

Train high-performance XGBoost models for regression and classification

Deploy the selected models within an interactive Streamlit interface

Enable users to run predictions on both existing and new project data

Models
Regression Model

Algorithm: XGBRegressor

Pipeline: preprocessing (OneHotEncoder, scaling) followed by model training

Performance:

MAE: 5.27

RMSE: 30.38

R²: 0.94

Classification Model

Algorithm: XGBClassifier

Pipeline: preprocessing + multiclass classification

Performance:

Accuracy: 0.80

F1-score: 0.78

Both models were compared with alternative methods; XGBoost provided the best overall results.

Streamlit Application
Tab 1 – Dataset and Predictions

Displays the cleaned dataset

Allows selecting a record to generate:

Investment amount prediction

Environmental category prediction

Shows summary performance metrics for both models

Tab 2 – New Entry Prediction

Users enter project attributes using dropdowns and sliders

Slider ranges are derived from the dataset’s actual min/max values

Returns predicted investment amount and environmental category

Project Structure
app.py                     # Streamlit application  
requirements.txt           # Dependencies  
reg_model.joblib           # Trained regression model  
clf_model.joblib           # Trained classification model  
ifc_clean_data.csv         # Clean dataset used by the UI  

Installation
pip install -r requirements.txt
streamlit run app.py

Deployment

The application is deployed via Streamlit Cloud. The repository contains all required files, including the trained models and dataset, ensuring consistent inference during deployment.

https://ifcinvestmenthybridml.streamlit.app/

Author

Anvarmirzo Rajabboev
Machine Learning Investment Prediction Project, 2025
