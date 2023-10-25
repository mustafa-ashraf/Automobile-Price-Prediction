# Automobile Price Prediction Project
![ Automobile Price Prediction](https://www.inovex.de/wp-content/uploads/2019/02/Price-Prediction-in-Online-Car-Marketplaces-1500x880.png) <!-- Replace 'project_image.jpg' with the actual image file and path -->

## Overview

This project aims to predict automobile prices based on various features and attributes of the vehicles. It uses a machine learning model, specifically a Linear Regression model, to make price predictions. The project includes data preprocessing, feature engineering, and model evaluation.

## Project Steps

### Importing Libraries
The project starts by importing necessary libraries for data analysis and machine learning. The libraries used include NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn for machine learning components.

### Read and Inspect the Data
The dataset is read from a CSV file into a Pandas DataFrame. The data is inspected to understand its dimensions, column names, data types, and missing values. The dataset contains 205 rows and 26 columns.

### Dealing with Null Values
Handling missing values is a crucial part of the project. The 'normalized-losses' column has the most missing values, and these are filled using the mean of the column. Other columns with missing values are also imputed using appropriate strategies.

### Categorical Feature Transformation
Categorical columns are transformed using one-hot encoding, and some columns are mapped to numerical values. This prepares the data for machine learning.

### Data Correlation
Data correlation is analyzed using a heatmap and feature correlation. Correlations between numeric features are visualized and explored.

### Feature Reduction
Feature reduction is performed by dropping some columns from the dataset. This step is essential for simplifying the model and removing irrelevant or highly correlated features.

### Categorical Feature Transformation (Again)
One-hot encoding is applied again to the reduced dataset to ensure the transformed data is up-to-date.

### Splitting the Dataset
The dataset is split into training and testing sets for both features and the target variable. This allows for model training and evaluation.

### Numerical Feature Scaling
Feature scaling is applied to both feature variables (X) and the target variable (y) using Min-Max scaling. This step is important for ensuring that all features have the same scale and making the model more effective.

### Applying Linear Regression
A Linear Regression model is initialized, fitted to the training data, and used to make predictions on the scaled testing data.

### Evaluating the Model
The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Both scaled and original values are considered for evaluation. The project also includes a scatter plot to visualize the actual vs. predicted prices.

## Conclusion

This project demonstrates a comprehensive process of building a machine learning model for automobile price prediction. It covers data preprocessing, feature engineering, model training, and evaluation. The model's performance is assessed using standard regression metrics, and the results show how well the model predicts automobile prices.
