Car Price Prediction using Regression Models

 Project Overview:

This project aims to build a regression model to predict car prices based on various features such as engine size, horsepower, fuel type, and other specifications. The objective is to analyze the relationship between predictors and car price and develop a model that can accurately estimate car prices.

 Dataset:

The dataset contains information about different car attributes including:

Car name

Fuel type

Engine size

Horsepower

Car body type

Drive wheel type

Price (target variable)

The dataset was preprocessed to handle categorical variables and scaling.

 Data Preprocessing:

The following preprocessing steps were applied-

Handling categorical variables using one-hot encoding

Dropping redundant columns to avoid dummy variable trap

Use VIF and Correlation to check which independent variables are depend each other

Feature scaling using standard scaling

Train-test split of the dataset

4. Model Used

The following regression models were applied:

Lasso Regression with GridSearchCV for hyperparameter tuning

Ridge Regression with GridSearchCV for hyperparameter tuning

5. Model Evaluation

The model performance was evaluated using:

Mean Squared Error (MSE)

R² score

Training accuracy vs Test accuracy comparison

Residual analysis was also performed to check regression assumptions

Diagnostic Checks:

To validate regression assumptions, the following plots were used:

Residual vs Fitted Plot

Q-Q Plot for normality of residuals

Residual distribution analysis

Results:

The final model achieved approximately:

Training accuracy: 95%

Test accuracy: 91%

The small difference between training and test performance suggests that the model generalizes well and does not significantly overfit.

Technologies Used:

Python

pandas

numpy

matplotlib

scikit-learn

scipy
