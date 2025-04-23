# Using Machine Learning to Predict Premier League Results

## Abstract

The project investigates the design, development, and evaluation of machine learning techniques to predict Premier League match outcomes (i.e. home win, draw, or away win) while providing probability estimates that convey prediction confidence.

The project was carried out by first collecting match data spanning the past 15 Premier League seasons (2010/11 to 2024/25). Following data collection, extensive preprocessing was conducted to clean, standardise, and engineer relevant features such as recent team form, seasonal performance averages, and key match statistics. Two final datasets—one label‑encoded for tree‑based models and one one‑hot‑encoded for linear and kernel‑based methods—are split chronologically into training and test sets to prevent data leakage.

Using Scikit-learn and Tensorflow, Logistic Regression (with L2 regularization), Random Forest, XGBoost, Support Vector Machine (SVM) and Multi-layer Perceptron (MLP) models are successfully trained and evaluated. On unseen test data, the L2-regularized Logistic Regression model achieved the highest accuracy of 65.11\%. XGBoost and the MLP followed with test accuracies of 63.55\% and 62.55\% respectively, while Random Forest and SVM both scored just over 60\% accuracy.

Finally, the models each predicted the result for upcoming 2024/25 fixtures, demonstrating practical utilisation. Whilst models reliably forecasted clear favourites with superior teams, they struggled with predicting draws and rare upsets, underscoring football's inherent unpredictability and suggesting areas for future work - such as integrating real-time data or richer contextual features.

## Code details

Data folder - contains all the datasets used in this project

Pre-processing folder - contains the code that preprocesses the dataset. This includes combining the datasets from the different seasons, feature engineering (form), feature selection, encoding as well as train-test split chronologically.

Models folder - contains all the code to train all the models. all_model_report.ipynb file also tests all the models at the same time and compares the results on a table.

Predictions folder - contains the code to predict future fixtures (code currently used to predict PL GW30 fixtures.
