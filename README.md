# Comparative Analysis of Regression Models for Predicting Unicorn Startup Valuations in India

## Project Overview

This project explores the application of machine learning regression techniques to predict the valuations of Indian unicorn startups. Using a dataset of privately held Indian startups valued at $1 billion or more (as of June 2023), this study compares **Linear Regression**, **Ridge Regression**, and **Lasso Regression** in terms of prediction accuracy and model performance.

The analysis evaluates each model using key metrics like **Root Mean Squared Error (RMSE)** and **R² score**, with a focus on understanding how regularization and feature selection impact predictive performance.

---

## Objective

The primary objectives of this project are:

* Compare the performance of Linear, Ridge, and Lasso regression models in predicting Indian unicorn startup valuations.
* Examine the effect of regularization on model accuracy and generalization.
* Demonstrate the importance of data preprocessing, feature engineering, and model selection in real-world valuation problems.
* Provide a visual and quantitative comparison of model performance.

---

## Dataset

The dataset used in this project contains information about Indian unicorn startups, including:

* **Company Name**
* **Sector**
* **Founding Year**
* **Location**
* **Entry Valuation ($B)**
* **Current Valuation ($B)**
* **Notable Investors**

> Source: CSV file `Indian Unicorn startups 2023 updated.csv`

**Key Notes on the Dataset:**

* Missing values are handled through removal or imputation strategies.
* Categorical features like `Sector` and `Location` are one-hot encoded.
* Numerical features such as `Entry Valuation`, `Entry Year`, and `Investor Count` are standardized for uniform scale.

---

## Data Preprocessing

The following preprocessing steps were implemented:

1. **Dropping irrelevant columns:** `No.` and `Company` were removed.
2. **Renaming columns:** `Entry Valuation^^ ($B)` → `Entry_Valuation`, `Valuation ($B)` → `Valuation`.
3. **Extracting year:** Converted `Entry` to numeric `Entry_year`.
4. **Investor Count feature:** Counted the number of investors per startup from `Select Investors`.
5. **Feature separation:** Divided features into categorical and numerical.
6. **Encoding & Scaling:**

   * Categorical features → OneHotEncoder
   * Numerical features → StandardScaler
7. **Pipeline creation:** Used `ColumnTransformer` and `Pipeline` for reproducible preprocessing.
8. **Train-test split:** 80% training, 20% testing.

---

## Models Used

The following regression models were compared:

| Model                 | Description                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Linear Regression** | Simple regression without regularization.                                                                      |
| **Ridge Regression**  | Linear regression with L2 regularization, controlling extreme coefficient values.                              |
| **Lasso Regression**  | Linear regression with L1 regularization, shrinking less important coefficients to zero for feature selection. |

All models were evaluated using **RMSE** and **R² score** on both training and testing data.

---

## Output

### 1. Actual vs Predicted Scatter Plot

* Visualizes prediction accuracy of each model.
* **Linear Regression:** Wide scatter → less accurate.
* **Ridge Regression:** Moderate scatter → balanced and reliable.
* **Lasso Regression:** Compressed predictions → overly simplified.

### 2. Train vs Test RMSE Bar Chart

* Compare training and testing errors.
* **Ridge Regression** shows the lowest test RMSE and consistent performance.

### 3. Best Model

* Selected based on lowest test RMSE and highest R² score.
* **Ridge Regression** is the best-performing model.

---

## Results Summary

| Model             | Train RMSE | Test RMSE | R² Score | Prediction Pattern     |
| ----------------- | ---------- | --------- | -------- | ---------------------- |
| Linear Regression | High       | High      | Lower    | Widely scattered       |
| Ridge Regression  | Moderate   | Low       | High     | Balanced, accurate     |
| Lasso Regression  | Low        | Higher    | Moderate | Compressed, simplified |

**Inference:**
Ridge Regression provides the best trade-off between bias and variance, making it the most suitable model for predicting startup valuations in this dataset.

---

## Conclusion

This project demonstrates how regression models can predict Indian unicorn startup valuations and highlights:

* The importance of **data preprocessing** and **feature engineering**.
* How **regularization** improves predictive stability (Ridge and Lasso).
* Ridge Regression as the most effective model, balancing accuracy and generalization.
* Insights into how startup characteristics like sector, entry valuation, and investor count influence valuation.

---

## Dependencies

* Python 3.9+
* pandas
* numpy
* matplotlib
* scikit-learn

---

## Future Improvements

* Incorporate more advanced models like **Random Forest** or **XGBoost**.
* Include additional features such as revenue, employee count, and funding rounds.
* Explore **hyperparameter tuning** for Ridge and Lasso models.
* Create a **web-based app** for interactive valuation predictions.
