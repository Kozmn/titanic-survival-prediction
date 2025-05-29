## Project Overview

The `titanic_survival_prediction.py` script covers the entire machine learning workflow, from data loading and preprocessing to model evaluation and comparison, including the application of SMOTE for handling imbalanced datasets.

## Script Description:

This script performs the following key steps:

1.  **Data Loading**:
    * Loads `train_data.csv` [cite: 45] and `test_data.csv` [cite: 1] into pandas DataFrames.
2.  **Data Cleaning and Preprocessing**:
    * Drops irrelevant columns (`Name`, `Ticket`, `Cabin`).
    * Handles missing values:
        * Removes rows where `Embarked` is missing.
        * Fills missing `Age` values with the median age.
    * Removes any duplicate rows.
    * Separates features (`X`) from the target variable (`y`, which is `Survived`).
    * Defines numerical and categorical features for distinct preprocessing.
3.  **Feature Engineering (Preprocessing Pipelines)**:
    * **Numerical Features**: Missing values are imputed with the median, and features are scaled using `StandardScaler`.
    * **Categorical Features**: Converted into numerical format using `OneHotEncoder` (dropping the first category to avoid multicollinearity).
    * These transformations are encapsulated within `sklearn` Pipelines and a `ColumnTransformer` for efficient and consistent preprocessing.
4.  **Data Splitting**:
    * The training data (`X`, `y`) is split into a training/validation set (80%) and a final hold-out test set (20%), stratified by the `Survived` target variable to maintain class distribution.
5.  **Model Comparison (Cross-Validation)**:
    * Compares the performance of three common classification models: `Logistic Regression`, `Random Forest`, and `Support Vector Machine (SVM)`.
    * Each model is evaluated using 5-fold `StratifiedKFold` cross-validation on the `X_train_val` set.
    * Performance metrics (Accuracy, Precision, Recall, F1 Score) are calculated and averaged across the folds.
    * A comparison table and a bar plot (showing Accuracy and F1 Score) are generated to visualize model performance.
6.  **Final Model Evaluation (without SMOTE)**:
    * The `Random Forest Classifier` (often a strong performer) is chosen as a baseline.
    * A pipeline is trained on `X_train_val` and evaluated on `X_test_final`.
    * A classification report and confusion matrix are printed and plotted.
7.  **Final Model Evaluation (with SMOTE)**:
    * To address potential class imbalance, `SMOTE` (Synthetic Minority Over-sampling Technique) is integrated into the `Random Forest` pipeline.
    * This pipeline is trained on `X_train_val` and evaluated on `X_test_final`.
    * A classification report and confusion matrix are printed and plotted for the SMOTE-enhanced model.
8.  **Performance Comparison (SMOTE vs. No SMOTE)**:
    * A final comparison table and bar plot are generated, highlighting the impact of SMOTE on Accuracy, Precision, Recall, and F1 Score for the Random Forest model on the final test set.

## Dataset

The project uses the classic Titanic dataset, split into two CSV files:
* `train_data.csv`: Contains passenger information including `Survived` status, which is the target variable[cite: 45].
* `test_data.csv`: Contains passenger information for prediction, without the `Survived` column[cite: 1].

Key features used in the model include:
* `Pclass`: Passenger Class (1st, 2nd, 3rd)
* `Sex`: Gender (male/female)
* `Age`: Age in years
* `SibSp`: Number of siblings/spouses aboard the Titanic
* `Parch`: Number of parents/children aboard the Titanic
* `Fare`: Passenger fare
* `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
