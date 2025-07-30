import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load # For saving and loading the model
import os # For checking file existence and creating directories

# Scikit-learn modules for modeling and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay
)
from sklearn.impute import SimpleImputer # For basic imputation of missing values
from sklearn.inspection import permutation_importance # For more robust feature importances
from sklearn.calibration import CalibratedClassifierCV # For probability calibration

# --- Configuration Section ---
# Centralized parameters for easy modification and script configuration.
CONFIG = {
    "csv_path": "xyz.csv", # Path to your GEE-exported CSV
    "features": [
        'ndwi_post',      # Corrected to match CSV column name
        'mndwi_post',     # Corrected to match CSV column name
        'awei_post',      # Corrected to match CSV column name
        'VV',             # Corrected to match CSV column name
        'elevation',      # Corrected to match CSV column name
        'slope',          # Corrected to match CSV column name
        'Map',            # Corrected to match CSV column name
        'jrc_seasonality' # Corrected to match CSV column name
    ],
    "categorical_features": ['Map'], # Corrected to match CSV column name for the categorical feature
    "target": 'flood', # The label: 1=flood, 0=no flood
    "test_size": 0.25, # Proportion of the dataset to include in the test split
    "random_state": 42, # Seed for reproducibility of splits and model training
    "n_splits_cv": 5, # Number of folds for cross-validation
    "n_iter_random_search": 50, # Number of parameter settings to sample for RandomizedSearchCV
    "output_model_path": 'rf_flood_predictor.joblib', # File path for saving the trained model
    "plot_dir": 'plots' # Directory for saving generated plots (e.g., ROC, PR curves, feature importances)
}

# --- 1. LOAD DATA ---
def load_data(csv_path):
    """
    Loads data from a CSV file with robust error handling.
    Args:
        csv_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame or None: Loaded DataFrame if successful, None otherwise.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}. Please ensure it exists.")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns.")
        print("First 5 rows of data:\n", df.head())
        return df
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return None

# --- 2. DATA CLEANING & FEATURE PREPARATION ---
def preprocess_data(df, features, categorical_features, target):
    """
    Performs data cleaning (imputation) and feature preparation (one-hot encoding).
    Args:
        df (pd.DataFrame): The raw input DataFrame.
        features (list): List of feature column names.
        categorical_features (list): List of categorical feature column names to one-hot encode.
        target (str): Name of the target variable column.
    Returns:
        tuple: (X (pd.DataFrame), y (pd.Series)) prepared features and target.
    """
    # Create a copy to avoid modifying the original DataFrame and potential SettingWithCopyWarning
    df_processed = df.copy()

    # Handle missing values:
    # 1. For numerical features, use median imputation (robust to outliers).
    numerical_features = [f for f in features if f not in categorical_features]
    for col in numerical_features:
        if col in df_processed.columns and df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"Imputed missing values in numerical column '{col}' with median: {median_val}")
        elif col not in df_processed.columns:
            print(f"Warning: Numerical feature '{col}' not found in DataFrame. Skipping imputation for this column.")

    # 2. For categorical features, convert NaNs to a specific 'missing' category before one-hot encoding.
    for col in categorical_features:
        if col in df_processed.columns:
            if df_processed[col].isnull().any():
                df_processed[col].fillna('missing_category', inplace=True)
                print(f"Imputed missing values in categorical column '{col}' with 'missing_category'")
            # Ensure categorical columns are of 'str' type for consistent one-hot encoding
            df_processed[col] = df_processed[col].astype(str)
        else:
            print(f"Warning: Categorical feature '{col}' not found in DataFrame. Skipping processing for this column.")

    # Ensure all required features are present before one-hot encoding
    missing_expected_features = [f for f in features if f not in df_processed.columns]
    if missing_expected_features:
        raise ValueError(f"Missing required features in DataFrame: {missing_expected_features}. "
                         "Please check your GEE export and Python CONFIG['features'] list.")

    # One-hot encode specified categorical features.
    # `dummy_na=False` because we've already handled NaNs by imputing them as 'missing_category'.
    try:
        X = pd.get_dummies(df_processed[features], columns=categorical_features, dummy_na=False)
    except KeyError as e:
        print(f"Error during one-hot encoding: {e}. Check if categorical_features exist in df_processed after imputation.")
        raise

    # Extract target variable
    if target not in df_processed.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    y = df_processed[target]

    # Final check: Remove any rows where the target variable is still NaN (should be rare if GEE samples are good)
    if y.isnull().any():
        nan_target_indices = y[y.isnull()].index
        X = X.drop(nan_target_indices)
        y = y.drop(nan_target_indices)
        print(f"Removed {len(nan_target_indices)} rows due to NaN values in the target variable.")

    print(f"Data after preprocessing: X shape {X.shape}, y shape {y.shape}.")
    return X, y

# --- 3. MODEL TRAINING & OPTIMIZATION ---
def train_model(X_train, y_train, n_iter_search):
    """
    Trains a RandomForestClassifier using RandomizedSearchCV for hyperparameter tuning.
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        n_iter_search (int): Number of parameter settings to sample for RandomizedSearchCV.
    Returns:
        sklearn.ensemble.RandomForestClassifier: The best trained model.
    """
    # Define the parameter distribution for RandomizedSearchCV.
    # These ranges can be tuned further if needed.
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500], # Number of trees in the forest
        'max_depth': [None, 10, 20, 30], # Maximum depth of the tree (None means unlimited)
        'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
        'max_features': ['sqrt', 'log2', 0.8, 1.0], # Number of features to consider when looking for the best split
        'class_weight': ['balanced', 'balanced_subsample'] # Handle class imbalance
    }

    # Base classifier with parallel processing and reproducibility
    base_rf = RandomForestClassifier(random_state=CONFIG["random_state"], n_jobs=-1)

    # Stratified K-Fold for cross-validation within RandomizedSearchCV
    # Ensures each fold has roughly the same proportion of target classes as the full dataset.
    cv_strategy = StratifiedKFold(n_splits=CONFIG["n_splits_cv"], shuffle=True, random_state=CONFIG["random_state"])

    # Randomized Search Cross-Validation
    random_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter_search, # Number of parameter settings that are sampled
        cv=cv_strategy,
        scoring='roc_auc', # Metric to optimize
        random_state=CONFIG["random_state"],
        n_jobs=-1, # Use all available CPU cores
        verbose=2 # Higher verbosity shows more details during search
    )

    print(f"\nStarting RandomizedSearchCV with {n_iter_search} iterations and {CONFIG['n_splits_cv']}-fold CV...")
    random_search.fit(X_train, y_train)

    print("\n--- RandomizedSearchCV Results ---")
    print("Best parameters found:", random_search.best_params_)
    print(f"Best cross-validation ROC-AUC: {random_search.best_score_:.4f}")

    # The best_estimator_ is the model trained with the best parameters on the full X_train.
    best_rf = random_search.best_estimator_
    return best_rf

# --- 4. EVALUATION ---
def evaluate_model(model, X_test, y_test, X_train_columns, plot_dir, y_train_for_calibration=None):
    """
    Evaluates the model on the test set and generates various plots.
    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        X_train_columns (list): Column names from the training features, needed for permutation importance.
        plot_dir (str): Directory to save plots.
        y_train_for_calibration (pd.Series, optional): Training target used for CalibratedClassifierCV.
                                                    Pass if you want to calibrate based on original training labels.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (flood=1)

    print("\n--- Model Evaluation on Holdout Test Set ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # Classification report includes Accuracy, Precision, Recall, F1-score for each class
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Holdout Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Holdout Test F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Holdout Test Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Holdout Test Recall: {recall_score(y_test, y_pred):.4f}")

    # Create directory for plots if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Plot ROC Curve
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
    plt.close(fig_roc) # Close plot to free memory

    # Plot Precision-Recall Curve (Highly recommended for imbalanced datasets)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
    ax_pr.set_title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'precision_recall_curve.png'))
    plt.close(fig_pr)

    # Feature Importances (Permutation Importance for robustness)
    print("\n--- Feature Importances (Permutation Importance) ---")
    # Permutation importance is more robust than Gini importance, calculated on test data.
    # Pass X_test as a DataFrame to retain column names for the output Series.
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=CONFIG["random_state"], n_jobs=-1
    )
    # Ensure the indices match the original feature names after one-hot encoding
    perm_importances = pd.Series(result.importances_mean, index=X_train_columns)
    perm_importances_sorted = perm_importances.sort_values(ascending=False)
    print(perm_importances_sorted)

    fig_fi, ax_fi = plt.subplots(figsize=(12, 7))
    perm_importances_sorted.plot(kind='bar', ax=ax_fi)
    ax_fi.set_title('Permutation Feature Importances (Test Set)')
    ax_fi.set_ylabel('Mean decrease in ROC-AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'permutation_feature_importances.png'))
    plt.close(fig_fi)

    # Optional: Probability Calibration (Isotonic Regression)
    if y_train_for_calibration is not None:
        print("\n--- Probability Calibration (Isotonic Regression) ---")
        # 'prefit' method means the estimator is already fitted, CalibratedClassifierCV uses its probabilities
        # The internal cross-validation for CalibratedClassifierCV is used for calibrating.
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=CONFIG["n_splits_cv"])
        # Fit on training data (probabilities and true labels)
        calibrated_model.fit(X_train_columns, y_train_for_calibration)
        y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        print(f"Holdout Test ROC-AUC (Calibrated): {roc_auc_score(y_test, y_proba_calibrated):.4f}")
        # You could also plot reliability diagrams here if desired.
    else:
        print("\nSkipping probability calibration as y_train_for_calibration was not provided to evaluate_model.")


# --- 5. SAVE MODEL ---
def save_model(model, path):
    """
    Saves the trained model to a file using joblib.
    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        path (str): File path to save the model.
    """
    try:
        dump(model, path)
        print(f"\nModel successfully saved to {path}")
    except Exception as e:
        print(f"Error saving model to {path}: {e}")

# --- Main Execution Flow ---
def main():
    """Orchestrates the entire machine learning workflow."""
    print("--- Starting Flood Prediction ML Script ---")

    # 1. Load Data
    df = load_data(CONFIG["csv_path"])
    if df is None:
        print("Data loading failed. Exiting script.")
        return

    # 2. Data Cleaning & Feature Preparation
    try:
        X, y = preprocess_data(df, CONFIG["features"], CONFIG["categorical_features"], CONFIG["target"])
    except ValueError as e:
        print(f"Data preprocessing failed: {e}. Exiting script.")
        return

    # Store the final column names after one-hot encoding for consistent use in evaluation and prediction
    processed_feature_columns = X.columns.tolist()

    # 3. Stratified Train/Test Split
    # Stratify ensures that the proportion of flood/no-flood samples is maintained in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"]
    )
    print(f"\nDataset split into: {len(y_train)} training samples, {len(y_test)} test samples.")

    # 4. Model Training & Optimization (with RandomizedSearchCV)
    best_rf_model = train_model(X_train, y_train, CONFIG["n_iter_random_search"])

    # 5. Evaluation on Holdout Set
    # Pass processed_feature_columns (which are X.columns) for permutation importance and calibration.
    evaluate_model(best_rf_model, X_test, y_test, processed_feature_columns, CONFIG["plot_dir"], y_train_for_calibration=y_train)

    # 6. Save Model
    save_model(best_rf_model, CONFIG["output_model_path"])

    print("\n--- Script completed successfully! ---")
    print(f"Check the '{CONFIG['plot_dir']}' directory for evaluation plots.")

if __name__ == "__main__":
    main()
