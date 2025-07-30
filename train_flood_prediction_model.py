import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from joblib import dump

# 1. LOAD DATA
df = pd.read_csv("/content/FloodSamples_20240101.csv")
df = df.dropna()  # Drop any incomplete rows

# 2. FEATURE PREPARATION
features = [
    'ndwi_post', 'mndwi_post', 'awei_post', 'VV',
    'elevation', 'slope', 'Map', 'jrc_seasonality'
]
target = 'flood'

# One-hot encode the landcover categorical variable
X = pd.get_dummies(df[features], columns=['Map'])
y = df[target]

# 3. STRATIFIED TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 4. MODEL TRAINING & CROSS-VALIDATION
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
print("CV ROC AUC scores:", cvs)
print("Mean CV ROC AUC:", np.mean(cvs))

rf.fit(X_train, y_train)

# 5. EVALUATE ON TEST SET
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))

# 6. PROBABILITY CALIBRATION (ISOTONIC REGRESSION)
calibrated_model = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
calibrated_model.fit(X_train, y_train)
y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
print("Test ROC AUC (Calibrated):", roc_auc_score(y_test, y_proba_calibrated))

# 7. PLOT FEATURE IMPORTANCE
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
plt.tight_layout()
plt.show()

# 8. SAVE MODEL FOR FUTURE USE
dump(rf, 'rf_flood_predictor.joblib')
