import pandas as pd
from joblib import load

# Step 1: Load your trained model (if you used a calibrated model, load that instead)
rf = load('rf_flood_predictor.joblib')  # or change filename as needed

# Step 2: Load the new AOI features exported from GEE (same columns as training)
new_aoi = pd.read_csv("FloodSamples_NEW_AOI.csv")  # Change path as needed

# Step 3: One-hot encode Map (landcover) and align columns with training data
X_new = pd.get_dummies(new_aoi[['ndwi_post', 'mndwi_post', 'awei_post', 'VV',
                                'elevation', 'slope', 'Map', 'jrc_seasonality']],
                       columns=['Map'])

# VERY IMPORTANT: Align feature columns for model input (use features from training)
X_new = X_new.reindex(rf.feature_names_in_, axis=1, fill_value=0)

# Step 4: Predict flood probability and class for new AOI
proba_new = rf.predict_proba(X_new)[:, 1]
pred_new = rf.predict(X_new)

# Step 5: Save or join predictions back to coordinates for GIS/mapping
new_aoi['flood_probability'] = proba_new
new_aoi['flood_predicted'] = pred_new

new_aoi.to_csv("FloodPredictions_NEW_AOI.csv", index=False)

