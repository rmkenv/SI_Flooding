import joblib
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio import features
from shapely.geometry import shape
import logging

from stac_data_handler import STACDataHandler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_flood_from_stac(bbox: list, post_daterange: str, resolution: int, model_path: str) -> gpd.GeoDataFrame:
    """
    Generates flood predictions from STAC data for a given area of interest.

    :param bbox: Bounding box [minx, miny, maxx, maxy] for the analysis area.
    :param post_daterange: Date range for the "post-event" data.
    :param resolution: The spatial resolution in meters for the analysis.
    :param model_path: Path to the pre-trained .joblib model file.
    :return: A GeoDataFrame with flood predictions.
    """
    # 1. Get predictor data from STAC
    logger.info("Fetching predictor data from STAC...")
    stac_handler = STACDataHandler()
    predictor_ds = stac_handler.get_predictor_dataset(
        bbox=bbox,
        post_daterange=post_daterange,
        resolution=resolution
    )

    if not predictor_ds.coords:
        logger.warning("Predictor dataset is empty. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(columns=['geometry', 'flood_predicted'], crs=f"EPSG:{predictor_ds.rio.crs.to_epsg()}")

    # 2. Load the pre-trained model
    logger.info(f"Loading pre-trained model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise

    # The model was trained on a dataframe with specific columns. We need to replicate that structure.
    training_cols = model.feature_names_in_

    # 3. Prepare the dataset for prediction
    logger.info("Preparing data for prediction...")
    # Convert the xarray Dataset to a pandas DataFrame
    df_pred = predictor_ds.to_dataframe().reset_index()
    df_pred = df_pred.dropna()

    if df_pred.empty:
        logger.warning("DataFrame is empty after dropping NaNs. No data to predict.")
        return gpd.GeoDataFrame(columns=['geometry', 'flood_predicted'], crs=f"EPSG:{predictor_ds.rio.crs.to_epsg()}")

    # One-hot encode the landcover 'Map' feature
    # This needs to be done carefully to match the columns the model was trained on.
    df_pred = pd.get_dummies(df_pred, columns=['Map'], prefix='Map')

    # Align columns with the training data
    # Add missing columns (if any) and fill with 0
    for col in training_cols:
        if col not in df_pred.columns:
            df_pred[col] = 0

    # Ensure the order of columns is the same as in training
    df_pred = df_pred[training_cols]


    # 4. Run prediction
    logger.info("Running flood prediction...")
    predictions = model.predict(df_pred)

    # 5. Convert predictions back to a raster format
    logger.info("Converting predictions to raster...")
    # Create a new DataFrame with coordinates and predictions
    output_df = pd.DataFrame({'prediction': predictions}, index=df_pred.index)

    # Merge back with original coordinates to place predictions in the right spatial location
    # We need to create a full coordinate grid first
    full_coords_df = predictor_ds.to_dataframe().reset_index()[['x', 'y']]
    # Merge predictions, filling non-predicted areas with 0 (no flood)
    result_df = full_coords_df.merge(output_df, left_index=True, right_index=True, how='left').fillna(0)

    # Convert the DataFrame back to an xarray DataArray
    prediction_da = result_df.set_index(['y', 'x']).to_xarray()['prediction']
    prediction_da.rio.write_crs(predictor_ds.rio.crs, inplace=True)
    prediction_da.rio.write_transform(predictor_ds.rio.transform(), inplace=True)


    # 6. Vectorize the prediction raster
    logger.info("Vectorizing prediction raster...")
    # Get shapes of flooded areas (where prediction == 1)
    shapes = features.shapes(
        source=prediction_da.astype(np.int32).values,
        mask=(prediction_da == 1).values,
        transform=prediction_da.rio.transform()
    )

    # Create a GeoDataFrame from the shapes
    geometries = []
    flood_values = []
    for geom, value in shapes:
        if value == 1:
            geometries.append(shape(geom))
            flood_values.append(int(value))

    if not geometries:
        logger.info("No flooded areas were predicted.")
        return gpd.GeoDataFrame(columns=['geometry', 'flood_predicted'], crs=prediction_da.rio.crs)

    gdf_flooded = gpd.GeoDataFrame(
        {'flood_predicted': flood_values},
        geometry=geometries,
        crs=prediction_da.rio.crs
    )

    logger.info(f"Vectorization complete. Found {len(gdf_flooded)} flooded polygons.")
    return gdf_flooded

if __name__ == '__main__':
    # Example usage:
    # This requires the model file 'rf_flood_predictor.joblib' to be present.
    # Define a bounding box for an area of interest (e.g., Miami)
    miami_bbox = [-80.8738, 25.1398, -80.1308, 25.9564]

    # Define the "post-flood" date range
    post_flood_daterange = "2024-01-01/2024-02-01"

    # Define resolution
    analysis_resolution = 30 # meters

    # Path to the model
    model_path = 'rf_flood_predictor.joblib'

    try:
        # Run the prediction
        flood_gdf = predict_flood_from_stac(
            bbox=miami_bbox,
            post_daterange=post_flood_daterange,
            resolution=analysis_resolution,
            model_path=model_path
        )

        # Save the output to a file
        if not flood_gdf.empty:
            output_filename = 'stac_flood_predictions.geojson'
            flood_gdf.to_file(output_filename, driver='GeoJSON')
            logger.info(f"Flood predictions saved to {output_filename}")
        else:
            logger.info("No flood predictions to save.")

    except Exception as e:
        logger.error(f"An error occurred during the prediction process: {e}", exc_info=True)
