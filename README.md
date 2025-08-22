# SI_Flooding: Cloud-Native Flood Prediction

**A cloud-native, GEE-free toolkit for flood risk analysis and visualization in the United States.**

This project leverages public STAC catalogs to perform on-the-fly flood prediction for any user-defined area of interest (AOI). It compares model-predicted flood areas to FEMA's Special Flood Hazard Areas (SFHA) and visualizes the results on a satellite basemap.

The entire workflow runs in a local Python environment, with no Google Earth Engine account required.

---

## Features

- **Cloud-Native Data Access:** Fetches and processes required satellite imagery (Sentinel-1, Sentinel-2), DEM, and landcover data directly from the Microsoft Planetary Computer's STAC catalog.
- **On-the-Fly Prediction:** Uses a pre-trained Random Forest model (`.joblib`) to generate flood predictions for your AOI without needing to pre-process data.
- **FEMA Comparison:** Automatically fetches FEMA NFHL flood zone polygons for your AOI via the ArcGIS REST API.
- **Risk Identification:** Identifies and highlights model-predicted flooded areas that fall outside FEMA's officially mapped SFHA zones.
- **Rich Visualization:** Generates a high-quality map visualizing the analysis results on a satellite basemap.
- **Comprehensive Outputs:** Saves the flood predictions as GeoJSON, the map as PNG and PDF, and a summary report in JSON format.
- **Legacy GEE Support:** The old Google Earth Engine-based workflow is still available for reference in the `/GEE` directory.

---

## Quickstart

Get up and running in two simple steps:

**1. Install Dependencies**

Ensure you have Python 3.8+ installed. Then, install the required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**2. Run the Analysis**

Execute the main analysis script. The script is pre-configured to run an analysis for an example area in Miami, FL.

```bash
python flood_analyzer.py
```

That's it! The script will perform the entire workflow:
1.  Fetch data from the STAC catalog.
2.  Run the flood prediction model.
3.  Fetch FEMA data.
4.  Analyze the results and generate outputs.

Outputs will be saved in a new directory named `stac_analysis_outputs/`.

---

## How It Works

The toolkit automates the following steps:

1.  **Data Curation (stac_data_handler.py):**
    -   Connects to the Microsoft Planetary Computer STAC catalog.
    -   Searches for Sentinel-1, Sentinel-2, Copernicus DEM, and ESA WorldCover data for the specified AOI and date range.
    -   Loads the data into `xarray` datasets using `stackstac`.
    -   Calculates the required features for the model (e.g., water indices like NDWI, MNDWI, AWEI).

2.  **Flood Prediction (predict_from_stac.py):**
    -   Loads the pre-trained Random Forest model (`rf_flood_predictor.joblib`).
    -   Prepares the curated data into a format suitable for the model.
    -   Applies the model to predict flooded pixels, generating a flood raster.
    -   Vectorizes the raster into polygons, creating a GeoDataFrame of flooded areas.

3.  **Analysis & Visualization (flood_analyzer.py):**
    -   Takes the GeoDataFrame of flooded areas as input.
    -   Fetches corresponding FEMA flood hazard data.
    -   Performs a spatial analysis to find predicted floods outside of FEMA zones.
    -   Generates and saves the map, GeoJSON files, and a JSON summary report.

---

## Usage and Customization

To run the analysis for a different area or time period, simply edit the parameters in the `main` function at the bottom of `flood_analyzer.py`:

```python
def main():
    # ...
    # --- EDIT THESE PARAMETERS FOR YOUR ANALYSIS ---
    # Bounding box [minx, miny, maxx, maxy] in WGS84
    miami_bbox = [-80.8738, 25.1398, -80.1308, 25.9564]

    # Date range for the analysis period
    post_flood_daterange = "2024-01-01/2024-02-01"

    # Spatial resolution in meters
    analysis_resolution = 30

    # Path to the pre-trained model
    model_path = 'rf_flood_predictor.joblib'

    # Name of the output directory
    output_dir = 'stac_analysis_outputs'
    # --- END OF PARAMETERS ---

    try:
        # 1. Get flood predictions
        flood_gdf = predict_flood_from_stac(
            bbox=miami_bbox,
            post_daterange=post_flood_daterange,
            resolution=analysis_resolution,
            model_path=model_path
        )
        # ...
        # 2. Run the analysis
        config = FloodAnalysisConfig(output_dir=output_dir)
        analyzer = FloodAnalyzer(config)
        report = analyzer.run_analysis(input_gdf=flood_gdf)
        # ...
```

---

## Requirements

This project requires Python 3.8+. The main dependencies are listed below. See `requirements.txt` for a complete list of packages and versions.

- **Core:** `pandas`, `numpy`, `scikit-learn`, `joblib`
- **Geospatial:** `geopandas`, `shapely`, `rasterio`, `contextily`
- **STAC & Cloud-Native:** `pystac-client`, `planetary-computer`, `stackstac`, `rioxarray`, `xarray`, `dask`
- **Web:** `requests`

---

## Model Training

The pre-trained model `rf_flood_predictor.joblib` was trained on data generated by the GEE script `GEE/gee_flood_feature_extraction.js`. The training script `train_flood_prediction_model.py` can be used as a reference for how to train a similar model. It expects a `FloodSamples_*.csv` file generated by the GEE script.

---

## Troubleshooting

- **Model loading errors:** Ensure the `rf_flood_predictor.joblib` file is present in the project's root directory.
- **No FEMA features found:** Your AOI may not overlap with any FEMA-mapped flood zones.
- **STAC data issues:** The availability of satellite imagery can vary. If you get an error about missing data, try adjusting the date range or AOI.
- **Basemap not displaying:** Requires an active internet connection to download map tiles.

---

## License

MIT License

---

## Acknowledgments

This project is made possible by the following open data and software:
- FEMA National Flood Hazard Layer (NFHL)
- Microsoft Planetary Computer
- Copernicus Programme (Sentinel data, Copernicus DEM)
- ESA WorldCover
- The open source Python geospatial community

---

**For questions or contributions, please open an issue or pull request!**
