# SI_Flooding

**Flood Risk Analysis and Visualization Toolkit**

This project processes model-predicted flood areas, compares them to FEMA's Special Flood Hazard Areas (SFHA), and visualizes the results on a satellite basemap for any user-defined area of interest (AOI) in the United States.  
Now supports loading a pre-trained flood prediction model via a `.joblib` file.

---

## Features

- Loads flood prediction data exported from Google Earth Engine (GEE) as CSV with GeoJSON geometries.
- Optionally loads a pre-trained flood prediction model (`.joblib`) for local prediction.
- Converts CSV to a GeoDataFrame with robust geometry parsing.
- Fetches FEMA NFHL flood zone polygons for your AOI via the ArcGIS REST API.
- Identifies model-predicted flooded areas that fall outside FEMA's mapped SFHA zones.
- Visualizes all results on a satellite basemap, highlighting areas of interest.
- Saves outputs (GeoJSON, PNG, PDF, and JSON summary report) to an `outputs/` directory.

---

## Model Training and Results

### Model Overview

The flood prediction model used in this toolkit is trained using satellite imagery and historical flood data. The workflow is designed to be flexible and can be adapted to any area of interest (AOI) in the United States.

- **Input Data:**  
  The model leverages multi-temporal satellite imagery (e.g., Sentinel-1 SAR, Sentinel-2 optical) and, where available, historical flood event records.
- **Features:**  
  Features may include spectral indices (NDWI, MNDWI), backscatter values, topography, land cover, and temporal change metrics.
- **Labels:**  
  Training labels are derived from known flood extents, such as those mapped by government agencies or from high-confidence remote sensing flood maps.
- **Model Type:**  
  A supervised machine learning classifier (e.g., Random Forest, Gradient Boosting, or similar) is trained to distinguish flooded from non-flooded areas based on the extracted features.

### Using a Pre-trained Model (`.joblib`)

- You can use a pre-trained model saved as a `.joblib` file (e.g., `flood_model.joblib`) for local flood prediction.
- The script will load this model and apply it to your input features if specified in the configuration.
- Example:
  ```python
  from joblib import load
  model = load('flood_model.joblib')
  predictions = model.predict(X)  # X is your feature matrix
  ```

### Prediction and Export

- The trained model is applied to new satellite imagery or feature data to generate flood predictions for the AOI.
- Results are exported from Google Earth Engine as a CSV file, or generated locally using the `.joblib` model, with each row representing a spatial feature (polygon or point) and a `flood_predicted` value (1 for flooded, 0 for not flooded).

### Results Interpretation

- The toolkit compares model-predicted flooded areas to FEMA's mapped Special Flood Hazard Areas (SFHA).
- Areas predicted as flooded by the model but not mapped as SFHA by FEMA are highlighted, helping to identify potential gaps in official flood risk mapping.
- All results are visualized on a satellite basemap for easy interpretation.

**Note:**  
Model performance (accuracy, recall, etc.) will vary depending on the quality and quantity of training data, the AOI, and the features used. Users are encouraged to validate predictions with local knowledge or additional data sources where possible.

---

## Requirements

- Python 3.8+
- [pandas](https://pandas.pydata.org/)
- [geopandas](https://geopandas.org/)
- [shapely](https://shapely.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [contextily](https://contextily.readthedocs.io/)
- [requests](https://requests.readthedocs.io/)
- [joblib](https://joblib.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [dataclasses](https://docs.python.org/3/library/dataclasses.html) (Python 3.7+)
- [urllib3](https://urllib3.readthedocs.io/)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Quickstart

Get up and running in 3 simple steps:

```python
from flood_analysis import FloodAnalyzer, FloodAnalysisConfig

# 1. Configure your analysis (add model_path if using a .joblib model)
config = FloodAnalysisConfig(
    csv_file_path='your_flood_predictions.csv',  # Path to your GEE export or feature CSV
    model_path='flood_model.joblib',             # Path to your pre-trained model (optional)
    output_dir='results'                         # Where to save outputs
)

# 2. Initialize and run analysis
analyzer = FloodAnalyzer(config)
report = analyzer.run_analysis()

# 3. View results
print(f"Analysis complete! Found {report['flood_predictions']} flood predictions")
print(f"Areas outside FEMA zones: {report['areas_outside_fema']}")
```

**That's it!** Your analysis will run automatically, create visualizations, and save all outputs to the specified directory.

---

## Usage

1. **Prepare your CSV**  
   Export your flood prediction data from GEE, or generate it locally using your `.joblib` model. The CSV must include:
   - `.geo` column (GeoJSON geometry as string)
   - `flood_predicted` column (1 for flooded, 0 for not flooded)  
   *or*  
   - Feature columns required by your model, if you want to generate predictions locally.

2. **Place your CSV and model file**  
   Put your CSV and `.joblib` model file in the project root or specify their paths in the config.

3. **Run the analysis**
   ```bash
   python flood_analysis.py
   ```
   (Replace `flood_analysis.py` with your actual script name.)

4. **Outputs**  
   - `outputs/flood_predictions.geojson`: All model predictions as GeoJSON
   - `outputs/outside_fema_predictions.geojson`: Model-predicted flooded areas outside FEMA SFHA
   - `outputs/flood_analysis_map.png` and `.pdf`: Visualization
   - `outputs/analysis_report.json`: Summary statistics

---

## Configuration

Edit the configuration at the top of the script or in the `FloodAnalysisConfig` dataclass:
- `csv_file_path`: Path to your CSV file
- `model_path`: Path to your `.joblib` model file (optional)
- `initial_crs`: CRS of your data (default: `EPSG:4326`)
- `fema_sfha_layer_id`: FEMA MapServer layer ID for SFHA (default: 27)
- `sfha_zones`: List of FEMA flood zone codes considered as SFHA

### Advanced Configuration Example

```python
config = FloodAnalysisConfig(
    csv_file_path='data/miami_flood_predictions.csv',
    model_path='models/miami_flood_model.joblib',
    initial_crs='EPSG:4326',
    web_mercator_crs='EPSG:3857',
    fema_sfha_layer_id=27,
    sfha_zones=['A', 'AE', 'AH', 'AO', 'AR', 'A99', 'V', 'VE', 'VO'],
    output_dir='miami_analysis_results'
)
```

---

## Troubleshooting

- **No FEMA features found in AOI**  
  Your area of interest may not overlap with any FEMA-mapped flood zones. Check your bounding box and ensure your AOI is covered by FEMA data.

- **Geometry parsing errors**  
  Ensure your `.geo` column contains valid GeoJSON strings.

- **Basemap not displaying**  
  Requires internet connection and the `contextily` package.

- **Model loading errors**  
  Ensure your `.joblib` file is present and compatible with your feature columns.

---

## Example

![Flood Analysis Map Example](outputs/flood_analysis_map.png)

---

## License

MIT License

---

## Acknowledgments

- FEMA National Flood Hazard Layer (NFHL)
- Google Earth Engine
- Open source Python geospatial community

---

**For questions or contributions, please open an issue or pull request!**

