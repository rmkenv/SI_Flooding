import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import json
import sys

# --- Configuration ---
CSV_FILE_PATH = 'FloodPredictions_NEW_AOI.csv'
OUTPUT_CRS = 'EPSG:4326'

# --- 1. Load the CSV Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("CSV loaded successfully.")
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print(f"Total rows: {len(df)}")
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found. Please check the path.")
    sys.exit(1)

# --- 2. Convert to GeoDataFrame ---
print("Converting geometries...")
geometries = []
valid_count = 0
invalid_count = 0

for index, row in df.iterrows():
    geo_str = row.get('.geo', None)
    
    # Check if it's a valid string that looks like GeoJSON
    if isinstance(geo_str, str) and geo_str.strip().startswith('{'):
        try:
            # Use json.loads instead of ast.literal_eval for GeoJSON
            geojson_dict = json.loads(geo_str)
            geom = shape(geojson_dict)
            geometries.append(geom)
            valid_count += 1
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Could not parse GeoJSON for row {index}. Error: {e}")
            geometries.append(None)
            invalid_count += 1
    else:
        # Not a valid GeoJSON string
        geometries.append(None)
        invalid_count += 1

print(f"Valid geometries: {valid_count}")
print(f"Invalid geometries: {invalid_count}")

# Create the GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=OUTPUT_CRS)

# Remove rows where geometry is None
gdf_clean = gdf.dropna(subset=['geometry'])

if gdf_clean.empty:
    print("Error: No valid geometries found in the CSV to plot. Exiting.")
    sys.exit(1)

print(f"\nGeoDataFrame created successfully with {len(gdf_clean)} valid geometries.")
print("First 5 rows of the GeoDataFrame:")
print(gdf_clean.head())

# --- 3. Plot the Flood Predictions ---
print("\nPlotting flood predictions...")

fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Plot all areas (context) in light gray
gdf_clean.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.7, linewidth=0.5)

# Plot predicted flooded areas in blue, if present
if 'flood_predicted' in gdf_clean.columns:
    flooded_areas = gdf_clean[gdf_clean['flood_predicted'] == 1]
    if not flooded_areas.empty:
        flooded_areas.plot(ax=ax, color='blue', alpha=0.8, edgecolor='darkblue', linewidth=0.5)
        print(f"Found {len(flooded_areas)} areas predicted as flooded.")
    else:
        print("No areas predicted as flooded (flood_predicted == 1) to plot separately.")
        
    # Also plot by flood probability if you want a gradient
    if 'flood_probability' in gdf_clean.columns:
        # Uncomment the next 3 lines if you want to color by probability instead
        # gdf_clean.plot(column='flood_probability', cmap='Blues', linewidth=0.5, 
        #                ax=ax, edgecolor='0.8', legend=True,
        #                legend_kwds={'label': "Flood Probability", 'orientation': "horizontal"})
        pass
else:
    print("Warning: 'flood_predicted' column not found in the data.")

# Set title and labels
ax.set_title('Predicted Flood Areas in Norfolk AOI', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.3)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightgray', edgecolor='black', label='All AOI areas'),
    Patch(facecolor='blue', edgecolor='darkblue', label='Predicted Flooded Areas')
]
ax.legend(handles=legend_elements, loc='upper right')

# Adjust layout and show
plt.tight_layout()
plt.show()

print(f"\nPlotting complete! Displayed {len(gdf_clean)} total areas.")
if 'flood_predicted' in gdf_clean.columns:
    flooded_count = len(gdf_clean[gdf_clean['flood_predicted'] == 1])
    print(f"Of these, {flooded_count} are predicted as flooded.")
