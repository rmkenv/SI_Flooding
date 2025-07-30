Okay, you want to switch the FEMA data source from the previous MapServer to a **FeatureServer**. This is a good move, as FeatureServers are often more optimized for querying features directly.

The new URL is `https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer`.

**Important Note:** When switching to a new ArcGIS service (especially from a MapServer to a FeatureServer, or a different FeatureServer entirely), the **layer IDs are almost certainly different**. I've inspected this FeatureServer, and the most relevant layer for flood hazard areas appears to be `Layer 0: Flood_Hazard_Areas_with_LOMR_and_LOMA`. I will update the `FEMA_SFHA_LAYER_ID` to `0` for this new service.

I'll provide the complete code with this change incorporated into the `FloodAnalysisConfig`.

```python
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
import json
import requests
import os
import contextily as ctx
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FloodAnalysisConfig:
    """Configuration class for flood analysis parameters"""
    csv_file_path: str = '/content/FloodPredictions_NEW_AOI.csv' # Consider making this flexible for deployment
    initial_crs: str = 'EPSG:4326' # WGS84 for lat/lon data
    web_mercator_crs: str = 'EPSG:3857' # Standard for web maps/contextily
    
    # --- UPDATED FEMA SERVICE URL AND LAYER ID ---
    fema_nfhl_url: str = 'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer'
    fema_sfha_layer_id: int = 0 # UPDATED: Layer ID for 'Flood_Hazard_Areas_with_LOMR_and_LOMA' in this FeatureServer
    # ---------------------------------------------

    sfha_zones: List[str] = None # List of FEMA flood zone designators for SFHAs
    output_dir: str = 'outputs' # Directory for saving results
    
    def __post_init__(self):
        # Initialize SFHA zones if not provided
        if self.sfha_zones is None:
            self.sfha_zones = ['A', 'AE', 'AH', 'AO', 'AR', 'A99', 'V', 'VE', 'VO']
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{self.output_dir}' ensured.")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns and data"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"DataFrame is missing required columns: {missing_cols}")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty after column check.")
        return False
    
    return True

def safe_geometry_parsing(geo_string: str) -> Optional[Any]:
    """Safely parse geometry string with multiple fallback methods"""
    if pd.isna(geo_string) or not geo_string:
        return None
    
    try:
        # Try JSON parsing first (most common for GEE exports)
        geojson_dict = json.loads(geo_string)
        return shape(geojson_dict)
    except (json.JSONDecodeError, TypeError):
        try:
            # Fallback to ast.literal_eval for non-strict JSON-like strings
            import ast # Import here as it's a fallback and not always needed
            geojson_dict = ast.literal_eval(geo_string)
            return shape(geojson_dict)
        except (ValueError, SyntaxError):
            logger.debug(f"Failed to parse geometry with both JSON and AST: {str(geo_string)[:75]}...")
            return None

class FloodAnalyzer:
    def __init__(self, config: FloodAnalysisConfig):
        self.config = config
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load CSV data with validation for required columns."""
        logger.info(f"Attempting to load data from: {self.config.csv_file_path}")
        try:
            df = pd.read_csv(self.config.csv_file_path)
            # Validate essential columns exist for GeoDataFrame conversion and analysis
            if not validate_dataframe(df, ['.geo', 'flood_predicted']):
                raise ValueError("Input CSV DataFrame structure is invalid.")
            logger.info(f"Successfully loaded {len(df)} rows from CSV.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: The file '{self.config.csv_file_path}' was not found. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the CSV: {e}")
            raise
    
    def create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert pandas DataFrame to GeoDataFrame with robust geometry parsing."""
        logger.info("Converting DataFrame to GeoDataFrame...")
        
        geometries = []
        parse_stats = {'success': 0, 'failed': 0} # No 'null' needed, failed covers it
        
        for idx, row in df.iterrows():
            geom = safe_geometry_parsing(row['.geo'])
            geometries.append(geom)
            
            if geom is None:
                parse_stats['failed'] += 1
            else:
                parse_stats['success'] += 1
        
        logger.info(f"Geometry parsing summary - Success: {parse_stats['success']}, Failed: {parse_stats['failed']}.")
        
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.config.initial_crs)
        original_len = len(gdf)
        gdf = gdf.dropna(subset=['geometry']) # Drop rows where geometry parsing failed
        dropped = original_len - len(gdf)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows due to invalid or unparsable geometries.")
        
        if gdf.empty:
            raise ValueError("GeoDataFrame is empty after parsing geometries. No valid geospatial data to proceed.")
            
        logger.info(f"Created GeoDataFrame with {len(gdf)} valid features.")
        logger.info(f"GeoDataFrame CRS: {gdf.crs}")
        logger.info(f"GeoDataFrame Bounding Box (Initial CRS): {gdf.total_bounds.tolist()}")
        return gdf
    
    def fetch_fema_data(self, bounds: tuple) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch FEMA NFHL data from ArcGIS REST service with retry logic.
        Filters for SFHA zones and dissolves polygons.
        """
        logger.info("Attempting to fetch FEMA NFHL data...")
        
        # Setup retry strategy for robust requests
        session = requests.Session()
        retry_strategy = Retry(
            total=3, # Total retries
            backoff_factor=1, # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504], # Status codes to retry on
            allowed_methods=["HEAD", "GET", "OPTIONS"] # Methods to retry
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        minx, miny, maxx, maxy = bounds
        query_params: Dict[str, Any] = {
            'where': '1=1', # Select all features in the geometry extent
            'outFields': 'FLD_ZONE', # Request only the flood zone field
            'geometry': f'{minx},{miny},{maxx},{maxy}', # Bounding box from your AOI
            'geometryType': 'esriGeometryEnvelope',
            'inSr': self.config.initial_crs.split(':')[-1], # Input spatial reference (e.g., '4326')
            'spatialRel': 'esriSpatialRelIntersects', # Relationship: features intersecting AOI
            'outSr': self.config.initial_crs.split(':')[-1], # Output spatial reference
            'f': 'geojson', # Request GeoJSON format
            'returnGeometry': 'true',
            'returnTrueCurves': 'false' # Often necessary for older services
        }
        
        try:
            url = f"{self.config.fema_nfhl_url}/{self.config.fema_sfha_layer_id}/query"
            logger.info(f"Querying FEMA URL: {url} with bounds {bounds}")
            response = session.get(url, params=query_params, timeout=60) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            data = response.json()
            
            if 'features' in data:
                num_features_returned = len(data['features'])
                logger.info(f"FEMA API returned {num_features_returned} features.")
                if num_features_returned == 0:
                    logger.warning("No FEMA flood zones found in your AOI from this service. This might mean the area is not mapped, or the AOI is too small.")
                    return None
            else:
                logger.warning("FEMA API response did not contain a 'features' key or was malformed.")
                logger.debug(f"Full FEMA response (partial): {str(data)[:500]}") # Log partial response for debug
                return None

            fema_gdf = gpd.GeoDataFrame.from_features(data['features'], crs=self.config.initial_crs)
            logger.info(f"Successfully loaded {len(fema_gdf)} raw FEMA features into GeoDataFrame.")
            
            # Filter for Special Flood Hazard Areas (SFHAs)
            # The field name 'FLD_ZONE' is common, but it's crucial for this new service to have it.
            if 'FLD_ZONE' in fema_gdf.columns:
                sfha_gdf = fema_gdf[fema_gdf['FLD_ZONE'].isin(self.config.sfha_zones)].copy()
                if not sfha_gdf.empty:
                    dissolved_sfha = sfha_gdf.dissolve()
                    logger.info(f"Filtered to {len(sfha_gdf)} SFHA features. Dissolved into {len(dissolved_sfha)} polygons.")
                    return dissolved_sfha
                else:
                    logger.warning(f"No SFHA zones ({self.config.sfha_zones}) found in the filtered FEMA data for your AOI. Check 'FLD_ZONE' values or if SFHAs are present in this specific service.")
                    return None # Return None if no SFHA found after filtering
            else:
                logger.warning(f"Column 'FLD_ZONE' not found in FEMA data from Layer {self.config.fema_sfha_layer_id} of this new FeatureServer. Cannot filter for SFHAs. Available columns: {fema_gdf.columns.tolist()}")
                # If FLD_ZONE is missing, we can't properly filter for SFHA.
                return None
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching FEMA data: {e}. Status: {response.status_code}. Response: {response.text[:200]}...")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching FEMA data: {e}. Check internet connection or FEMA service availability.")
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout fetching FEMA data: {e}. Service might be slow or data too large.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from FEMA response: {e}. Response might not be valid JSON. Content: {response.text[:500]}...")
            return None
        except Exception as e:
            logger.critical(f"An unexpected error occurred while fetching FEMA data: {e}", exc_info=True)
            return None
    
    def identify_outside_fema(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """
        Identifies flood predictions that fall outside FEMA Special Flood Hazard Areas.
        If no FEMA data is provided, all flood predictions are considered 'outside FEMA'.
        """
        logger.info("Identifying flood predictions outside FEMA SFHA zones...")
        
        # If no FEMA data, all predicted floods are technically "outside FEMA zones"
        if fema_gdf is None or fema_gdf.empty:
            logger.warning("No valid FEMA data available. All model-predicted floods will be considered 'outside FEMA'.")
            # Ensure we return a GeoDataFrame with the correct CRS
            return gdf[gdf['flood_predicted'] == 1].copy()
        
        # Ensure GeoDataFrames are in the same CRS for accurate spatial operations
        if gdf.crs != fema_gdf.crs:
            logger.info(f"Reprojecting input GeoDataFrame from {gdf.crs} to match FEMA CRS {fema_gdf.crs} for spatial analysis.")
            gdf_for_sjoin = gdf.to_crs(fema_gdf.crs)
        else:
            gdf_for_sjoin = gdf.copy() # Work on a copy to avoid modifying original gdf
        
        # Perform a left spatial join:
        # If a prediction intersects a FEMA polygon, 'index_right' will have a value.
        # If it does NOT intersect (i.e., is outside), 'index_right' will be NaN.
        sjoin_result = gpd.sjoin(gdf_for_sjoin, fema_gdf, how="left", predicate="intersects")
        
        # Select rows where 'index_right' is NaN (meaning no intersection with FEMA)
        outside_fema_predictions = sjoin_result[sjoin_result['index_right'].isna()].copy()
        
        # From these, select only the ones where flood_predicted is 1
        flooded_outside_fema = outside_fema_predictions[outside_fema_predictions['flood_predicted'] == 1]
        
        logger.info(f"Total model predictions analyzed: {len(gdf)}")
        logger.info(f"Model predictions spatially outside any FEMA SFHA (regardless of flood prediction): {len(outside_fema_predictions)}")
        logger.info(f"Model predicted flooded areas *outside* FEMA SFHA: {len(flooded_outside_fema)}")
        
        return flooded_outside_fema
    
    def create_enhanced_plot(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame], 
                            outside_fema: gpd.GeoDataFrame) -> plt.Figure:
        """
        Create an enhanced visualization of flood predictions and FEMA zones
        on a satellite basemap.
        """
        logger.info("Creating visualization...")
        
        # Reproject all data to Web Mercator for consistent plotting with contextily
        gdf_proj = gdf.to_crs(self.config.web_mercator_crs)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot study area/all prediction points (as a very faint background)
        # Using a very low alpha to not obscure the basemap or other layers
        gdf_proj.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.05,
                      linewidth=0.5, label='Total Prediction Areas')
        
        # Plot FEMA SFHA zones if available
        if fema_gdf is not None and not fema_gdf.empty:
            fema_proj = fema_gdf.to_crs(self.config.web_mercator_crs)
            # Using 'cyan' and a slightly higher alpha for better visibility against satellite
            fema_proj.plot(ax=ax, color='cyan', edgecolor='darkblue', alpha=0.6, 
                           linewidth=1.2, label='FEMA SFHA')
            logger.info("FEMA SFHA layer included in plot.")
        else:
            logger.warning("FEMA SFHA layer not included in plot (data is empty or not available).")
        
        # Plot all model-predicted flooded areas
        flooded_all_predictions = gdf_proj[gdf_proj['flood_predicted'] == 1]
        if not flooded_all_predictions.empty:
            # Using 'red' for general model floods, slightly offset if desired for clarity
            flooded_all_predictions.plot(ax=ax, color='red', alpha=0.8, markersize=8, 
                                         label=f'Model Predicted Flood (All: {len(flooded_all_predictions)})')
            logger.info(f"Plotted {len(flooded_all_predictions)} 'Model Predicted Flood (All)' points.")
        else:
            logger.info("No 'Model Predicted Flood (All)' data to plot.")
        
        # Highlight areas predicted to flood *outside* FEMA SFHA
        if not outside_fema.empty:
            outside_fema_proj = outside_fema.to_crs(self.config.web_mercator_crs)
            # Using 'yellow X' for strong emphasis on these critical points
            outside_fema_proj.plot(ax=ax, color='yellow', marker='X', markersize=30, 
                                   edgecolor='black', linewidth=2, 
                                   label=f'Predicted Flood Outside FEMA ({len(outside_fema_proj)})')
            logger.info(f"Plotted {len(outside_fema_proj)} 'Predicted Flood Outside FEMA' points.")
        else:
            logger.info("No 'Model Flooded, Outside FEMA SFHA' data to highlight.")
        
        # Add satellite basemap with error handling
        try:
            # Alpha for basemap to allow underlying data to show through if desired
            ctx.add_basemap(ax, crs=self.config.web_mercator_crs, 
                            source=ctx.providers.Esri.WorldImagery, alpha=0.85)
            logger.info("Satellite basemap added successfully.")
        except Exception as e:
            logger.warning(f"Could not add basemap. Check internet connection or contextily setup: {e}")
        
        # Enhanced styling for plot title and labels
        ax.set_title('Flood Risk Analysis: Model Predictions vs FEMA NFHL\nNorfolk, VA', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Eastings (m)', fontsize=12)
        ax.set_ylabel('Northings (m)', fontsize=12)
        
        # Set map extent with a small buffer around the study area
        if not gdf_proj.empty:
            minx, miny, maxx, maxy = gdf_proj.total_bounds
            # Calculate a buffer as a percentage of the extent
            buffer_x = (maxx - minx) * 0.15 # Increased buffer slightly for better context
            buffer_y = (maxy - miny) * 0.15
            ax.set_xlim(minx - buffer_x, maxx + buffer_x)
            ax.set_ylim(miny - buffer_y, maxy + buffer_y)
            logger.info(f"Plotting extent set to: X[{ax.get_xlim()}], Y[{ax.get_ylim()}].")
        else:
            logger.warning("Cannot set specific plot limits as primary GeoDataFrame is empty.")
        
        # Better legend placement and appearance
        legend = ax.legend(loc='upper left', frameon=True, facecolor='white', 
                          edgecolor='black', shadow=True, fontsize=10)
        legend.get_frame().set_alpha(0.9) # Make legend slightly transparent
        
        # Grid and layout adjustments
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def generate_summary_report(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame], 
                               outside_fema: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis summary report."""
        total_predictions = len(gdf)
        flood_predictions_count = len(gdf[gdf['flood_predicted'] == 1])
        
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_areas_analyzed': total_predictions,
            'model_flood_predictions_count': flood_predictions_count,
            'model_flood_percentage_of_total': round((flood_predictions_count / total_predictions * 100), 2) if total_predictions > 0 else 0,
            'fema_data_processed': fema_gdf is not None and not fema_gdf.empty,
            'fema_sfha_zones_count': len(fema_gdf) if fema_gdf is not None else 0,
            'model_flooded_outside_fema_count': len(outside_fema),
            'area_of_interest_bbox_initial_crs': gdf.total_bounds.tolist(),
            'aoi_crs': str(gdf.crs)
        }
        
        # Save report to JSON file
        report_path = Path(self.config.output_dir) / 'analysis_report.json'
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Analysis report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
        
        return report
    
    def save_outputs(self, gdf: gpd.GeoDataFrame, outside_fema: gpd.GeoDataFrame, 
                    fig: plt.Figure) -> None:
        """Save all generated outputs (GeoDataFrames, plots) to files."""
        output_dir = Path(self.config.output_dir)
        
        try:
            # Save GeoDataFrames as GeoJSON for easy sharing/viewing
            gdf.to_file(output_dir / 'all_flood_predictions.geojson', driver='GeoJSON')
            logger.info("Saved all flood predictions to GeoJSON.")
            
            if not outside_fema.empty:
                outside_fema.to_file(output_dir / 'model_flooded_outside_fema.geojson', driver='GeoJSON')
                logger.info("Saved model-predicted flooded areas outside FEMA to GeoJSON.")
            else:
                logger.info("No 'model_flooded_outside_fema' data to save.")
            
            # Save plots in high-resolution PNG and PDF formats
            fig.savefig(output_dir / 'flood_risk_analysis_map.png', dpi=300, bbox_inches='tight')
            fig.savefig(output_dir / 'flood_risk_analysis_map.pdf', bbox_inches='tight')
            logger.info("Saved visualization plots (PNG and PDF).")
            
        except Exception as e:
            logger.error(f"Error saving outputs: {e}", exc_info=True)
    
    def run_analysis(self) -> Dict[str, Any]:
        """Execute the complete flood analysis pipeline."""
        logger.info("\n--- Starting Flood Risk Analysis Pipeline ---")
        try:
            # Step 1 & 2: Load CSV and create GeoDataFrame
            df = self.load_and_validate_data()
            gdf = self.create_geodataframe(df)
            
            # Step 3: Fetch FEMA data
            fema_gdf = self.fetch_fema_data(gdf.total_bounds)
            
            # Step 4: Analyze areas outside FEMA zones
            # This function now correctly handles fema_gdf being None/empty
            outside_fema = self.identify_outside_fema(gdf, fema_gdf)
            
            # Step 5: Create visualization
            fig = self.create_enhanced_plot(gdf, fema_gdf, outside_fema)
            
            # Step 6: Generate reports and save outputs
            report = self.generate_summary_report(gdf, fema_gdf, outside_fema)
            self.save_outputs(gdf, outside_fema, fig)
            
            # Display the plot
            plt.show()
            
            logger.info("Flood risk analysis completed successfully!")
            logger.info(f"Summary: {report['model_flood_predictions_count']}/{report['total_areas_analyzed']} areas predicted to flood ({report['model_flood_percentage_of_total']}%).")
            logger.info(f"Areas predicted to flood *outside* FEMA SFHA: {report['model_flooded_outside_fema_count']}.")
            
            return report
            
        except Exception as e:
            logger.critical(f"Flood analysis pipeline failed: {e}", exc_info=True)
            raise # Re-raise the exception after logging for main() to catch

def main():
    """Main execution function for the flood analysis application."""
    print("\n" + "="*60)
    print("      Initiating Flood Risk Analysis Application")
    print("="*60 + "\n")
    try:
        # Initialize configuration and analyzer
        config = FloodAnalysisConfig()
        analyzer = FloodAnalyzer(config)
        
        # Run the full analysis pipeline
        report = analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("                 FINAL ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Date:             {report['analysis_date']}")
        print(f"Total Areas Analyzed:      {report['total_areas_analyzed']}")
        print(f"Model Flood Predictions:   {report['model_flood_predictions_count']} ({report['model_flood_percentage_of_total']}%)")
        print(f"FEMA Data Processed:       {'Yes' if report['fema_data_processed'] else 'No'}")
        print(f"SFHA Zones Processed:      {report['fema_sfha_zones_count']}")
        print(f"Predicted Flooded Outside FEMA: {report['model_flooded_outside_fema_count']}")
        print(f"Area of Interest BBox:     {report['area_of_interest_bbox_initial_crs']}")
        print(f"AOI CRS:                   {report['aoi_crs']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Application terminated due to an error: {e}")
        print(f"\nFATAL ERROR: {e}")
        print("Please review the log messages above for detailed error information.")

if __name__ == "__main__":
    main()

```
