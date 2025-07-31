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
    csv_file_path: str = '/content/FloodPredictions_balt_AOI.csv'
    initial_crs: str = 'EPSG:4326'
    web_mercator_crs: str = 'EPSG:3857'
    
    # Multiple FEMA service options for fallback
    fema_services: List[Dict[str, Any]] = None
    sfha_zones: List[str] = None
    output_dir: str = 'outputs'
    
    def __post_init__(self):
        if self.sfha_zones is None:
            self.sfha_zones = ['A', 'AE', 'AH', 'AO', 'AR', 'A99', 'V', 'VE', 'VO']
        
        if self.fema_services is None:
            self.fema_services = [
                {
                    'name': 'FEMA_FeatureServer',
                    'url': 'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer',
                    'layer_id': 0,
                    'zone_field': 'FLD_ZONE'
                },
                {
                    'name': 'FEMA_MapServer_Legacy',
                    'url': 'https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer',
                    'layer_id': 27,
                    'zone_field': 'FLD_ZONE'
                },
                {
                    'name': 'FEMA_MapServer_Alt',
                    'url': 'https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer',
                    'layer_id': 28,
                    'zone_field': 'FLD_ZONE'
                }
            ]
        
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
        geojson_dict = json.loads(geo_string)
        return shape(geojson_dict)
    except (json.JSONDecodeError, TypeError):
        try:
            import ast
            geojson_dict = ast.literal_eval(geo_string)
            return shape(geojson_dict)
        except (ValueError, SyntaxError):
            logger.debug(f"Failed to parse geometry: {str(geo_string)[:75]}...")
            return None

class FloodAnalyzer:
    def __init__(self, config: FloodAnalysisConfig):
        self.config = config
    
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load CSV data with validation for required columns."""
        logger.info(f"Attempting to load data from: {self.config.csv_file_path}")
        try:
            df = pd.read_csv(self.config.csv_file_path)
            if not validate_dataframe(df, ['.geo', 'flood_predicted']):
                raise ValueError("Input CSV DataFrame structure is invalid.")
            logger.info(f"Successfully loaded {len(df)} rows from CSV.")
            return df
        except FileNotFoundError:
            logger.error(f"Error: The file '{self.config.csv_file_path}' was not found.")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading the CSV: {e}")
            raise
    
    def create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert pandas DataFrame to GeoDataFrame with robust geometry parsing."""
        logger.info("Converting DataFrame to GeoDataFrame...")
        
        geometries = []
        parse_stats = {'success': 0, 'failed': 0}
        
        for idx, row in df.iterrows():
            geom = safe_geometry_parsing(row['.geo'])
            geometries.append(geom)
            
            if geom is None:
                parse_stats['failed'] += 1
            else:
                parse_stats['success'] += 1
        
        logger.info(f"Geometry parsing - Success: {parse_stats['success']}, Failed: {parse_stats['failed']}")
        
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.config.initial_crs)
        original_len = len(gdf)
        gdf = gdf.dropna(subset=['geometry'])
        dropped = original_len - len(gdf)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows due to invalid geometries.")
        
        if gdf.empty:
            raise ValueError("GeoDataFrame is empty after parsing geometries.")
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} valid features.")
        logger.info(f"Bounding Box: {gdf.total_bounds.tolist()}")
        return gdf
    
    def test_fema_service(self, service_config: Dict[str, Any], bounds: tuple) -> Optional[Dict[str, Any]]:
        """Test a single FEMA service configuration"""
        logger.info(f"Testing FEMA service: {service_config['name']}")
        
        session = requests.Session()
        retry_strategy = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        minx, miny, maxx, maxy = bounds
        
        # First, test if the service is accessible
        try:
            service_url = f"{service_config['url']}/{service_config['layer_id']}"
            test_response = session.get(f"{service_url}?f=json", timeout=30)
            test_response.raise_for_status()
            service_info = test_response.json()
            
            logger.info(f"Service {service_config['name']} is accessible")
            logger.debug(f"Service info: {service_info.get('name', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"Service {service_config['name']} is not accessible: {e}")
            return None
        
        # Now try to query data
        query_params = {
            'where': '1=1',
            'outFields': service_config['zone_field'],
            'geometry': f'{minx},{miny},{maxx},{maxy}',
            'geometryType': 'esriGeometryEnvelope',
            'inSr': self.config.initial_crs.split(':')[-1],
            'spatialRel': 'esriSpatialRelIntersects',
            'outSr': self.config.initial_crs.split(':')[-1],
            'f': 'geojson',
            'returnGeometry': 'true',
            'returnTrueCurves': 'false'
        }
        
        try:
            query_url = f"{service_url}/query"
            logger.info(f"Querying: {query_url}")
            response = session.get(query_url, params=query_params, timeout=60)
            response.raise_for_status()
            
            # Debug: Log response content type and first part of response
            logger.debug(f"Response content type: {response.headers.get('content-type')}")
            logger.debug(f"Response text (first 200 chars): {response.text[:200]}")
            
            data = response.json()
            
            # Check for different response formats
            if 'features' in data:
                features = data['features']
                logger.info(f"Service {service_config['name']} returned {len(features)} features")
                return {'service': service_config, 'data': data}
            elif 'error' in data:
                logger.warning(f"Service {service_config['name']} returned error: {data['error']}")
                return None
            else:
                logger.warning(f"Service {service_config['name']} returned unexpected format. Keys: {list(data.keys())}")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {service_config['name']}: {e}")
            logger.debug(f"Raw response: {response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error querying {service_config['name']}: {e}")
            return None
    
    def fetch_fema_data(self, bounds: tuple) -> Optional[gpd.GeoDataFrame]:
        """Fetch FEMA data with multiple service fallback"""
        logger.info("Attempting to fetch FEMA NFHL data from multiple services...")
        
        for service_config in self.config.fema_services:
            result = self.test_fema_service(service_config, bounds)
            
            if result is not None:
                try:
                    data = result['data']
                    service_config = result['service']
                    
                    if len(data['features']) == 0:
                        logger.warning(f"No features found in {service_config['name']} for your AOI")
                        continue
                    
                    fema_gdf = gpd.GeoDataFrame.from_features(data['features'], crs=self.config.initial_crs)
                    logger.info(f"Successfully loaded {len(fema_gdf)} features from {service_config['name']}")
                    
                    # Check available fields
                    logger.info(f"Available fields: {fema_gdf.columns.tolist()}")
                    
                    # Try to filter for SFHA zones
                    zone_field = service_config['zone_field']
                    if zone_field in fema_gdf.columns:
                        # Log unique zone values for debugging
                        unique_zones = fema_gdf[zone_field].unique()
                        logger.info(f"Unique flood zones found: {unique_zones}")
                        
                        sfha_gdf = fema_gdf[fema_gdf[zone_field].isin(self.config.sfha_zones)].copy()
                        if not sfha_gdf.empty:
                            dissolved_sfha = sfha_gdf.dissolve()
                            logger.info(f"Successfully filtered to {len(sfha_gdf)} SFHA features from {service_config['name']}")
                            return dissolved_sfha
                        else:
                            logger.warning(f"No SFHA zones found in {service_config['name']} after filtering")
                            # Return all features if no SFHA zones match
                            logger.info("Returning all flood zone features instead of filtering for SFHA")
                            return fema_gdf.dissolve()
                    else:
                        logger.warning(f"Zone field '{zone_field}' not found. Using all features.")
                        return fema_gdf.dissolve()
                        
                except Exception as e:
                    logger.error(f"Error processing data from {service_config['name']}: {e}")
                    continue
        
        logger.error("All FEMA services failed or returned no data")
        return None
    
    def identify_outside_fema(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Identify flood predictions outside FEMA zones"""
        logger.info("Identifying flood predictions outside FEMA zones...")
        
        if fema_gdf is None or fema_gdf.empty:
            logger.warning("No FEMA data available - all flood predictions considered 'outside FEMA'")
            return gdf[gdf['flood_predicted'] == 1].copy()
        
        if gdf.crs != fema_gdf.crs:
            logger.info(f"Reprojecting for spatial analysis: {gdf.crs} -> {fema_gdf.crs}")
            gdf_for_sjoin = gdf.to_crs(fema_gdf.crs)
        else:
            gdf_for_sjoin = gdf.copy()
        
        sjoin_result = gpd.sjoin(gdf_for_sjoin, fema_gdf, how="left", predicate="intersects")
        outside_fema_predictions = sjoin_result[sjoin_result['index_right'].isna()].copy()
        flooded_outside_fema = outside_fema_predictions[outside_fema_predictions['flood_predicted'] == 1]
        
        logger.info(f"Total predictions: {len(gdf)}")
        logger.info(f"Outside FEMA zones: {len(outside_fema_predictions)}")
        logger.info(f"Flooded outside FEMA: {len(flooded_outside_fema)}")
        
        return flooded_outside_fema
    
    def create_enhanced_plot(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame],
                            outside_fema: gpd.GeoDataFrame) -> plt.Figure:
        """Create enhanced visualization"""
        logger.info("Creating visualization...")
        
        gdf_proj = gdf.to_crs(self.config.web_mercator_crs)
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot study area
        gdf_proj.plot(ax=ax, color='lightgray', edgecolor='gray', alpha=0.05,
                      linewidth=0.5, label='Study Area')
        
        # Plot FEMA zones
        if fema_gdf is not None and not fema_gdf.empty:
            fema_proj = fema_gdf.to_crs(self.config.web_mercator_crs)
            fema_proj.plot(ax=ax, color='cyan', edgecolor='darkblue', alpha=0.6,
                           linewidth=1.2, label='FEMA Flood Zones')
            logger.info("FEMA zones included in plot")
        else:
            logger.warning("No FEMA zones to plot")
        
        # Plot flood predictions
        flooded_all = gdf_proj[gdf_proj['flood_predicted'] == 1]
        if not flooded_all.empty:
            flooded_all.plot(ax=ax, color='red', alpha=0.8, markersize=8,
                            label=f'Model Predicted Flood ({len(flooded_all)})')
        
        # Highlight outside FEMA
        if not outside_fema.empty:
            outside_proj = outside_fema.to_crs(self.config.web_mercator_crs)
            outside_proj.plot(ax=ax, color='yellow', marker='X', markersize=30,
                             edgecolor='black', linewidth=2,
                             label=f'Outside FEMA Zones ({len(outside_proj)})')
        
        # Add basemap
        try:
            ctx.add_basemap(ax, crs=self.config.web_mercator_crs,
                           source=ctx.providers.Esri.WorldImagery, alpha=0.85)
            logger.info("Basemap added successfully")
        except Exception as e:
            logger.warning(f"Could not add basemap: {e}")
        
        # Styling
        ax.set_title('Flood Risk Analysis: Model Predictions vs FEMA Data\nBaltimore AOI',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Eastings (m)', fontsize=12)
        ax.set_ylabel('Northings (m)', fontsize=12)
        
        # Set extent
        if not gdf_proj.empty:
            minx, miny, maxx, maxy = gdf_proj.total_bounds
            buffer_x = (maxx - minx) * 0.15
            buffer_y = (maxy - miny) * 0.15
            ax.set_xlim(minx - buffer_x, maxx + buffer_x)
            ax.set_ylim(miny - buffer_y, maxy + buffer_y)
        
        # Legend
        legend = ax.legend(loc='upper left', frameon=True, facecolor='white',
                          edgecolor='black', shadow=True, fontsize=10)
        legend.get_frame().set_alpha(0.9)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def generate_summary_report(self, gdf: gpd.GeoDataFrame, fema_gdf: Optional[gpd.GeoDataFrame],
                               outside_fema: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        total_predictions = len(gdf)
        flood_predictions_count = len(gdf[gdf['flood_predicted'] == 1])
        
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_areas_analyzed': total_predictions,
            'model_flood_predictions_count': flood_predictions_count,
            'model_flood_percentage_of_total': round((flood_predictions_count / total_predictions * 100), 2) if total_predictions > 0 else 0,
            'fema_data_processed': fema_gdf is not None and not fema_gdf.empty,
            'fema_zones_count': len(fema_gdf) if fema_gdf is not None else 0,
            'model_flooded_outside_fema_count': len(outside_fema),
            'area_of_interest_bbox': gdf.total_bounds.tolist(),
            'aoi_crs': str(gdf.crs)
        }
        
        # Save report
        report_path = Path(self.config.output_dir) / 'analysis_report.json'
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
        
        return report
    
    def save_outputs(self, gdf: gpd.GeoDataFrame, outside_fema: gpd.GeoDataFrame,
                    fig: plt.Figure) -> None:
        """Save all outputs"""
        output_dir = Path(self.config.output_dir)
        
        try:
            gdf.to_file(output_dir / 'all_flood_predictions.geojson', driver='GeoJSON')
            logger.info("Saved all predictions to GeoJSON")
            
            if not outside_fema.empty:
                outside_fema.to_file(output_dir / 'flooded_outside_fema.geojson', driver='GeoJSON')
                logger.info("Saved outside FEMA predictions to GeoJSON")
            
            fig.savefig(output_dir / 'flood_analysis_map.png', dpi=300, bbox_inches='tight')
            fig.savefig(output_dir / 'flood_analysis_map.pdf', bbox_inches='tight')
            logger.info("Saved visualization plots")
            
        except Exception as e:
            logger.error(f"Error saving outputs: {e}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis pipeline"""
        logger.info("Starting Flood Risk Analysis Pipeline")
        try:
            # Load and process data
            df = self.load_and_validate_data()
            gdf = self.create_geodataframe(df)
            
            # Fetch FEMA data with fallback services
            fema_gdf = self.fetch_fema_data(gdf.total_bounds)
            
            # Analyze areas outside FEMA zones
            outside_fema = self.identify_outside_fema(gdf, fema_gdf)
            
            # Create visualization
            fig = self.create_enhanced_plot(gdf, fema_gdf, outside_fema)
            
            # Generate reports and save outputs
            report = self.generate_summary_report(gdf, fema_gdf, outside_fema)
            self.save_outputs(gdf, outside_fema, fig)
            
            plt.show()
            
            logger.info("Analysis completed successfully!")
            logger.info(f"Summary: {report['model_flood_predictions_count']}/{report['total_areas_analyzed']} areas predicted to flood")
            logger.info(f"Areas outside FEMA: {report['model_flooded_outside_fema_count']}")
            
            return report
            
        except Exception as e:
            logger.critical(f"Analysis failed: {e}", exc_info=True)
            raise

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("      Flood Risk Analysis Application")
    print("="*60 + "\n")
    
    try:
        config = FloodAnalysisConfig()
        analyzer = FloodAnalyzer(config)
        report = analyzer.run_analysis()
        
        print("\n" + "="*60)
        print("                 ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Date:             {report['analysis_date']}")
        print(f"Total Areas Analyzed:      {report['total_areas_analyzed']}")
        print(f"Model Flood Predictions:   {report['model_flood_predictions_count']} ({report['model_flood_percentage_of_total']}%)")
        print(f"FEMA Data Available:       {'Yes' if report['fema_data_processed'] else 'No'}")
        print(f"FEMA Zones Processed:      {report['fema_zones_count']}")
        print(f"Flooded Outside FEMA:      {report['model_flooded_outside_fema_count']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
