"""
Flood Risk Analysis Pipeline
============================

A comprehensive tool for analyzing flood predictions using machine learning models
and comparing them with FEMA flood zone data.

Features:
- Random Forest flood prediction model training
- FEMA NFHL data integration with multiple service fallbacks
- Geospatial analysis and visualization
- Comprehensive reporting and output generation
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from joblib import dump, load
from requests.adapters import HTTPAdapter
from shapely.geometry import shape
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FloodAnalysisConfig:
    """Configuration class for flood analysis parameters."""
    
    # File paths
    csv_file_path: str = '/content/FloodPredictions_balt_AOI.csv'
    output_dir: str = 'outputs'
    
    # Coordinate reference systems
    initial_crs: str = 'EPSG:4326'  # WGS84
    web_mercator_crs: str = 'EPSG:3857'  # Web Mercator for mapping
    
    # FEMA service configurations (multiple for fallback)
    fema_services: List[Dict[str, Any]] = None
    sfha_zones: List[str] = None
    
    def __post_init__(self):
        """Initialize default values and create output directory."""
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
                }
            ]
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory '{self.output_dir}' created/verified")


class FloodModelTrainer:
    """Handles training and evaluation of flood prediction models."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.calibrated_model = None
        
    def load_and_prepare_data(self) -> tuple:
        """Load and prepare training data."""
        logger.info(f"Loading training data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        df = df.dropna()
        
        features = [
            'ndwi_post', 'mndwi_post', 'awei_post', 'VV',
            'elevation', 'slope', 'Map', 'jrc_seasonality'
        ]
        target = 'flood'
        
        # One-hot encode categorical variables
        X = pd.get_dummies(df[features], columns=['Map'])
        y = df[target]
        
        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train Random Forest model with cross-validation."""
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        logger.info(f"CV ROC AUC scores: {cv_scores}")
        logger.info(f"Mean CV ROC AUC: {np.mean(cv_scores):.3f}")
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        logger.info("Evaluating model performance...")
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Print evaluation metrics
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        
        test_auc = roc_auc_score(y_test, y_proba)
        print(f"Test ROC AUC: {test_auc:.3f}")
        
        return {'test_auc': test_auc}
    
    def calibrate_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Apply probability calibration to the model."""
        logger.info("Calibrating model probabilities...")
        
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrated_model.fit(X_train, y_train)
        
    def plot_feature_importance(self) -> None:
        """Plot and display feature importance."""
        if self.model is None:
            logger.warning("Model not trained yet")
            return
            
        importances = pd.Series(
            self.model.feature_importances_, 
            index=self.model.feature_names_in_
        )
        
        plt.figure(figsize=(10, 6))
        importances.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")


class FloodPredictor:
    """Handles flood prediction on new areas of interest."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self) -> None:
        """Load trained model from file."""
        logger.info(f"Loading model from {self.model_path}")
        self.model = load(self.model_path)
        
    def predict_flood_map(self, data_path: str, output_path: str) -> None:
        """Generate flood predictions for new AOI."""
        if self.model is None:
            self.load_model()
            
        logger.info(f"Loading prediction data from {data_path}")
        new_aoi = pd.read_csv(data_path)
        
        # Prepare features (same as training)
        feature_cols = ['ndwi_post', 'mndwi_post', 'awei_post', 'VV',
                       'elevation', 'slope', 'Map', 'jrc_seasonality']
        
        X_new = pd.get_dummies(new_aoi[feature_cols], columns=['Map'])
        
        # Align columns with training data
        X_new = X_new.reindex(self.model.feature_names_in_, axis=1, fill_value=0)
        
        # Make predictions
        proba_new = self.model.predict_proba(X_new)[:, 1]
        pred_new = self.model.predict(X_new)
        
        # Add predictions to original data
        new_aoi['flood_probability'] = proba_new
        new_aoi['flood_predicted'] = pred_new
        
        # Save results
        new_aoi.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate DataFrame has required columns and data."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    return True


def safe_geometry_parsing(geo_string: str) -> Optional[Any]:
    """Safely parse geometry string with fallback methods."""
    if pd.isna(geo_string) or not geo_string:
        return None
    
    try:
        # Try JSON parsing first
        geojson_dict = json.loads(geo_string)
        return shape(geojson_dict)
    except (json.JSONDecodeError, TypeError):
        try:
            # Fallback to ast.literal_eval
            import ast
            geojson_dict = ast.literal_eval(geo_string)
            return shape(geojson_dict)
        except (ValueError, SyntaxError):
            logger.debug(f"Failed to parse geometry: {str(geo_string)[:75]}...")
            return None


class FloodAnalyzer:
    """Main class for flood risk analysis and visualization."""
    
    def __init__(self, config: FloodAnalysisConfig):
        self.config = config
        
    def load_and_validate_data(self) -> pd.DataFrame:
        """Load CSV data with validation."""
        logger.info(f"Loading data from: {self.config.csv_file_path}")
        
        try:
            df = pd.read_csv(self.config.csv_file_path)
            if not validate_dataframe(df, ['.geo', 'flood_predicted']):
                raise ValueError("Invalid DataFrame structure")
            
            logger.info(f"Successfully loaded {len(df)} rows")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {self.config.csv_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def create_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame with geometry parsing."""
        logger.info("Converting to GeoDataFrame...")
        
        geometries = []
        parse_stats = {'success': 0, 'failed': 0}
        
        for _, row in df.iterrows():
            geom = safe_geometry_parsing(row['.geo'])
            geometries.append(geom)
            
            if geom is None:
                parse_stats['failed'] += 1
            else:
                parse_stats['success'] += 1
        
        logger.info(f"Geometry parsing - Success: {parse_stats['success']}, Failed: {parse_stats['failed']}")
        
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=self.config.initial_crs)
        gdf = gdf.dropna(subset=['geometry'])
        
        if gdf.empty:
            raise ValueError("No valid geometries found")
        
        logger.info(f"Created GeoDataFrame with {len(gdf)} features")
        return gdf
    
    def test_fema_service(self, service_config: Dict[str, Any], bounds: tuple) -> Optional[Dict[str, Any]]:
        """Test a single FEMA service configuration."""
        logger.info(f"Testing FEMA service: {service_config['name']}")
        
        session = requests.Session()
        retry_strategy = Retry(
            total=2, 
            backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        minx, miny, maxx, maxy = bounds
        
        # Test service accessibility
        try:
            service_url = f"{service_config['url']}/{service_config['layer_id']}"
            test_response = session.get(f"{service_url}?f=json", timeout=30)
            test_response.raise_for_status()
            
            logger.info(f"Service {service_config['name']} is accessible")
            
        except Exception as e:
            logger.warning(f"Service {service_config['name']} not accessible: {e}")
            return None
        
        # Query data
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
            response = session.get(query_url, params=query_params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            if 'features' in data:
                features = data['features']
                logger.info(f"Service returned {len(features)} features")
                return {'service': service_config, 'data': data}
            else:
                logger.warning(f"Unexpected response format from {service_config['name']}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying {service_config['name']}: {e}")
            return None
    
    def fetch_fema_data(self, bounds: tuple) -> Optional[gpd.GeoDataFrame]:
        """Fetch FEMA data with multiple service fallback."""
        logger.info("Fetching FEMA NFHL data...")
        
        for service_config in self.config.fema_services:
            result = self.test_fema_service(service_config, bounds)
            
            if result is not None:
                try:
                    data = result['data']
                    service_config = result['service']
                    
                    if len(data['features']) == 0:
                        logger.warning(f"No features in {service_config['name']}")
                        continue
                    
                    fema_gdf = gpd.GeoDataFrame.from_features(
                        data['features'], 
                        crs=self.config.initial_crs
                    )
                    
                    logger.info(f"Loaded {len(fema_gdf)} features from {service_config['name']}")
                    
                    # Filter for SFHA zones
                    zone_field = service_config['zone_field']
                    if zone_field in fema_gdf.columns:
                        unique_zones = fema_gdf[zone_field].unique()
                        logger.info(f"Available flood zones: {unique_zones}")
                        
                        sfha_gdf = fema_gdf[fema_gdf[zone_field].isin(self.config.sfha_zones)]
                        
                        if not sfha_gdf.empty:
                            dissolved_sfha = sfha_gdf.dissolve()
                            logger.info(f"Filtered to {len(sfha_gdf)} SFHA features")
                            return dissolved_sfha
                        else:
                            logger.warning("No SFHA zones found, using all features")
                            return fema_gdf.dissolve()
                    else:
                        logger.warning(f"Zone field '{zone_field}' not found")
                        return fema_gdf.dissolve()
                        
                except Exception as e:
                    logger.error(f"Error processing {service_config['name']}: {e}")
                    continue
        
        logger.error("All FEMA services failed")
        return None
    
    def identify_outside_fema(self, gdf: gpd.GeoDataFrame, 
                            fema_gdf: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Identify flood predictions outside FEMA zones."""
        logger.info("Identifying predictions outside FEMA zones...")
        
        if fema_gdf is None or fema_gdf.empty:
            logger.warning("No FEMA data - all floods considered 'outside FEMA'")
            return gdf[gdf['flood_predicted'] == 1].copy()
        
        # Ensure same CRS
        if gdf.crs != fema_gdf.crs:
            gdf_for_sjoin = gdf.to_crs(fema_gdf.crs)
        else:
            gdf_for_sjoin = gdf.copy()
        
        # Spatial join
        sjoin_result = gpd.sjoin(gdf_for_sjoin, fema_gdf, how="left", predicate="intersects")
        outside_fema = sjoin_result[sjoin_result['index_right'].isna()]
        flooded_outside_fema = outside_fema[outside_fema['flood_predicted'] == 1]
        
        logger.info(f"Total predictions: {len(gdf)}")
        logger.info(f"Outside FEMA: {len(outside_fema)}")
        logger.info(f"Flooded outside FEMA: {len(flooded_outside_fema)}")
        
        return flooded_outside_fema
    
    def create_visualization(self, gdf: gpd.GeoDataFrame, 
                           fema_gdf: Optional[gpd.GeoDataFrame],
                           outside_fema: gpd.GeoDataFrame) -> plt.Figure:
        """Create enhanced visualization with basemap."""
        logger.info("Creating visualization...")
        
        # Reproject to Web Mercator
        gdf_proj = gdf.to_crs(self.config.web_mercator_crs)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot study area
        gdf_proj.plot(ax=ax, color='lightgray', edgecolor='gray', 
                     alpha=0.05, linewidth=0.5, label='Study Area')
        
        # Plot FEMA zones
        if fema_gdf is not None and not fema_gdf.empty:
            fema_proj = fema_gdf.to_crs(self.config.web_mercator_crs)
            fema_proj.plot(ax=ax, color='cyan', edgecolor='darkblue', 
                          alpha=0.6, linewidth=1.2, label='FEMA Flood Zones')
        
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
                            label=f'Outside FEMA ({len(outside_proj)})')
        
        # Add basemap
        try:
            ctx.add_basemap(ax, crs=self.config.web_mercator_crs,
                          source=ctx.providers.Esri.WorldImagery, alpha=0.85)
            logger.info("Basemap added successfully")
        except Exception as e:
            logger.warning(f"Could not add basemap: {e}")
        
        # Styling
        ax.set_title('Flood Risk Analysis: Model Predictions vs FEMA Data',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Eastings (m)', fontsize=12)
        ax.set_ylabel('Northings (m)', fontsize=12)
        
        # Set extent with buffer
        if not gdf_proj.empty:
            minx, miny, maxx, maxy = gdf_proj.total_bounds
            buffer_x = (maxx - minx) * 0.15
            buffer_y = (maxy - miny) * 0.15
            ax.set_xlim(minx - buffer_x, maxx + buffer_x)
            ax.set_ylim(miny - buffer_y, maxy + buffer_y)
        
        # Legend and grid
        legend = ax.legend(loc='upper left', frameon=True, facecolor='white',
                          edgecolor='black', shadow=True, fontsize=10)
        legend.get_frame().set_alpha(0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, gdf: gpd.GeoDataFrame, 
                       fema_gdf: Optional[gpd.GeoDataFrame],
                       outside_fema: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        total_predictions = len(gdf)
        flood_count = len(gdf[gdf['flood_predicted'] == 1])
        
        report = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_areas_analyzed': total_predictions,
            'flood_predictions_count': flood_count,
            'flood_percentage': round((flood_count / total_predictions * 100), 2) if total_predictions > 0 else 0,
            'fema_data_available': fema_gdf is not None and not fema_gdf.empty,
            'fema_zones_count': len(fema_gdf) if fema_gdf is not None else 0,
            'flooded_outside_fema_count': len(outside_fema),
            'bbox': gdf.total_bounds.tolist(),
            'crs': str(gdf.crs)
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
        """Save all analysis outputs."""
        output_dir = Path(self.config.output_dir)
        
        try:
            # Save GeoDataFrames
            gdf.to_file(output_dir / 'all_predictions.geojson', driver='GeoJSON')
            
            if not outside_fema.empty:
                outside_fema.to_file(output_dir / 'outside_fema.geojson', driver='GeoJSON')
            
            # Save plots
            fig.savefig(output_dir / 'analysis_map.png', dpi=300, bbox_inches='tight')
            fig.savefig(output_dir / 'analysis_map.pdf', bbox_inches='tight')
            
            logger.info("All outputs saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving outputs: {e}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis pipeline."""
        logger.info("Starting Flood Risk Analysis Pipeline")
        
        try:
            # Load and process data
            df = self.load_and_validate_data()
            gdf = self.create_geodataframe(df)
            
            # Fetch FEMA data
            fema_gdf = self.fetch_fema_data(gdf.total_bounds)
            
            # Analyze areas outside FEMA zones
            outside_fema = self.identify_outside_fema(gdf, fema_gdf)
            
            # Create visualization
            fig = self.create_visualization(gdf, fema_gdf, outside_fema)
            
            # Generate report and save outputs
            report = self.generate_report(gdf, fema_gdf, outside_fema)
            self.save_outputs(gdf, outside_fema, fig)
            
            plt.show()
            
            logger.info("Analysis completed successfully!")
            return report
            
        except Exception as e:
            logger.critical(f"Analysis failed: {e}", exc_info=True)
            raise


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("    Flood Risk Analysis Application")
    print("="*60 + "\n")
    
    try:
        # Initialize configuration and run analysis
        config = FloodAnalysisConfig()
        analyzer = FloodAnalyzer(config)
        report = analyzer.run_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("    ANALYSIS SUMMARY")
        print("="*60)
        print(f"Analysis Date: {report['analysis_date']}")
        print(f"Total Areas: {report['total_areas_analyzed']}")
        print(f"Flood Predictions: {report['flood_predictions_count']} ({report['flood_percentage']}%)")
        print(f"FEMA Data Available: {'Yes' if report['fema_data_available'] else 'No'}")
        print(f"FEMA Zones: {report['fema_zones_count']}")
        print(f"Flooded Outside FEMA: {report['flooded_outside_fema_count']}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()
