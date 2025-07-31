"""
SI_Flooding: A comprehensive flood risk analysis toolkit.

This package combines machine learning models with FEMA flood zone data
for geospatial analysis and flood prediction.
"""

__version__ = "1.0.0"
__author__ = "SI_Flooding Team"
__email__ = "contact@si-flooding.org"

from .si_flooding.flood_analyzer import FloodAnalyzer
from .si_flooding.geofloodcast import GeoFloodCast
from .si_flooding.utils import (
    setup_logging,
    validate_coordinates,
    fetch_fema_data_cached,
    create_enhanced_visualization,
)

__all__ = [
    "FloodAnalyzer",
    "GeoFloodCast",
    "setup_logging",
    "validate_coordinates",
    "fetch_fema_data_cached",
    "create_enhanced_visualization",
]
