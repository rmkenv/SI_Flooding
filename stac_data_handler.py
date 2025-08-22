import pystac_client
import planetary_computer
import rioxarray
import stackstac
import xarray as xr
import numpy as np
from typing import Optional, List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class STACDataHandler:
    """
    Handles searching, loading, and processing of geospatial data from STAC catalogs.
    """

    def __init__(self, stac_endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1"):
        """
        Initializes the data handler with a specific STAC endpoint.

        :param stac_endpoint: The URL of the STAC API endpoint.
        """
        self.catalog = pystac_client.Client.open(stac_endpoint)
        try:
            planetary_computer.set_subscription_key_from_environ()
        except Exception:
            logger.warning("Planetary Computer subscription key not found in environment variables. Performance may be limited.")


    def search_items(self, collections: List[str], bbox: List[float], daterange: str) -> List[Dict[str, Any]]:
        """
        Searches for STAC items and returns a list of signed item dictionaries.
        """
        search = self.catalog.search(
            collections=collections,
            bbox=bbox,
            datetime=daterange,
        )
        items = list(search.get_items())
        logger.info(f"Found {len(items)} items for collections {collections} in {daterange}")
        return [planetary_computer.sign(item).to_dict() for item in items]

    def load_s2_data(self, items: List[Dict[str, Any]], bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads Sentinel-2 L2A data, performs cloud masking, and creates a median composite."""
        if not items:
            return xr.Dataset()

        # TODO: Implement a proper cloud masking strategy.
        s2_assets = ['B03', 'B04', 'B08', 'B11', 'B12'] # green, red, nir, swir1, swir2

        ds = stackstac.stack(
            items,
            assets=s2_assets,
            resolution=resolution,
            bounds=bbox,
            sortby_date="asc",
            dtype="float32",
            fill_value=np.nan, # Use nan for fill value for proper nanmedian
        )

        median_ds = ds.median(dim="time", skipna=True)
        return median_ds.rename({'B03': 'green', 'B04': 'red', 'B08': 'nir', 'B11': 'swir1', 'B12': 'swir2'})

    def load_s1_data(self, bbox: List[float], daterange: str, resolution: int) -> xr.Dataset:
        """Loads Sentinel-1 GRD data."""
        s1_items = self.search_items(['sentinel-1-grd'], bbox, daterange)
        if not s1_items:
            return xr.Dataset()

        ds_s1 = stackstac.stack(
            s1_items,
            assets=['vv'],
            resolution=resolution,
            bounds=bbox,
            dtype="float32",
            fill_value=np.nan,
        )
        return ds_s1.rename({'vv': 'VV'}).median(dim="time", skipna=True)

    def load_dem_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads DEM data (Copernicus DEM GLO-30) and calculates slope."""
        dem_items = self.search_items(['cop-dem-glo-30'], bbox, "2020-01-01/2023-01-01")
        if not dem_items:
            raise ValueError("No DEM items found.")

        ds_dem = stackstac.stack(
            dem_items,
            assets=['data'],
            resolution=resolution,
            bounds=bbox,
            dtype="float32",
            fill_value=np.nan,
        )

        elevation = ds_dem.rename({'data': 'elevation'}).median(dim="time", skipna=True)

        # Calculate slope. Note: The CRS should be projected for accurate slope calculation.
        # Here we use the default CRS from stackstac which is usually geographic.
        # For a production system, reprojecting to a local UTM zone would be better.
        try:
            slope = rioxarray.terrain.slope(elevation.elevation)
            slope = slope.rename('slope')
        except Exception as e:
            logger.warning(f"Could not calculate slope: {e}. Returning zero slope.")
            slope = xr.zeros_like(elevation.elevation).rename('slope')

        return xr.merge([elevation, slope])


    def load_landcover_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads ESA WorldCover data."""
        wc_items = self.search_items(['esa-worldcover'], bbox, "2020-01-01/2021-12-31")
        if not wc_items:
            raise ValueError("No WorldCover items found.")

        ds_wc = stackstac.stack(
            wc_items,
            assets=['map'],
            resolution=resolution,
            bounds=bbox,
            dtype="uint8",
            fill_value=0,
        )
        return ds_wc.rename({'map': 'landcover'}).median(dim="time", skipna=True)

    def load_jrc_gsw_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads JRC Global Surface Water data."""
        jrc_items = self.search_items(['jrc-gsw'], bbox, "2020-01-01/2021-12-31")
        if not jrc_items:
            logger.warning("No JRC GSW items found. Returning zero seasonality.")
            return xr.Dataset({'jrc_seasonality': (('y', 'x'), np.zeros((1,1)))})

        ds_jrc = stackstac.stack(
            jrc_items,
            assets=['seasonality'],
            resolution=resolution,
            bounds=bbox,
            dtype="uint8",
            fill_value=0,
        )
        return ds_jrc.rename({'seasonality': 'jrc_seasonality'}).median(dim="time", skipna=True)

    def calculate_indices(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculates water indices (NDWI, MNDWI, AWEI)."""
        required_bands = ['green', 'nir', 'swir1', 'swir2']
        if not all(band in ds for band in required_bands):
            logger.warning("Not all required bands present for index calculation.")
            return ds

        ds['ndwi'] = (ds['green'] - ds['nir']) / (ds['green'] + ds['nir'])
        ds['mndwi'] = (ds['green'] - ds['swir1']) / (ds['green'] + ds['swir1'])
        ds['awei'] = 4 * (ds['green'] - ds['swir1']) - (0.25 * ds['nir'] + 2.75 * ds['swir2'])
        return ds

    def get_predictor_dataset(self, bbox: List[float], post_daterange: str, resolution: int, pre_daterange: str = None) -> xr.Dataset:
        """
        Orchestrates the process to return a dataset of predictor variables.
        """
        # 1. Load Post-event Sentinel-2 data and calculate indices
        s2_post_items = self.search_items(['sentinel-2-l2a'], bbox, post_daterange)
        ds_post = self.load_s2_data(s2_post_items, bbox, resolution)
        ds_post = self.calculate_indices(ds_post)
        ds_post = ds_post.rename({
            'ndwi': 'ndwi_post',
            'mndwi': 'mndwi_post',
            'awei': 'awei_post'
        })

        # 2. Load other datasets
        ds_s1 = self.load_s1_data(bbox, post_daterange, resolution)
        ds_dem = self.load_dem_data(bbox, resolution)
        ds_wc = self.load_landcover_data(bbox, resolution)
        ds_jrc = self.load_jrc_gsw_data(bbox, resolution)

        # 3. Combine all datasets
        datasets = [ds_post, ds_s1, ds_dem, ds_wc, ds_jrc]

        # Align all datasets
        aligned_datasets = []
        common_coords = None
        for ds in datasets:
            if not ds.coords: # Skip empty datasets
                continue
            if common_coords is None:
                common_coords = ds
            aligned_ds, _ = xr.align(ds, common_coords, join="left")
            aligned_datasets.append(aligned_ds)

        predictor_ds = xr.merge(aligned_datasets)

        # The ML model expects features named 'Map' for landcover.
        if 'landcover' in predictor_ds:
            predictor_ds = predictor_ds.rename({'landcover': 'Map'})

        # Fill NA values - the model expects numeric inputs
        # Using -9999 as a nodata value, but a more sophisticated imputation could be used.
        predictor_ds = predictor_ds.fillna(-9999)

        # Ensure all required variables are present
        expected_vars = ['ndwi_post', 'mndwi_post', 'awei_post', 'VV', 'elevation', 'slope', 'Map', 'jrc_seasonality']
        for var in expected_vars:
            if var not in predictor_ds:
                logger.warning(f"Predictor variable '{var}' is missing. Adding a zero-filled array.")
                predictor_ds[var] = xr.zeros_like(list(predictor_ds.values())[0])


        return predictor_ds[expected_vars]
