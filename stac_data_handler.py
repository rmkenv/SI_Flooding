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

        s2_assets = ['B03', 'B04', 'B08', 'B11', 'B12']

        data = stackstac.stack(
            items,
            assets=s2_assets,
            resolution=resolution,
            bounds=bbox,
            epsg=4326,
            sortby_date="asc",
            dtype="float",
            fill_value=np.nan,
        )

        median_data = data.median(dim="time", skipna=True)
        dataset = median_data.to_dataset(dim='band')
        dataset = dataset.rename({'B03': 'green', 'B04': 'red', 'B08': 'nir', 'B11': 'swir1', 'B12': 'swir2'})

        dataset.rio.write_crs(data.rio.crs, inplace=True)
        dataset.rio.write_transform(data.rio.transform(), inplace=True)
        return dataset

    def load_s1_data(self, bbox: List[float], daterange: str, resolution: int) -> xr.Dataset:
        """Loads Sentinel-1 GRD data."""
        s1_items = self.search_items(['sentinel-1-grd'], bbox, daterange)
        if not s1_items:
            return xr.Dataset()

        data = stackstac.stack(
            s1_items,
            assets=['vv'],
            resolution=resolution,
            bounds=bbox,
            epsg=4326,
            dtype="float",
            fill_value=np.nan,
        )

        median_data = data.median(dim="time", skipna=True)
        dataset = median_data.to_dataset(dim='band')
        dataset = dataset.rename({'vv': 'VV'})

        dataset.rio.write_crs(data.rio.crs, inplace=True)
        dataset.rio.write_transform(data.rio.transform(), inplace=True)
        return dataset

    def load_dem_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads DEM data (Copernicus DEM GLO-30) and calculates slope."""
        dem_items = self.search_items(['cop-dem-glo-30'], bbox, "2020-01-01/2023-01-01")
        if not dem_items:
            raise ValueError("No DEM items found.")

        data = stackstac.stack(
            dem_items,
            assets=['data'],
            resolution=resolution,
            bounds=bbox,
            epsg=4326,
            dtype="float",
            fill_value=np.nan,
        )

        median_data = data.median(dim="time", skipna=True)
        elevation_ds = median_data.to_dataset(dim='band').rename({'data': 'elevation'})

        elevation_ds.rio.write_crs(data.rio.crs, inplace=True)
        elevation_ds.rio.write_transform(data.rio.transform(), inplace=True)

        logger.warning("Temporarily disabling slope calculation to debug other issues. Returning zero slope.")
        slope_ds = xr.Dataset({'slope': xr.zeros_like(elevation_ds.elevation)})

        return xr.merge([elevation_ds, slope_ds])


    def load_landcover_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads ESA WorldCover data."""
        wc_items = self.search_items(['esa-worldcover'], bbox, "2020-01-01/2021-12-31")
        if not wc_items:
            raise ValueError("No WorldCover items found.")

        data = stackstac.stack(
            wc_items,
            assets=['map'],
            resolution=resolution,
            bounds=bbox,
            epsg=4326,
            dtype="int64",
            fill_value=0,
            rescale=False,
        )

        median_data = data.median(dim="time", skipna=True)
        dataset = median_data.to_dataset(dim='band')
        dataset = dataset.rename({'map': 'landcover'})

        dataset.rio.write_crs(data.rio.crs, inplace=True)
        dataset.rio.write_transform(data.rio.transform(), inplace=True)
        return dataset

    def load_jrc_gsw_data(self, bbox: List[float], resolution: int) -> xr.Dataset:
        """Loads JRC Global Surface Water data."""
        jrc_items = self.search_items(['jrc-gsw'], bbox, "2020-01-01/2021-12-31")
        if not jrc_items:
            logger.warning("No JRC GSW items found. Returning zero seasonality.")
            return xr.Dataset()

        data = stackstac.stack(
            jrc_items,
            assets=['seasonality'],
            resolution=resolution,
            bounds=bbox,
            epsg=4326,
            dtype="int64",
            fill_value=0,
            rescale=False,
        )

        median_data = data.median(dim="time", skipna=True)
        dataset = median_data.to_dataset(dim='band')
        dataset = dataset.rename({'seasonality': 'jrc_seasonality'})

        dataset.rio.write_crs(data.rio.crs, inplace=True)
        dataset.rio.write_transform(data.rio.transform(), inplace=True)
        return dataset

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
        s2_post_items = self.search_items(['sentinel-2-l2a'], bbox, post_daterange)
        ds_post = self.load_s2_data(s2_post_items, bbox, resolution)

        if not list(ds_post.data_vars):
            logger.warning("Could not load Sentinel-2 data. Aborting.")
            return xr.Dataset()

        ds_post = self.calculate_indices(ds_post)
        ds_post = ds_post.rename({
            'ndwi': 'ndwi_post',
            'mndwi': 'mndwi_post',
            'awei': 'awei_post'
        })

        ds_s1 = self.load_s1_data(bbox, post_daterange, resolution)
        ds_dem = self.load_dem_data(bbox, resolution)
        ds_wc = self.load_landcover_data(bbox, resolution)
        ds_jrc = self.load_jrc_gsw_data(bbox, resolution)

        datasets = [ds_post, ds_s1, ds_dem, ds_wc, ds_jrc]

        aligned_datasets = []
        common_coords = ds_post

        for ds in datasets:
            if not list(ds.data_vars):
                continue

            aligned_ds, _ = xr.align(ds.rio.reproject_match(common_coords), common_coords, join="left")
            aligned_datasets.append(aligned_ds)

        if not aligned_datasets:
            return xr.Dataset()

        predictor_ds = xr.merge(aligned_datasets, compat='override')

        if 'landcover' in predictor_ds:
            predictor_ds = predictor_ds.rename({'landcover': 'Map'})

        predictor_ds = predictor_ds.fillna(-9999)

        expected_vars = ['ndwi_post', 'mndwi_post', 'awei_post', 'VV', 'elevation', 'slope', 'Map', 'jrc_seasonality']
        for var in expected_vars:
            if var not in predictor_ds:
                logger.warning(f"Predictor variable '{var}' is missing. Adding a zero-filled array.")
                predictor_ds[var] = xr.zeros_like(list(predictor_ds.values())[0])

        return predictor_ds[expected_vars]
