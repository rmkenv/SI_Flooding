"""
Miami Flood Detection and Analysis using Google Earth Engine Python API
======================================================================

This script performs flood detection for Miami, FL using Sentinel-2 optical 
and Sentinel-1 SAR data, calculates water indices, and exports results.
"""

import ee

# Initialize Earth Engine
ee.Initialize()

# ========================= USER SETTINGS ===================================
pre_start = '2020-01-01'
pre_end = '2023-12-31'
post_start = '2024-01-01'
post_end = '2024-12-31'
cloud_filt = 50  # Cloud probability threshold (e.g., 50 means <50% cloud probability)
scale = 30  # Resolution for exports and analysis in meters
max_images_pre = 50  # Max images for pre-event composite (not currently used in monthly composite)
max_images_post = 20  # Max images for post-event composite (not currently used in monthly composite)

# ================= AREA OF INTEREST & NAMING ==============================
# Define the Area of Interest (AOI) - Miami, FL
miami = ee.Geometry.Rectangle([-80.8738, 25.1398, -80.1308, 25.9564])
date_string = post_start.replace('-', '')  # Create a date string for file naming (GEE-safe)

print(f"AOI defined for Miami: {miami.getInfo()}")
print(f"Date string for exports: {date_string}")

# ========== ROBUST JOIN: S2_SR_HARMONIZED + S2_CLOUD_PROBABILITY ==========
# Load Sentinel-2 Surface Reflectance Harmonized data
s2sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(miami)
        .filterDate(pre_start, post_end))

# Load Sentinel-2 Cloud Probability data
s2clouds = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(miami)
            .filterDate(pre_start, post_end))

# Perform a robust join to combine Sentinel-2 SR with cloud probabilities
join = ee.Join.saveFirst('cloud_mask')
filter_join = ee.Filter.equals(leftField='system:index', rightField='system:index')
s2_with_cloud = join.apply(s2sr, s2clouds, filter_join)

def check_valid_bands(image):
    """Function to check if an image has all required bands for water indices."""
    image = ee.Image(image)
    bn = image.bandNames()  # bn is an ee.List

    # Check if each band exists (these are ee.Bool objects)
    has_b3_bool = bn.contains('B3')
    has_b4_bool = bn.contains('B4')
    has_b8_bool = bn.contains('B8')
    has_b11_bool = bn.contains('B11')
    has_b12_bool = bn.contains('B12')

    # Convert each ee.Bool to an ee.Number (1 for true, 0 for false) using ee.Algorithms.If
    val_b3 = ee.Algorithms.If(has_b3_bool, ee.Number(1), ee.Number(0))
    val_b4 = ee.Algorithms.If(has_b4_bool, ee.Number(1), ee.Number(0))
    val_b8 = ee.Algorithms.If(has_b8_bool, ee.Number(1), ee.Number(0))
    val_b11 = ee.Algorithms.If(has_b11_bool, ee.Number(1), ee.Number(0))
    val_b12 = ee.Algorithms.If(has_b12_bool, ee.Number(1), ee.Number(0))

    # Cast the results to ee.Number to ensure add() works correctly
    val_b3 = ee.Number(val_b3)
    val_b4 = ee.Number(val_b4)
    val_b8 = ee.Number(val_b8)
    val_b11 = ee.Number(val_b11)
    val_b12 = ee.Number(val_b12)

    # Sum these integer values
    sum_of_checks = (val_b3
                     .add(val_b4)
                     .add(val_b8)
                     .add(val_b11)
                     .add(val_b12))

    # If all 5 bands are present, the sum will be 5
    is_valid = sum_of_checks.eq(5)

    return image.set('isValid', is_valid)

# Apply band validation and filter
s2_with_cloud = (ee.ImageCollection(s2_with_cloud)
                 .map(check_valid_bands)
                 .filter(ee.Filter.eq('isValid', 1)))

print(f"S2 collection after validation: {s2_with_cloud.size().getInfo()} images")

def mask_s2_with_cloud_prob(image):
    """S2 cloud/shadow and probability masking function."""
    # Explicitly cast the input 'image' to an ee.Image for type safety
    image = ee.Image(image)

    # Get the cloud probability image from the joined collection
    cloud_prob_image = ee.Image(image.get('cloud_mask'))
    # Handle cases where cloudProbImage might be null
    cloud_prob = ee.Algorithms.If(cloud_prob_image, 
                                  cloud_prob_image.select('probability'), 
                                  ee.Image(0))
    cloud_prob = ee.Image(cloud_prob)  # Cast the result of If to an Image

    # Create a cloud mask based on the cloud probability threshold
    cloud_mask = cloud_prob.lt(cloud_filt)

    # Apply the cloud mask, scale bands, select relevant bands, and copy properties
    return (image.updateMask(cloud_mask)
            .divide(10000)
            .select(['B3', 'B4', 'B8', 'B11', 'B12'])
            .copyProperties(image, ['system:time_start']))

# ========== MONTHLY TIME SERIES COMPOSITES FOR PRE/POST ====================
def composite_monthly(collection, start, end):
    """Function to create monthly median composites from an image collection."""
    start_date = ee.Date(start)
    end_date = ee.Date(end)

    # Generate a sequence of months from start to end date
    months_to_process = ee.List.sequence(0, end_date.difference(start_date, 'month').round())

    def create_monthly_composite(offset_val):
        offset = ee.Number(offset_val)  # Cast to ee.Number for server-side operations
        current_month_start = start_date.advance(offset, 'month')
        current_month_end = current_month_start.advance(1, 'month')

        filtered = (collection
                   .filterDate(current_month_start, current_month_end)
                   .map(mask_s2_with_cloud_prob))

        # Explicitly cast 'filtered' to ee.ImageCollection to ensure .median() is available
        filtered = ee.ImageCollection(filtered)

        # Check the size of the filtered collection BEFORE computing median
        count = filtered.size()

        monthly_result = ee.Algorithms.If(
            count.gt(0),  # If there are images in the filtered collection
            filtered.median().set({
                'system:time_start': current_month_start.millis(),
                'year': current_month_start.get('year'),
                'month': current_month_start.get('month')
            }),
            # Else return an empty, dummy image (placeholder will be filtered out later)
            ee.Image([]).set('system:time_start', current_month_start.millis())
        )
        return monthly_result

    monthly_composites = months_to_process.map(create_monthly_composite)
    monthly_composites = ee.List(monthly_composites)

    # Convert the list of images to an ImageCollection
    img_collection = ee.ImageCollection(monthly_composites)

    # Filter out the empty placeholder images and clip to Miami
    return (img_collection
            .filter(ee.Filter.listContains('system:band_names', 'B3'))
            .map(lambda image: image.clip(miami)))

# Generate pre-event and post-event monthly composites
pre_monthly = composite_monthly(s2_with_cloud, pre_start, pre_end)
post_monthly = composite_monthly(s2_with_cloud, post_start, post_end)

print(f"Pre-monthly ImageCollection size: {pre_monthly.size().getInfo()}")
print(f"Post-monthly ImageCollection size: {post_monthly.size().getInfo()}")

# FINAL GUARD FOR EMPTY COLLECTIONS BEFORE MEDIAN
def create_dummy_image():
    """Create a dummy image with expected bands (zeros)."""
    return (ee.Image()
            .addBands(ee.Image.constant(0).rename('B3'))
            .addBands(ee.Image.constant(0).rename('B4'))
            .addBands(ee.Image.constant(0).rename('B8'))
            .addBands(ee.Image.constant(0).rename('B11'))
            .addBands(ee.Image.constant(0).rename('B12')))

pre = ee.Image(ee.Algorithms.If(
    pre_monthly.size().gt(0),
    pre_monthly.median().clip(miami),  # Ensure final median is also clipped
    create_dummy_image()
))

post = ee.Image(ee.Algorithms.If(
    post_monthly.size().gt(0),
    post_monthly.median().clip(miami),  # Ensure final median is also clipped
    create_dummy_image()
))

print("Pre-event and post-event composite images created")

# =============== WATER INDICES: NDWI, MNDWI, AWEI ==========================
# Calculate Normalized Difference Water Index (NDWI)
ndwi_pre = pre.normalizedDifference(['B3', 'B8']).rename('ndwi_pre')
ndwi_post = post.normalizedDifference(['B3', 'B8']).rename('ndwi_post')

# Calculate Modified Normalized Difference Water Index (MNDWI)
mndwi_pre = pre.normalizedDifference(['B3', 'B11']).rename('mndwi_pre')
mndwi_post = post.normalizedDifference(['B3', 'B11']).rename('mndwi_post')

# Calculate Automated Water Extraction Index (AWEI)
awei_pre = pre.expression(
    '4*(GREEN-SWIR1)-(0.25*NIR+2.75*SWIR2)', {
        'GREEN': pre.select('B3'),
        'SWIR1': pre.select('B11'),
        'NIR': pre.select('B8'),
        'SWIR2': pre.select('B12')
    }).rename('awei_pre')

awei_post = post.expression(
    '4*(GREEN-SWIR1)-(0.25*NIR+2.75*SWIR2)', {
        'GREEN': post.select('B3'),
        'SWIR1': post.select('B11'),
        'NIR': post.select('B8'),
        'SWIR2': post.select('B12')
    }).rename('awei_post')

# ----------- DYNAMIC/SCENE THRESHOLDS ---------------
# These thresholds should ideally be determined dynamically or locally
t_ndwi = 0.30
t_mndwi = 0.0
t_awei = 0.0

# Apply thresholds to create binary water masks for each index
water_pre_ndwi = ndwi_pre.gt(t_ndwi)
water_post_ndwi = ndwi_post.gt(t_ndwi)
water_pre_mndwi = mndwi_pre.gt(t_mndwi)
water_post_mndwi = mndwi_post.gt(t_mndwi)
water_pre_awei = awei_pre.gt(t_awei)
water_post_awei = awei_post.gt(t_awei)

# Create a consensus water mask (at least 2 out of 3 indices agree)
water_pre_sum = water_pre_ndwi.add(water_pre_mndwi).add(water_pre_awei)
water_post_sum = water_post_ndwi.add(water_post_mndwi).add(water_post_awei)
water_pre_consensus = water_pre_sum.gte(2)
water_post_consensus = water_post_sum.gte(2)

# ============ SAR FLOOD MASK (SENTINEL-1) ==================================
# Load Sentinel-1 Ground Range Detected (GRD) data
s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
      .filterBounds(miami)
      .filterDate(post_start, post_end)
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
      .filter(ee.Filter.eq('instrumentMode', 'IW'))
      .select(['VV']))

print(f"S1 collection size: {s1.size().getInfo()}")

# Guard against empty S1 collection
s1_composite = ee.Image(ee.Algorithms.If(
    s1.size().gt(0),
    s1.median().clip(miami),
    ee.Image.constant(0).rename('VV')  # Dummy VV band if no S1 data
))

vv_threshold = -17  # Threshold for water detection in VV band (dB)
s1_water = s1_composite.select('VV').lt(vv_threshold)  # Binary water mask from S1

# ======== FLOOD CHANGE DETECTION, POST-PROCESSING, URBAN/CLEANUP ===========
# Identify potential flood areas: Post-event water where pre-event was not water
flood_map = water_post_consensus.And(water_pre_consensus.Not())

# Combine optical flood map with SAR-derived water
combined_flood = flood_map.Or(s1_water)

# Categorize water change:
persistent_water = water_post_consensus.And(water_pre_consensus)  # Water in both periods
new_water = water_post_consensus.And(water_pre_consensus.Not())  # New water (flood)
lost_water = water_pre_consensus.And(water_post_consensus.Not())  # Water lost

# Create a change map with different values for each category
change_map = (persistent_water.multiply(3)  # Persistent water = 3
              .add(new_water.multiply(2))   # New water = 2
              .add(lost_water.multiply(1))) # Lost water = 1

# Post-processing: Remove small isolated pixels (noise)
size_threshold = 10  # Minimum connected pixel count to be considered valid
connected_pixels = combined_flood.connectedPixelCount(size_threshold, False)
cleaned_flood = combined_flood.updateMask(connected_pixels.gte(size_threshold))

# Mask out urban areas from the flood map (assuming urban areas are not floods)
urban_mask = ee.Image('ESA/WorldCover/v100/2020').eq(50)  # WorldCover urban class is 50
flood_urban_masked = cleaned_flood.updateMask(urban_mask.Not())

# ============ REMOVE PERMANENT WATER (JRC GLOBAL SURFACE WATER) ============
# Load JRC Global Surface Water dataset
gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
seasonality = gsw.select('seasonality')  # Select seasonality band
# Pixels with seasonality >= 10 are considered permanent water
permanent_water_mask = seasonality.gte(10)
not_permanent_water = permanent_water_mask.Not()  # Mask for non-permanent water

# Final flood map by removing permanent water bodies
flood_final = flood_urban_masked.updateMask(not_permanent_water)

# ============ ADDITIONAL ML PREDICTORS =====================================
# Add auxiliary data for potential machine learning classification
elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(miami)
slope = ee.Terrain.slope(elevation)
landcover = ee.Image('ESA/WorldCover/v100/2020').select('Map').clip(miami)

print("All processing completed successfully!")

# ============ EXPORT FUNCTIONS ===============================
def export_flood_mask():
    """Export the final flood mask to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=flood_final,
        description=f'flood_mask_final_{date_string}',
        folder='GEE_Flood',
        fileNamePrefix=f'flood_mask_final_{date_string}',
        region=miami,
        scale=scale,
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Flood mask export task started: {task.id}")
    return task

def export_samples():
    """Prepare stratified samples for machine learning and export to Google Drive."""
    samples = (flood_final.rename('flood')  # Renamed the flood mask band for clarity
               .addBands(ndwi_post)
               .addBands(mndwi_post)
               .addBands(awei_post)
               .addBands(s1_water)
               .addBands(elevation)
               .addBands(slope)
               .addBands(landcover)
               .addBands(seasonality.rename('jrc_seasonality'))
               .stratifiedSample(
                   numPoints=3000,
                   classBand='flood',  # Use the renamed band
                   region=miami,
                   scale=scale,
                   seed=42,
                   geometries=True
               ))

    task = ee.batch.Export.table.toDrive(
        collection=samples,
        description=f'FloodSamples_{date_string}',
        folder='GEE_Flood',
        fileNamePrefix=f'FloodSamples_{date_string}',
        fileFormat='CSV'
    )
    task.start()
    print(f"Samples export task started: {task.id}")
    return task

def export_change_map():
    """Export the change map to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=change_map,
        description=f'changeMap_{date_string}',
        folder='GEE_Flood',
        fileNamePrefix=f'changeMap_{date_string}',
        region=miami,
        scale=scale,
        maxPixels=1e9,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"Change map export task started: {task.id}")
    return task

def export_predictor_bands():
    """Export predictor rasters for offline ML prediction."""
    # 1. Rename bands and EXPLICITLY CAST TO FLOAT32 for data type consistency
    ndwi_post_renamed = ndwi_post.rename('ndwi_post').toFloat()  # Cast to Float32
    mndwi_post_renamed = mndwi_post.rename('mndwi_post').toFloat()  # Cast to Float32
    awei_post_renamed = awei_post.rename('awei_post').toFloat()  # Cast to Float32
    # The s1Composite already selects 'VV' and is clipped, so we just rename that band
    s1_vv_renamed = s1_composite.select('VV').rename('VV').toFloat()  # Cast to Float32

    elevation_renamed = elevation.rename('elevation').toFloat()  # Cast to Float32
    slope_renamed = slope.rename('slope').toFloat()  # Cast to Float32
    # WorldCover 'Map' band is Uint8. Cast to Float32
    landcover_renamed = landcover.rename('Map').toFloat()  # Cast to Float32
    # JRC 'seasonality' band is Uint8. Cast to Float32
    jrc_seasonality_renamed = seasonality.rename('jrc_seasonality').toFloat()  # Cast to Float32

    # 2. Combine all predictor bands into a single image
    predictor_image = (ndwi_post_renamed
                      .addBands(mndwi_post_renamed)
                      .addBands(awei_post_renamed)
                      .addBands(s1_vv_renamed)
                      .addBands(elevation_renamed)
                      .addBands(slope_renamed)
                      .addBands(landcover_renamed)
                      .addBands(jrc_seasonality_renamed))

    print(f"Combined Predictor Image bands: {predictor_image.bandNames().getInfo()}")

    # 3. Export the combined predictor image to Google Drive as a GeoTIFF
    task = ee.batch.Export.image.toDrive(
        image=predictor_image,
        description=f'predictor_bands_for_ml_{date_string}',  # Unique description for the task
        folder='GEE_Flood_Predictors',  # A new folder in your Drive for these GeoTIFFs
        fileNamePrefix=f'predictors_{date_string}',  # Prefix for the output GeoTIFF file
        region=miami,  # Your Area of Interest
        scale=scale,  # Resolution (defined in USER SETTINGS)
        maxPixels=1e9,  # Maximum number of pixels to export
        fileFormat='GeoTIFF'
        # You might want to specify 'crs' if you need a specific projection, e.g., 'EPSG:4326'
        # crs='EPSG:4326'
    )
    task.start()
    print(f"Predictor bands export task started: {task.id}")
    return task

# ============ MAIN EXECUTION ===============================
if __name__ == "__main__":
    print("Starting Miami flood detection and analysis...")
    
    # Start all export tasks
    flood_task = export_flood_mask()
    samples_task = export_samples()
    change_task = export_change_map()
    predictor_task = export_predictor_bands()
    
    print("\n" + "="*60)
    print("ALL EXPORT TASKS HAVE BEEN STARTED!")
    print("="*60)
    print("Check your Google Drive folders:")
    print("  - 'GEE_Flood' for flood masks, samples, and change maps")
    print("  - 'GEE_Flood_Predictors' for ML predictor bands")
    print("\nMonitor task progress at: https://code.earthengine.google.com/tasks")
    
    # Optional: Print some basic statistics
    try:
        print(f"\nBasic Statistics for Miami Analysis:")
        print(f"AOI area: {miami.area().divide(1e6).getInfo():.2f} km²")
        print(f"Pre-event monthly composites: {pre_monthly.size().getInfo()}")
        print(f"Post-event monthly composites: {post_monthly.size().getInfo()}")
        print(f"S1 SAR images used: {s1.size().getInfo()}")
    except Exception as e:
        print(f"Could not retrieve statistics: {e}")
    
    print("\nPredictors exported for ML include:")
    print("  - NDWI, MNDWI, AWEI (water indices)")
    print("  - Sentinel-1 VV backscatter")
    print("  - Elevation and slope")
    print("  - Land cover classification")
    print("  - JRC water seasonality")
    
    print("\nScript completed successfully!")
