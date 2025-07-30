// ========================= USER SETTINGS ===================================
var preStart = '2020-01-01';
var preEnd = '2023-12-31';
var postStart = '2024-01-01';
var postEnd = '2024-12-31';
var cloudFilt = 50; // Cloud probability threshold (e.g., 50 means <50% cloud probability)
var scale = 30; // Resolution for exports and analysis in meters
var maxImagesPre = 50; // Max images for pre-event composite (not currently used in monthly composite)
var maxImagesPost = 20; // Max images for post-event composite (not currently used in monthly composite)

// ================= AREA OF INTEREST & NAMING ==============================
// Define the Area of Interest (AOI) this example is for Miami, FL 
var miami = ee.Geometry.Rectangle([-80.8738, 25.1398, -80.1308, 25.9564]);
Map.centerObject(miami, 10); // Center the map on the AOI with a zoom level of 10
var dateString = postStart.split('-').join(''); // Create a date string for file naming (GEE-safe)

// ========== ROBUST JOIN: S2_SR_HARMONIZED + S2_CLOUD_PROBABILITY ==========
// Load Sentinel-2 Surface Reflectance Harmonized data
var s2sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(miami)
    .filterDate(preStart, postEnd);

// Load Sentinel-2 Cloud Probability data
var s2clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterBounds(miami)
    .filterDate(preStart, postEnd);

// Perform a robust join to combine Sentinel-2 SR with cloud probabilities
var join = ee.Join.saveFirst('cloud_mask');
var filter = ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
});
var s2WithCloud = join.apply(s2sr, s2clouds, filter);

// Function to check if an image has all required bands for water indices
// Using .map() to set a property, then filtering by that property for robustness.
s2WithCloud = ee.ImageCollection(s2WithCloud)
    .map(function(image) {
        var bn = image.bandNames(); // bn is an ee.List

        // Check if each band exists (these are ee.Bool objects)
        var hasB3_bool = bn.contains('B3');
        var hasB4_bool = bn.contains('B4');
        var hasB8_bool = bn.contains('B8');
        var hasB11_bool = bn.contains('B11');
        var hasB12_bool = bn.contains('B12');

        // Convert each ee.Bool to an ee.Number (1 for true, 0 for false) using ee.Algorithms.If
        // This is the most robust way when direct .toInt() or .and() chaining cause issues.
        var valB3 = ee.Algorithms.If(hasB3_bool, ee.Number(1), ee.Number(0));
        var valB4 = ee.Algorithms.If(hasB4_bool, ee.Number(1), ee.Number(0));
        var valB8 = ee.Algorithms.If(hasB8_bool, ee.Number(1), ee.Number(0));
        var valB11 = ee.Algorithms.If(hasB11_bool, ee.Number(1), ee.Number(0));
        var valB12 = ee.Algorithms.If(hasB12_bool, ee.Number(1), ee.Number(0));

        // Cast the results of ee.Algorithms.If to ee.Number to ensure add() works correctly
        valB3 = ee.Number(valB3);
        valB4 = ee.Number(valB4);
        valB8 = ee.Number(valB8);
        valB11 = ee.Number(valB11);
        valB12 = ee.Number(valB12);

        // Sum these integer values.
        var sumOfChecks = valB3
            .add(valB4)
            .add(valB8)
            .add(valB11)
            .add(valB12);

        // If all 5 bands are present, the sum will be 5.
        // This comparison (eq()) results in an ee.Bool (true if all present, false otherwise).
        var isValid = sumOfChecks.eq(5);

        return image.set('isValid', isValid);
    })
    .filter(ee.Filter.eq('isValid', 1)); // Filter to keep only valid images

print('s2WithCloud after validImage filter:', s2WithCloud); // Debug print

// S2 cloud/shadow and probability masking function
function maskS2withCloudProb(image) {
    // Explicitly cast the input 'image' to an ee.Image for type safety.
    image = ee.Image(image);

    // Get the cloud probability image from the joined collection
    var cloudProbImage = ee.Image(image.get('cloud_mask'));
    // Handle cases where cloudProbImage might be null (e.g., if join failed for an image)
    var cloudProb = ee.Algorithms.If(cloudProbImage, cloudProbImage.select('probability'), ee.Image(0));
    cloudProb = ee.Image(cloudProb); // Cast the result of If to an Image

    // Create a cloud mask based on the cloud probability threshold
    var cloudMask = cloudProb.lt(cloudFilt);

    // Apply the cloud mask, scale bands, select relevant bands, and copy properties
    return image.updateMask(cloudMask).divide(10000)
        .select(['B3', 'B4', 'B8', 'B11', 'B12'])
        .copyProperties(image, ['system:time_start']);
}

// ========== MONTHLY TIME SERIES COMPOSITES FOR PRE/POST ====================
// Function to create monthly median composites from an image collection
function compositeMonthly(collection, start, end) {
    var startDate = ee.Date(start);
    var endDate = ee.Date(end);

    // Generate a sequence of months from start to end date
    // Correction: Removed .subtract(1) to include the last month fully.
    var monthsToProcess = ee.List.sequence(0, endDate.difference(startDate, 'month').round());

    var monthlyComposites = monthsToProcess.map(function(offset_val) {
        var offset = ee.Number(offset_val); // Cast to ee.Number for server-side operations
        var currentMonthStart = startDate.advance(offset, 'month');
        var currentMonthEnd = currentMonthStart.advance(1, 'month');

        var filtered = collection
            .filterDate(currentMonthStart, currentMonthEnd)
            .map(maskS2withCloudProb);

        // Crucial: Explicitly cast 'filtered' to ee.ImageCollection to ensure .median() is available.
        filtered = ee.ImageCollection(filtered);

        // Check the size of the filtered collection BEFORE computing median
        var count = filtered.size();

        var monthlyResult = ee.Algorithms.If(
            count.gt(0), // If there are images in the filtered collection
            filtered.median().set({
                'system:time_start': currentMonthStart.millis(),
                'year': currentMonthStart.get('year'),
                'month': currentMonthStart.get('month')
            }),
            // Else (if no images for the month), return an empty, dummy image.
            // This placeholder will be filtered out later.
            ee.Image([]).set('system:time_start', currentMonthStart.millis())
        );
        return monthlyResult;
    });

    // Ensure monthlyComposites is treated as an ee.List of ee.Images
    monthlyComposites = ee.List(monthlyComposites);

    // Convert the list of images (including empty placeholders) to an ImageCollection.
    var imgCollection = ee.ImageCollection(monthlyComposites);

    // Filter out the empty placeholder images (those with no actual bands)
    // by checking for the presence of a specific band (e.g., 'B3').
    return imgCollection.filter(ee.Filter.listContains('system:band_names', 'B3'))
                        .map(function(image) {
                            return image.clip(miami);
                        });
}

// Generate pre-event and post-event monthly composites
var preMonthly = compositeMonthly(s2WithCloud, preStart, preEnd);
var postMonthly = compositeMonthly(s2WithCloud, postStart, postEnd);

print('preMonthly ImageCollection:', preMonthly); // Debug print
print('postMonthly ImageCollection:', postMonthly); // Debug print

// FINAL GUARD FOR EMPTY COLLECTIONS BEFORE MEDIAN
// This ensures that even if monthly collections are empty, 'pre' and 'post'
// are still valid images with expected band structure (dummy bands if empty).
var pre = ee.Image(ee.Algorithms.If(
    preMonthly.size().gt(0),
    preMonthly.median().clip(miami), // Ensure final median is also clipped
    // If preMonthly is empty, return a dummy image with expected bands (zeros)
    ee.Image().addBands(ee.Image.constant(0).rename('B3'))
              .addBands(ee.Image.constant(0).rename('B4'))
              .addBands(ee.Image.constant(0).rename('B8'))
              .addBands(ee.Image.constant(0).rename('B11'))
              .addBands(ee.Image.constant(0).rename('B12'))
));

var post = ee.Image(ee.Algorithms.If(
    postMonthly.size().gt(0),
    postMonthly.median().clip(miami), // Ensure final median is also clipped
    // If postMonthly is empty, return a dummy image with expected bands (zeros)
    ee.Image().addBands(ee.Image.constant(0).rename('B3'))
              .addBands(ee.Image.constant(0).rename('B4'))
              .addBands(ee.Image.constant(0).rename('B8'))
              .addBands(ee.Image.constant(0).rename('B11'))
              .addBands(ee.Image.constant(0).rename('B12'))
));

print('Pre-event composite image (pre):', pre); // Debug print
print('Post-event composite image (post):', post); // Debug print

// =============== WATER INDICES: NDWI, MNDWI, AWEI ==========================
// Calculate Normalized Difference Water Index (NDWI)
var ndwiPre = pre.normalizedDifference(['B3', 'B8']).rename('ndwi_pre');
var ndwiPost = post.normalizedDifference(['B3', 'B8']).rename('ndwi_post');

// Calculate Modified Normalized Difference Water Index (MNDWI)
var mndwiPre = pre.normalizedDifference(['B3', 'B11']).rename('mndwi_pre');
var mndwiPost = post.normalizedDifference(['B3', 'B11']).rename('mndwi_post');

// Calculate Automated Water Extraction Index (AWEI)
var aweiPre = pre.expression(
    '4*(GREEN-SWIR1)-(0.25*NIR+2.75*SWIR2)', {
        'GREEN': pre.select('B3'),
        'SWIR1': pre.select('B11'),
        'NIR': pre.select('B8'),
        'SWIR2': pre.select('B12')
    }).rename('awei_pre');
var aweiPost = post.expression(
    '4*(GREEN-SWIR1)-(0.25*NIR+2.75*SWIR2)', {
        'GREEN': post.select('B3'),
        'SWIR1': post.select('B11'),
        'NIR': post.select('B8'),
        'SWIR2': post.select('B12')
    }).rename('awei_post');

// ----------- DYNAMIC/SCENE THRESHOLDS ---------------
// These thresholds should ideally be determined dynamically or locally
var tNDWI = 0.30;
var tMNDWI = 0.0;
var tAWEI = 0.0;

// Apply thresholds to create binary water masks for each index
var waterPreNDWI = ndwiPre.gt(tNDWI);
var waterPostNDWI = ndwiPost.gt(tNDWI);
var waterPreMNDWI = mndwiPre.gt(tMNDWI);
var waterPostMNDWI = mndwiPost.gt(tMNDWI);
var waterPreAWEI = aweiPre.gt(tAWEI);
var waterPostAWEI = aweiPost.gt(tAWEI);

// Create a consensus water mask (at least 2 out of 3 indices agree)
var waterPreSum = waterPreNDWI.add(waterPreMNDWI).add(waterPreAWEI);
var waterPostSum = waterPostNDWI.add(waterPostMNDWI).add(waterPostAWEI);
var waterPreConsensus = waterPreSum.gte(2);
var waterPostConsensus = waterPostSum.gte(2);

// ============ SAR FLOOD MASK (SENTINEL-1) ==================================
// Load Sentinel-1 Ground Range Detected (GRD) data
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(miami)
    .filterDate(postStart, postEnd)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) // Filter for VV polarization
    .filter(ee.Filter.eq('instrumentMode', 'IW')) // Filter for Interferometric Wide Swath mode
    .select(['VV']); // Select the VV band

print('s1 collection:', s1); // Debug print

// Guard against empty S1 collection
var s1Composite = ee.Image(ee.Algorithms.If(
    s1.size().gt(0),
    s1.median().clip(miami),
    ee.Image.constant(0).rename('VV') // Dummy VV band if no S1 data
));

var vvThreshold = -17; // Threshold for water detection in VV band (dB)
var s1Water = s1Composite.select('VV').lt(vvThreshold); // Binary water mask from S1

// ======== FLOOD CHANGE DETECTION, POST-PROCESSING, URBAN/CLEANUP ===========
// Identify potential flood areas: Post-event water where pre-event was not water
var floodMap = waterPostConsensus.and(waterPreConsensus.not());

// Combine optical flood map with SAR-derived water
var combinedFlood = floodMap.or(s1Water);

// Categorize water change:
var persistentWater = waterPostConsensus.and(waterPreConsensus); // Water in both periods
var newWater = waterPostConsensus.and(waterPreConsensus.not()); // New water (flood)
var lostWater = waterPreConsensus.and(waterPostConsensus.not()); // Water lost

// Create a change map with different values for each category
var changeMap = persistentWater.multiply(3) // Persistent water = 3
    .add(newWater.multiply(2)) // New water = 2
    .add(lostWater.multiply(1)); // Lost water = 1

// Post-processing: Remove small isolated pixels (noise)
var sizeThreshold = 10; // Minimum connected pixel count to be considered valid
var connectedPixels = combinedFlood.connectedPixelCount(sizeThreshold, false);
var cleanedFlood = combinedFlood.updateMask(connectedPixels.gte(sizeThreshold));

// Mask out urban areas from the flood map (assuming urban areas are not floods)
var urbanMask = ee.Image('ESA/WorldCover/v100/2020').eq(50); // WorldCover urban class is 50
var floodUrbanMasked = cleanedFlood.updateMask(urbanMask.not());

// ============ REMOVE PERMANENT WATER (JRC GLOBAL SURFACE WATER) ============
// Load JRC Global Surface Water dataset
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater');
var seasonality = gsw.select('seasonality'); // Select seasonality band
// Pixels with seasonality >= 10 are considered permanent water
var permanentWaterMask = seasonality.gte(10);
var notPermanentWater = permanentWaterMask.not(); // Mask for non-permanent water

// Final flood map by removing permanent water bodies
var floodFinal = floodUrbanMasked.updateMask(notPermanentWater);

// ============ ADDITIONAL ML PREDICTORS =====================================
// Add auxiliary data for potential machine learning classification
var elevation = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(miami);
var slope = ee.Terrain.slope(elevation);
var landcover = ee.Image('ESA/WorldCover/v100/2020').select('Map').clip(miami);

// ============ METADATA, SAMPLES, AND EXPORT ===============================
// Visualize results on the map
Map.addLayer(post, {
    bands: ['B4', 'B3', 'B8'],
    min: 0,
    max: 0.3
}, 'RGB Post-event');
Map.addLayer(floodFinal, {
    palette: ['red']
}, 'Flood Mask, No Permanent Water');
Map.addLayer(changeMap, {
    min: 0,
    max: 3,
    palette: ['black', 'blue', 'green', 'orange']
}, 'Water Change (0: No change, 1: Lost, 2: New, 3: Persistent)');

// Export the final flood mask to Google Drive
Export.image.toDrive({
    image: floodFinal,
    description: 'flood_mask_final_' + dateString,
    folder: 'GEE_Flood',
    fileNamePrefix: 'flood_mask_final_' + dateString,
    region: miami,
    scale: scale,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
});

// Prepare stratified samples for machine learning (if applicable)
var samples = floodFinal.rename('flood') // Renamed the flood mask band for clarity
    .addBands(ndwiPost)
    .addBands(mndwiPost)
    .addBands(aweiPost)
    .addBands(s1Water)
    .addBands(elevation)
    .addBands(slope)
    .addBands(landcover)
    .addBands(seasonality.rename('jrc_seasonality'))
    .stratifiedSample({
        numPoints: 3000,
        classBand: 'flood', // Use the renamed band
        region: miami,
        scale: scale,
        seed: 42,
        geometries: true
    });

// Export the samples to Google Drive as a CSV
Export.table.toDrive({
    collection: samples,
    description: 'FloodSamples_' + dateString,
    folder: 'GEE_Flood',
    fileNamePrefix: 'FloodSamples_' + dateString,
    fileFormat: 'CSV'
});

// Export the change map to Google Drive
Export.image.toDrive({
    image: changeMap,
    description: 'changeMap_' + dateString,
    folder: 'GEE_Flood',
    fileNamePrefix: 'changeMap_' + dateString,
    region: miami,
    scale: scale,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF'
});


// ============ EXPORT PREDICTOR RASTERS FOR OFFLINE ML PREDICTION ===========
// 1. Rename bands and EXPLICITLY CAST TO FLOAT32 for data type consistency.
var ndwiPostRenamed = ndwiPost.rename('ndwi_post').toFloat(); // Cast to Float32
var mndwiPostRenamed = mndwiPost.rename('mndwi_post').toFloat(); // Cast to Float32
var aweiPostRenamed = aweiPost.rename('awei_post').toFloat(); // Cast to Float32
// The s1Composite already selects 'VV' and is clipped, so we just rename that band.
var s1VVRenamed = s1Composite.select('VV').rename('VV').toFloat(); // Cast to Float32

var elevationRenamed = elevation.rename('elevation').toFloat(); // Cast to Float32
var slopeRenamed = slope.rename('slope').toFloat(); // Cast to Float32
// WorldCover 'Map' band is Uint8. Cast to Float32.
var landcoverRenamed = landcover.rename('Map').toFloat(); // Cast to Float32
// JRC 'seasonality' band is Uint8. Cast to Float32.
var jrcSeasonalityRenamed = seasonality.rename('jrc_seasonality').toFloat(); // Cast to Float32

// 2. Combine all predictor bands into a single image
var predictorImage = ndwiPostRenamed
    .addBands(mndwiPostRenamed)
    .addBands(aweiPostRenamed)
    .addBands(s1VVRenamed)
    .addBands(elevationRenamed)
    .addBands(slopeRenamed)
    .addBands(landcoverRenamed)
    .addBands(jrcSeasonalityRenamed);

// Print the combined image to inspect its bands (optional)
print('Combined Predictor Image for Export (for ML):', predictorImage);

// 3. Export the combined predictor image to Google Drive as a GeoTIFF
Export.image.toDrive({
    image: predictorImage,
    description: 'predictor_bands_for_ml_' + dateString, // Unique description for the task
    folder: 'GEE_Flood_Predictors', // A new folder in your Drive for these GeoTIFFs
    fileNamePrefix: 'predictors_' + dateString, // Prefix for the output GeoTIFF file
    region: miami, // Your Area of Interest
    scale: scale, // Resolution (defined in USER SETTINGS)
    maxPixels: 1e9, // Maximum number of pixels to export (adjust if needed for larger AOIs)
    fileFormat: 'GeoTIFF',
    // Specify 'crs' if you need a specific projection, e.g., 'EPSG:4326'
    // crs: 'EPSG:4326'
});

print('Predictor image export task created.');
print('Script completed successfully!');
