#Helper module for data processing, including
#1. API function for AIS data (ice concentration data downloaded over web)
#2. Helper function to process ice raster
#3. Function to manually calculate asympotic local/global spatial autocorrelation (avoids expensive permutation testing)
#4. Function to classify pixels into lisa clusters
import requests
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import os
import io
import numpy as np
import time
import xarray as xr
import rioxarray
from scipy.ndimage import convolve
from scipy import stats


def retrieve_ais(bbox, start_year, stop_year, type='tif', step=1, width=5952, height=5201, crs=3995):
    '''
    Function that retrieves AIS data for a given year range from API using bbox

    Example link:
    https://gmtds.maplarge.com/ogc/ais:density/wms?REQUEST=GetMap&LAYERS=ais:density&STYLES=&FORMAT=image/png&TRANSPARENT=true
    &SERVICE=WMS&VERSION=1.3.0&WIDTH=1600&HEIGHT=693&CRS=EPSG:3995
    &BBOX=-10115103.215710418,-4227137.379356397,9208953.375633113,3393787.4388547074&TIME=2023-04-01T00:00:00Z
    &CQL_FILTER="category_column='All' AND category='All'"

    :param bbox: bounding box for query
    :param start_year: year to begin querying by
    :param stop_year: year to query to
    :param type: file type desired, png or tif
    :param step: step for querying (optional)
    :param width: width of image to be returned, default matches resolution
    :param height: height of image to return, default matches resolution
    :param crs: crs to projet into, default arctic

    Saves AIS directly to appropriate data folder. 
    '''
    #Establish global calibration colors known from online legend.
    #Data returns in PNG format, need to convert from RGB to actual AIS data. These colors are associated with subsequent AIS density (ship hours per km^2)
    CALIBRATION = [
        ('#cadeb9', 0.001),
        ('#c4d3a2', 0.05),
        ('#fef79a', 0.1),
        ('#fed375', 0.2),
        ('#f3a26f', 0.5),
        ('#fa7330', 2),
        ('#fb4e2b', 5),
        ('#d41a26', 10),
        ('#9c0026', 20),
        ('#61001f', 100),
        ('#3d0215', 2000)
    ]

    def hex_to_rgb(hex):
        '''
        Helper function that converts hex code to RGB

        :param hex: hex code
        '''
        #Strip #, use formula to return 3 value point for R, G, B
        h = hex.strip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    #Get calibration colors and values from dict.
    cal_colors = np.array([hex_to_rgb(h) for h, _ in CALIBRATION], dtype=np.float32)
    cal_values = np.array([v for _, v in CALIBRATION], dtype=np.float32)

    def decode(png_array, threshold=25.):
        '''
        Helper function to translate APi returned PNG to actual AIS data geotiff

        :param png_array: 3D array with RGB information
        :param threshold: threshold to clean up distorted pixels that are far from known color ramp
        '''
        #Get height and width of png, pixel values, and transparent pixels (RGBA format)
        h, w, _ = png_array.shape
        pixels = png_array[..., :3].reshape(-1, 3).astype(np.float32)
        alpha = png_array[..., 3].reshape(-1)
        
        # Initialize output arrays, find minimum distance from line segment and final assigned values (AIS)
        min_distances = np.full(pixels.shape[0], np.inf, dtype=np.float32)
        final_values = np.full(pixels.shape[0], np.nan, dtype=np.float32)

        # Vectorized segment-by-segment calculation with np
        for i in range(len(cal_colors) - 1):
            #Get calibration values and colors
            A, B = cal_colors[i], cal_colors[i+1]
            v1, v2 = cal_values[i], cal_values[i+1]
            
            #Calculate vector from difference and dot product, skip if difference is 0 (avoids errors)
            vec = B - A
            mag_sq = np.dot(vec, vec)
            if mag_sq == 0: continue
            
            #Project pixels onto segment (t is 0 to 1)
            t = np.sum((pixels - A) * vec, axis=1) / mag_sq
            t = np.clip(t, 0, 1)
            
            #Find distance to the closest point on this segment
            #Euclidian distance in 3D space
            closest_pts = A + t[:, np.newaxis] * vec
            dist = np.linalg.norm(pixels - closest_pts, axis=1)
            
            #Update pixels where this segment is the closest found so far, otherwise keep same
            mask = dist < min_distances
            min_distances[mask] = dist[mask]
            
            # Piecewise Exponential Interpolation
            # Formula: exp(log(v1) + t * (log(v2) - log(v1)))
                #Finds logarithmic center not linear center, given AIS legend scale
            log_v1, log_v2 = np.log(v1), np.log(v2)
            interp_vals = np.exp(log_v1 + t[mask] * (log_v2 - log_v1))
            final_values[mask] = interp_vals

        # Clean up, mask pixels that are too far from the ramp or transparent (will be recalculated on next iteration)
        invalid_mask = (min_distances > threshold) | (alpha == 0)
        final_values[invalid_mask] = np.nan
        
        #Return reshaped final values for each pixel
        return final_values.reshape(h, w)
    
    #Base for requests
    base = 'https://gmtds.maplarge.com/ogc/ais:density/wms'

    #loop through specified year
    for year in range(start_year, stop_year+1, step):
        time.sleep(1)
        #Generate correct timestamp for September
        timestamp = f'{year}-09-01T00:00:00Z'
        
        #Setup param dictionary for API for appropriate get request
        params = {
                'REQUEST': 'GetMap',
                'LAYERS': 'ais:density',
                'STYLES': '',
                'FORMAT': 'image/png',
                'TRANSPARENT': 'true',
                'SERVICE': 'WMS',
                'VERSION': '1.3.0',
                'WIDTH': width,
                'HEIGHT': height,
                'CRS': f'EPSG:{crs}',
                'BBOX': ','.join(map(str, bbox)),
                'TIME': timestamp,
                'CQL_FILTER': "category_column='All' AND category='All'"
            }

        #Set correct file format for saving
        if type == 'tif':
            filename = f'arctic_maritime_{year}.tif'
        else:   
            filename = f'arctic_maritime_{year}.png'

        #Send get request
        print(f'Requesting data for {year}:')
        try:
            response = requests.get(base, params=params, timeout=40)
            if response.status_code==200:
                
                if type == 'tif':
                    #Open png with PIL, use decode function.
                    png = Image.open(io.BytesIO(response.content)).convert('RGBA')
                    print('Decoding RGB values to AIS Traffic')
                    data = decode(np.array(png))
                    print("Min:", np.nanmin(data))
                    print("Max:", np.nanmax(data))
                    print("Mean:", np.nanmean(data))
                    print("NaN %:", np.isnan(data).mean() * 100)
                    #Transform array to correct bounds 
                    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

                    #Open with rasterio, use transformation and appropriate crs to reproject decoded data into appropriate tif format
                    #with rasterio.Env(GDAL_PAM_ENABLED="NO"):
                    with rasterio.open(
                        f'data/ais_traffic/{filename}', 
                        'w', 
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype='float32',
                        crs=f'EPSG:{crs}',
                        transform=transform
                    ) as file:
                        file.write(data, 1)
                #If png, just write what is retrieved
                else:
                    with open(f'data/ais_traffic/{filename}', 'wb') as file:
                        file.write(response.content)
                print(f'{year} successfully downloaded.')
            #{Print response code}
            else:
                print(f'Error: {response.status_code}')
        #Print exception
        except Exception as e:
            print(f'Failed: {e}')


def process_ice_raster(path):
    '''
    Helper function to simplify ice raster processing

    :param path: path to ice concentration raster
    '''
    #Get resolution and bounds from shipping image
    maritime = 'data/ais_traffic/arctic_maritime'
    with rasterio.open(f'{maritime}_{2012}.tif') as src:
                ship_res = src.res 
                bounds = src.bounds
    #Open path with xarray package to read .nc file format 
    ds = xr.open_dataset(path)

    #Retrieve appropriate attribute, convert to km to m, set spatial dimensions 
    temp = ds['ice_conc'].isel(time=0)
    temp['xc'] = temp['xc'] * 1000
    temp['yc'] = temp['yc'] * 1000
    temp = temp.rio.set_spatial_dims(x_dim='xc', y_dim='yc')
    temp.rio.write_crs('EPSG:6931', inplace=True)

    #Reproject to arctic crs
    temp_projected = temp.rio.reproject('EPSG:3995', resolution=ship_res)
    temp_clipped = temp_projected.rio.clip_box(minx=bounds.left,
                                                miny=bounds.bottom,
                                                maxx=bounds.right,
                                                maxy=bounds.top)
    
    #Remove unecessary attribute dimension, mask 0 values as NA to avoid clutter, return
    img_arr = temp_clipped.values.squeeze()
    img_arr = np.where(img_arr <= 0, np.nan, img_arr)
    return img_arr

def spatial_autocorrelation(raster):
    '''
    Function to determine global and local spatial autocorrelation for a given raster. 

    :param raster: input raster. 
    '''
    #Mask out NA values for calculation
    mask = ~np.isnan(raster)
    n = np.sum(mask)
    data = np.nan_to_num(raster, nan=0.0)

    #Extract means and standard devation to get z score for data
    means = data[mask].mean()
    std = data[mask].std()
    z = (data - means)/std

    z[~mask]=0

    #Establish kernel for local spatial autocorrelation, 'queens distance' 8 surrounding cells
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    #Spatial weights
    kernel_norm = kernel/kernel.sum()

    #Calculate spatial lag
    spatial_lag = convolve(z, kernel_norm, mode='constant', cval=0.0)

    #Use spatial lag to calculate local Moran's I
    local_morans = z * spatial_lag

    #Global I: 
    s2 = np.sum(z[mask]**2)
    global_morans = np.sum(local_morans[mask])/ s2

    #mask local morans for NAs:
    local_morans = np.where(mask, local_morans, np.nan)

    #Calculating p-values
    w_i2 = np.sum(kernel_norm**2)
    b2 = (n * np.sum(z[mask]**4)) / (s2**2)
    local_variance = (w_i2 * (n - b2)) / (n - 1)
    e_i = (w_i2 * (n-b2))/(n-1)

    #Calculate local z and p values based on above variance and epxectation calculation (for each pixel), using NP
    local_z = (local_morans - e_i) / np.sqrt(local_variance)
    local_p = 2 * (1 - stats.norm.cdf(np.abs(local_z)))

    #Mask NA values
    local_p = np.where(mask, local_p, np.nan)
    
    #Determine lisa clusters and return values
    clusters = lisa_clusters(z, spatial_lag)

    return global_morans, local_morans, local_p, clusters


def lisa_clusters(z, spatial_lag, p_values=None, alpha=0.01):
    '''
    Function to classify pixels into local indication of spatial autocorrelation (LISA) clusters

    :param z: local z score (Moran's I statistic)
    :param spatial_lag: spatial lag for each pixel
    '''

    #Define empty array for hotspot, coldspot categories
    # 0 (not significant, optional), 1: HH, 2: LL, 3: LH, 4: HL
    lisa_clusters = np.zeros(z.shape, dtype=int)

    #Define quadrants based on z and spatial lag
    hh = (z>0) & (spatial_lag > 0)
    ll = (z < 0) & (spatial_lag < 0)
    lh = (z < 0) & (spatial_lag > 0)
    hl = (z > 0) & (spatial_lag < 0)

    lisa_clusters[hh] = 1
    lisa_clusters[ll] = 2
    lisa_clusters[lh] = 3
    lisa_clusters[hl] = 4

    if p_values is not None:
        lisa_clusters[p_values > alpha] = 0
    
    #Return classified clusters
    return lisa_clusters