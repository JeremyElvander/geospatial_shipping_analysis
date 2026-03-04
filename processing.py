import requests
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import os
import io
import numpy as np
import time

def retrieve_ice_concentration(start_year, stop_year, type='tif', step=1, crs=3995):
    

    params = {
        "dataset_id": "EO:ECMWF:DAT:SATELLITE_SEA_ICE_CONCENTRATION",
        "variable": "all",
        "sensor": "ssmis",
        "region": [
            "northern_hemisphere"
        ],
        "cdr_type": [
            "cdr"
        ],
        "temporal_aggregation": "monthly",
        "year": [
            "2011",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
            "2018",
            "2019",
            "2020"
        ],
        "month": [
            "09"
        ],
        "version": "3_1",
        "itemsPerPage": 200,
        "startIndex": 0
        }
    raise NotImplementedError

def retrieve_ais(bbox, start_year, stop_year, type='tif', step=1, width=5952, height=5201, crs=3995):
    '''

    Example link:
    https://gmtds.maplarge.com/ogc/ais:density/wms?REQUEST=GetMap&LAYERS=ais:density&STYLES=&FORMAT=image/png&TRANSPARENT=true
    &SERVICE=WMS&VERSION=1.3.0&WIDTH=1600&HEIGHT=693&CRS=EPSG:3995
    &BBOX=-10115103.215710418,-4227137.379356397,9208953.375633113,3393787.4388547074&TIME=2023-04-01T00:00:00Z
    &CQL_FILTER="category_column='All' AND category='All'"
    '''
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
        '''
        h = hex.strip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    cal_colors = np.array([hex_to_rgb(h) for h, _ in CALIBRATION], dtype=np.float32)
    cal_values = np.array([v for _, v in CALIBRATION], dtype=np.float32)

    def decode(png_array, threshold=25.):
        '''
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
            A, B = cal_colors[i], cal_colors[i+1]
            v1, v2 = cal_values[i], cal_values[i+1]
            
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
            
            #Update pixels where this segment is the closest found so far
            mask = dist < min_distances
            min_distances[mask] = dist[mask]
            
            # Piecewise Exponential Interpolation
            # Formula: exp(log(v1) + t * (log(v2) - log(v1)))
                #Finds logarithmic center not linear center, given AIS legend scale
            log_v1, log_v2 = np.log(v1), np.log(v2)
            interp_vals = np.exp(log_v1 + t[mask] * (log_v2 - log_v1))
            final_values[mask] = interp_vals

        # Clean up, mask pixels that are too far from the ramp or transparent
        invalid_mask = (min_distances > threshold) | (alpha == 0)
        final_values[invalid_mask] = np.nan
        
        return final_values.reshape(h, w)
    
    base = 'https://gmtds.maplarge.com/ogc/ais:density/wms'

    for year in range(start_year, stop_year+1, step):
        time.sleep(1)
        timestamp = f'{year}-09-01T00:00:00Z'
        
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

        if type == 'tif':
            filename = f'arctic_maritime_{year}.tif'
        else:   
            filename = f'arctic_maritime_{year}.png'


        print(f'Requesting data for {year}:')
        try:
            response = requests.get(base, params=params, timeout=40)
            if response.status_code==200:
                
                if type == 'tif':

                    png = Image.open(io.BytesIO(response.content)).convert('RGBA')
                    print('Decoding RGB values to AIS Traffic')
                    data = decode(np.array(png))
                    print("Min:", np.nanmin(data))
                    print("Max:", np.nanmax(data))
                    print("Mean:", np.nanmean(data))
                    print("NaN %:", np.isnan(data).mean() * 100)

                    transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)

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
                else:
                    with open(f'data/ais_traffic/{filename}', 'wb') as file:
                        file.write(response.content)
                print(f'{year} successfully downloaded.')
            else:
                print(f'Error: {response.status_code}')
        except Exception as e:
            print(f'Failed: {e}')


# 


# base = 'https://gmtds.maplarge.com/ogc/ais:density/wms'
# html = requests.get(base, params={'SERVICE':'WMS', 'REQUEST':'GetCapabilities'})

# from pprint import pprint
# pprint(html.content)

# CALIBRATION = [
#         # ('#cadeb9', 0),
#         ('#c4d3a2', 0.05),
#         ('#fef79a', 0.1),
#         ('#fed375', 0.2),
#         ('#f3a26f', 0.5),
#         ('#fa7330', 2),
#         ('#fb4e2b', 5),
#         ('#d41a26', 10),
#         ('#9c0026', 20),
#         ('#61001f', 100)
#     ]
# def hex_to_rgb(hex):
#     h = hex.strip('#')
#     return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# # for tup in CALIBRATION:
# #     print(f'{hex_to_rgb(tup[0])}')


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import ListedColormap

# Define the points (x, y, z)
# points = [(196, 211, 162), (254, 247, 154), (254, 211, 117), (243, 162, 111), 
#           (250, 115, 48), (251, 78, 43), (212, 26, 38), (156, 0, 38), (97, 0, 31)]

# # Extract X, Y, Z coordinates
# x = [p[0] for p in points]
# y = [p[1] for p in points]
# z = [p[2] for p in points]

# # Create 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c=z, cmap=ListedColormap([tup[0] for tup in CALIBRATION]), s=50) # c=z colors by depth

# # Set labels
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.plot(x, y, z, color='gray', linestyle='-', alpha=0.5)
# plt.show()




                    # img_array = np.array(png)
                    # pixels = img_array.reshape(-1, 4)
                    # # Keep only pixels where Alpha > 0 and it's not pure black
                    # mask = (pixels[:, 3] > 0) & (np.sum(pixels[:, :3], axis=1) > 0)
                    # active_pixels = pixels[mask]

                    # px_x = active_pixels[:, 0]
                    # px_y = active_pixels[:, 1]
                    # px_z = active_pixels[:, 2]

                    # import matplotlib.pyplot as plt
                    # from mpl_toolkits.mplot3d import Axes3D
                    # from matplotlib.colors import ListedColormap

                    # # Define the points (x, y, z)
                    # points = [(196, 211, 162), (254, 247, 154), (254, 211, 117), (243, 162, 111), 
                    #         (250, 115, 48), (251, 78, 43), (212, 26, 38), (156, 0, 38), (97, 0, 31)]

                    # # Extract X, Y, Z coordinates
                    # x = [p[0] for p in points]
                    # y = [p[1] for p in points]
                    # z = [p[2] for p in points]

                    # # Create 3D scatter plot
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(px_x, px_y, px_z, c='black', s=1, alpha=0.2, label='Image Pixels')
                    # ax.scatter(x, y, z, c=z, cmap=ListedColormap([tup[0] for tup in CALIBRATION]), s=50) # c=z colors by depth

                    # # Set labels
                    # ax.set_xlabel('X Label')
                    # ax.set_ylabel('Y Label')
                    # ax.set_zlabel('Z Label')
                    # ax.plot(x, y, z, color='blue', linestyle='-', alpha=0.5)
                    # plt.show()