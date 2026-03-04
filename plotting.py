#Helper file to reduce clutter in main ipynb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np

import rasterio as rio
import shapely
import contextily as cx

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import img_tiles
import xarray as xr
import rioxarray


def dynamic_plot(years, bounds, interval, shipping=True, ice=False, save=False, filename='series.gif'):
    
    # with rio.open('data/ais_traffic/arctic_maritime_2020.tif') as src:
    #     img_arr = src.read(1)

    maritime = 'data/ais_traffic/arctic_maritime'
    ice_con = 'data/ice_concentration/ice_conc_nh_ease2-250_cdr-v3p1_'
    
    proj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=71)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    norm = LogNorm(vmin=0.0001, vmax=2000)

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    
    
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#717171", edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#0e0e0e', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#333333', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#333333', linewidth=0.3, zorder=2)



    if ice:
        with rio.open(f'{maritime}_{years[0]}.tif') as src:
            ship_res = src.res 
        ds = xr.open_dataset(f'{ice_con}{years[0]}09.nc')

        temp = ds['ice_conc'].isel(time=0)
        temp['xc'] = temp['xc'] * 1000
        temp['yc'] = temp['yc'] * 1000
        temp = temp.rio.set_spatial_dims(x_dim='xc', y_dim='yc')
        temp.rio.write_crs('EPSG:6931', inplace=True)

        temp_projected = temp.rio.reproject('EPSG:3995', resolution=ship_res)
        temp_clipped = temp_projected.rio.clip_box(minx=bounds.left,
                                                   miny=bounds.bottom,
                                                   maxx=bounds.right,
                                                   maxy=bounds.top)
        img_arr_ice = temp_clipped.values.squeeze()
        img_arr_ice = np.where(img_arr_ice <= 0, np.nan, img_arr_ice)

        im_ice = ax.imshow(img_arr_ice, cmap='Blues_r', extent=extent, norm=plt.Normalize(vmin=0,vmax=100), zorder=3)

    if shipping:
        with rio.open(f'{maritime}_{years[0]}.tif') as src:
            img_arr_mari = src.read(1)

        im_mari = ax.imshow(img_arr_mari, cmap='Reds_r', extent=extent, norm=norm, zorder=4)

    fig.canvas.draw()
    ax.set_position([0.02, 0.02, 0.84, 0.96])  # [left, bottom, width, height]

    #Shipping cbar
    cbar_ax = fig.add_axes([0.87, 0.08, 0.025, 0.40])
    cbar_ticks = [0.001, 0.05, 1, 2, 5, 10, 20, 100, 2000]
    cbar = plt.colorbar(im_mari, cax=cbar_ax, ticks=cbar_ticks, shrink=0.7)
    cbar.ax.set_yticklabels([str(i) for i in cbar_ticks])
    cbar.set_label('AIS Density', fontsize=12)

    #Ice cbar
    ice_cbar_ax = fig.add_axes([0.87, 0.52, 0.025, 0.40])
    ice_cbar = fig.colorbar(im_ice, cax=ice_cbar_ax)
    ice_cbar.set_label('Ice Concentration (%)', fontsize=12)

    title = ax.set_title('', fontsize=14)

    

    def update(frame):
        artists = [title]

        if ice:
            ds = xr.open_dataset(f'{ice_con}{frame}09.nc')
            temp = ds['ice_conc'].isel(time=0)
            temp['xc'] = temp['xc'] * 1000
            temp['yc'] = temp['yc'] * 1000
            temp = temp.rio.set_spatial_dims(x_dim='xc', y_dim='yc')
            temp.rio.write_crs('EPSG:6931', inplace=True)

            temp_projected = temp.rio.reproject('EPSG:3995', resolution=ship_res)
            temp_clipped = temp_projected.rio.clip_box(minx=bounds.left,
                                                    miny=bounds.bottom,
                                                    maxx=bounds.right,
                                                    maxy=bounds.top)
            img_arr_ice = temp_clipped.values.squeeze()
            img_arr_ice = np.where(img_arr_ice <= 0, np.nan, img_arr_ice)


            im_ice.set_data(img_arr_ice)

            title.set_text(f'Arctic Maritime Traffic Density (AIS Data) - {frame}')
                
        if shipping:
            with rio.open(f'{maritime}_{frame}.tif') as src:
                img_arr_mari = src.read(1)

                im_mari.set_data(img_arr_mari)

                title.set_text(f'Arctic Maritime Traffic Density (AIS Data) - {frame}')
        
        return artists
    
    anim = FuncAnimation(fig, update, frames=years, interval=interval)

    if save:
        anim.save(f'figures/{filename}', writer='pillow')

    return anim