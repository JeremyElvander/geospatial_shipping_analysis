#Helper file to reduce clutter in main ipynb
#Import dependencies
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, ListedColormap
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

def static_plot(shipping_raster=None, ice_raster = None, shipping=False, ice=False):
    '''
    Function that produces a static plot for shipping/ice concentration in the artic.

    :param shipping_raster: raster file for AIS data
    :param ice_raster: raster file for ice concentration
    :param shipping: Boolean to include shipping
    :param ice: boolean to include ice

    :return: figure and ax matplotlib objects
    '''
    #Retrieve shipping bounds (ice concentration is much larger raster)
    with rio.open('data/ais_traffic/arctic_maritime_2020.tif') as src:
        bounds = src.bounds

    #Retrieve Cartopy projection for arctic, extent
    proj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=71)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    #Create fig and ax objects and set ax limits, provide CCRS projection for cartopy features
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)

    #Add relevant features for basemap manually (ocean, land, coastline, borders)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#717171", edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#0e0e0e', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#333333', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#333333', linewidth=0.3, zorder=2)

    #If ice raster provided
    if ice:
        #Add ice raster, custom normalizer for cbar (percentage)
        im_ice = ax.imshow(ice_raster, cmap='Blues_r', extent=extent, norm=plt.Normalize(vmin=0,vmax=100), zorder=3)
        #Ice cbar
        #Reposition ice cbar if shipping cbar is also present
        if shipping:
            fig.canvas.draw()
            #Set position manually
            ax.set_position([0.02, 0.02, 0.84, 0.96]) 

            ice_cbar_ax = fig.add_axes([0.87, 0.52, 0.025, 0.40])
            ice_cbar = fig.colorbar(im_ice, cax=ice_cbar_ax)
        #Otherwise use default extent for ice CBAR
        else:
            ice_cbar = fig.colorbar(im_ice, ax=ax, shrink=0.7)
        ice_cbar.set_label('Ice Concentration (%)', fontsize=12)

    #If shipping raster is present
    if shipping:
        #Create log scale for cbar based on GMTDS scale
        norm = LogNorm(vmin=0.0001, vmax=2000)
        #Add shipping image
        im_shipping = ax.imshow(shipping_raster, cmap='Reds_r', extent=extent, norm=norm, zorder=3)

        #Shipping cbar with custom tick placement
        cbar_ticks = [0.001, 0.05, 1, 2, 5, 10, 20, 100, 2000]
        #Adjust cbar if ice is present otherwise use default settings
        if ice:
            cbar_ax = fig.add_axes([0.87, 0.08, 0.025, 0.40])
            cbar = plt.colorbar(im_shipping, cax=cbar_ax, ticks=cbar_ticks, shrink=0.7)
        else:
            cbar = plt.colorbar(im_shipping, ax=ax, ticks=cbar_ticks, shrink=0.7)
        cbar.ax.set_yticklabels([str(i) for i in cbar_ticks])
        cbar.set_label('AIS Density', fontsize=12)
    #Return figure and ax
    return fig, ax



def dynamic_plot(years, bounds, interval, shipping=True, ice=False, save=False, filename='series.gif'):
    '''
    Function to create (and optionally save) gif of arctic shipping/ice concentration. 

    :param years: the range of years to animate for (from downloaded rasters, 2011-2020)
    :param bounds: the bounds to crop rasters to
    :param interval: The speed at which to animate the gif
    :param shipping: boolean to include shipping information
    :param ice: boolean to include ice information
    :param save: boolean to save figure
    :param filename: optional filename if save is True

    :return anim: matplotlib animation function object
    '''
    #Get file extensions for both shipping and ice
    maritime = 'data/ais_traffic/arctic_maritime'
    ice_con = 'data/ice_concentration/ice_conc_nh_ease2-250_cdr-v3p1_'
    
    #Retrieve Cartopy arcitc projection, bounds, and appropriate cbar scale for shipping data
    proj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=71)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    norm = LogNorm(vmin=0.0001, vmax=2000)

    #Generate figure and ax objects and set ax limits, give fig cartopy projection for spatial data
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj})
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    
    #Manually add basemap features ocean, land, coastline, and borders
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#717171", edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#0e0e0e', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#333333', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#333333', linewidth=0.3, zorder=2)

    #If ice is selected, add base to map
    if ice:
        #Open shipping to get correct resolution
        with rio.open(f'{maritime}_{years[0]}.tif') as src:
            ship_res = src.res 
        #Use xarray to open .nc ice file
        ds = xr.open_dataset(f'{ice_con}{years[0]}09.nc')

        #Retrieve appropriate attribute (ice concentration), convert from km to m
        temp = ds['ice_conc'].isel(time=0)
        temp['xc'] = temp['xc'] * 1000
        temp['yc'] = temp['yc'] * 1000
        #Set spatial dimensions
        temp = temp.rio.set_spatial_dims(x_dim='xc', y_dim='yc')
        temp.rio.write_crs('EPSG:6931', inplace=True)

        #Project to arctic CRS
        temp_projected = temp.rio.reproject('EPSG:3995', resolution=ship_res)
        temp_clipped = temp_projected.rio.clip_box(minx=bounds.left,
                                                   miny=bounds.bottom,
                                                   maxx=bounds.right,
                                                   maxy=bounds.top)
        #Use squeeze to remove unused attribute dimension, filter out 0 values as NA to avoid map clutter 
        img_arr_ice = temp_clipped.values.squeeze()
        img_arr_ice = np.where(img_arr_ice <= 0, np.nan, img_arr_ice)
        #Add base image to gif
        im_ice = ax.imshow(img_arr_ice, cmap='Blues_r', extent=extent, norm=plt.Normalize(vmin=0,vmax=100), zorder=3)
    
    #Add shipping base image if present
    if shipping:
        #Open shipping raster and add with correct normalizing for cbar
        with rio.open(f'{maritime}_{years[0]}.tif') as src:
            img_arr_mari = src.read(1)

        im_mari = ax.imshow(img_arr_mari, cmap='Reds_r', extent=extent, norm=norm, zorder=4)

    #Draw figure and set axis for both shipping and ice, adjusting manually
    fig.canvas.draw()
    ax.set_position([0.02, 0.02, 0.84, 0.96])  # [left, bottom, width, height]

    #Shipping cbar
    cbar_ax = fig.add_axes([0.87, 0.08, 0.025, 0.40])
    #Custom ticks for logarithmic scale
    cbar_ticks = [0.001, 0.05, 1, 2, 5, 10, 20, 100, 2000]
    cbar = plt.colorbar(im_mari, cax=cbar_ax, ticks=cbar_ticks, shrink=0.7)
    cbar.ax.set_yticklabels([str(i) for i in cbar_ticks])
    cbar.set_label('AIS Density', fontsize=12)

    #Ice cbar
    ice_cbar_ax = fig.add_axes([0.87, 0.52, 0.025, 0.40])
    ice_cbar = fig.colorbar(im_ice, cax=ice_cbar_ax)
    ice_cbar.set_label('Ice Concentration (%)', fontsize=12)

    #Set blank placeholder title
    title = ax.set_title('', fontsize=14)

    

    def update(frame):
        '''
        Subfunction to update gif with new information

        :param frame: new year for frame
        '''
        #create artists variable to hold title object
        artists = [title]

        #Update ice raster with xarray, same process as before
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

            #Set new data
            im_ice.set_data(img_arr_ice)

            title.set_text(f'Arctic Maritime Traffic Density (AIS Data) - {frame}')
        
        #Update shipping data
        if shipping:
            with rio.open(f'{maritime}_{frame}.tif') as src:
                img_arr_mari = src.read(1)

                im_mari.set_data(img_arr_mari)

                title.set_text(f'Arctic Maritime Traffic Density (AIS Data) - {frame}')
        #Return updated title, updated figure implicitly returned via matplotlib
        return artists
    
    #Use update function to loop through years and update fig, save as anim object
    anim = FuncAnimation(fig, update, frames=years, interval=interval)

    #Save if specified
    if save:
        anim.save(f'figures/{filename}', writer='pillow')
    #Return function
    return anim


def plot_lisa_clusters(clusters, title= 'LISA Clusters'):
    '''
    Given lisa classifications, plot data
    '''
    #Custom lisa colors and generated cmap
    lisa_colors = ['#eeeeee', '#d7191c', '#2c7bb6', '#abd9e9', '#fdae61']
    cmap = ListedColormap(lisa_colors)

    #Retrieve appropriate bounds
    with rio.open('data/ais_traffic/arctic_maritime_2020.tif') as src:
        bounds = src.bounds

    #Get cartopy basemap information
    proj = ccrs.NorthPolarStereo(central_longitude=0, true_scale_latitude=71)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    #Generate fig and ax objects, set ax lims
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': proj})
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)

    #Add basemap features
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor="#717171", edgecolor='none', zorder=0)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#0e0e0e', edgecolor='none', zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#333333', linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#333333', linewidth=0.3, zorder=2)

    #Add clusters with appropriate color
    im = ax.imshow(clusters, extent=extent, cmap=cmap, interpolation='nearest')

    #Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#eeeeee', label='Not Significant'),
        Patch(facecolor='#d7191c', label='High-High (Hot Spot)'),
        Patch(facecolor='#2c7bb6', label='Low-Low (Cold Spot)'),
        Patch(facecolor='#abd9e9', label='Low-High (Outlier)'),
        Patch(facecolor='#fdae61', label='High-Low (Outlier)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    #Title and display
    ax.set_title(title)
    plt.show()
