import os
import numpy as np
import rasterio
from collections import defaultdict
from datetime import datetime

import pystac_client
import planetary_computer
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds as make_transform
from rasterio.crs import CRS

path = "Z:/Zajecia_pliki/poly_researcher" # here path were ur images will be saved
def download_area_data(bbox, output_dir=path+"/images", resolution=100):

    os.makedirs(output_dir, exist_ok=True)
    
    catalog = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace,
    )
    
    target_crs = CRS.from_epsg(32634) 
    wgs84 = CRS.from_epsg(4326)
    
    utm_bounds = transform_bounds(wgs84, target_crs, *bbox)
    width = int((utm_bounds[2] - utm_bounds[0]) / resolution)
    height = int((utm_bounds[3] - utm_bounds[1]) / resolution)
    grid_transform = make_transform(*utm_bounds, width, height)

    # image requirements 
    current_year = datetime.now().year
    search_params = {
        "collections": ['sentinel-2-l2a'],
        "bbox": bbox,
        "datetime": f"{current_year-1}-01-01/{current_year}-12-31", # serching since begining of previous year
        "query": {"eo:cloud_cover": {"lt": 20}} # max cloud coverage
    }
    
    search = catalog.search(**search_params)
    items = list(search.items())
    
    if not items:
        print("No images found for the given criteria.")
        return

    # selecting most recent images
    items.sort(key=lambda x: x.properties.get('datetime'), reverse=True)

    by_tile = defaultdict(list)
    for item in items:
        tile_id = item.properties.get('s2:mgrs_tile')
        by_tile[tile_id].append(item)

    selected_items = [tile_items[0] for tile_items in by_tile.values()]

    bands = ['B02', 'B03', 'B04', 'B08'] # selecting bands
    scale, offset = 0.0001, 0.0
    
    saved_files = {b: [] for b in bands}

    for item in selected_items:
        print(f"Processing tile: {item.properties.get('s2:mgrs_tile')} ({item.id})")
        for band in bands:
            asset = item.assets.get(band)
            if not asset: continue

            out_name = os.path.join(output_dir, f"{item.id}_{band}.tif")
            
            with rasterio.open(asset.href) as src:
                arr = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=grid_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=0
                )

                arr = np.clip(arr * scale + offset, 0, 1)
                
                with rasterio.open(
                    out_name, 'w', driver='GTiff',
                    height=height, width=width, count=1,
                    dtype='float32', crs=target_crs,
                    transform=grid_transform, nodata=0
                ) as dst:
                    dst.write(arr, 1)
                
                saved_files[band].append(out_name)

    print(f"\n--- Done ---")
    print(f"Files saved in: {os.path.abspath(output_dir)}")

bbox = [19.70, 49.95, 20.20, 50.15] # here define bounding box [lon_min, lat_min, lon_max, lat_max]

download_area_data(bbox)