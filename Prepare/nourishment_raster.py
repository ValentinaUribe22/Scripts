import rasterio
import geopandas as gpd
import rasterio.mask
import numpy as np
import imod
import xarray as xr
import rioxarray

# Input files
full_raster_path = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\Add_nourishment\DEM_model_2100.tif"
polygon_path = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\QGIS\all_polygon_nourishmnt.shp"
output_raster_path = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\Nourishment\final_DEM_IDF.tif"
output_idf_path = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\Nourishment\final_DEM_IDF.idf"
path_aoi = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/QGIS/aoi_model_adj.shp"
top_l1_file = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/TOP_L1.IDF"


## Read polygon shapefiles
nourish = gpd.read_file(polygon_path)
aoi = gpd.read_file(path_aoi)

# Open full raster
with rasterio.open(full_raster_path) as src:
    # Clip raster to AOI
    aoi_shapes = [geom for geom in aoi.to_crs(src.crs).geometry]
    clipped_data, clipped_transform = rasterio.mask.mask(src, aoi_shapes, crop=True, nodata=src.nodata)
    clipped_data = clipped_data[0]  # first band
    clipped_profile = src.profile.copy()
    clipped_profile.update({
        "height": clipped_data.shape[0],
        "width": clipped_data.shape[1],
        "transform": clipped_transform
    })

    # Create mask for nourishment polygon (True inside polygon)
    nourish_shapes = [geom for geom in nourish.to_crs(src.crs).geometry]
    nourish_mask = rasterio.features.geometry_mask(
        nourish_shapes,
        transform=clipped_transform,
        invert=True,
        out_shape=clipped_data.shape
    )

    # Add 2.046 to cells inside nourishment polygon
    clipped_data = clipped_data.astype(float)
    clipped_data[nourish_mask] = clipped_data[nourish_mask] + 2.046

top_l1 = imod.idf.open(top_l1_file)
top_l1 = top_l1.isel(layer=0)
x_coords = top_l1.x.values
y_coords = top_l1.y.values
nodata_value = top_l1.attrs.get("nodata", -9999)

# Convert to xarray.DataArray (required by imod)
da = xr.DataArray(
    clipped_data,
    dims=("y", "x"),
    coords={"y": y_coords, "x": x_coords},
    attrs=top_l1.attrs  # copy metadata from top_l1
)

# Save new raster
imod.idf.write(output_idf_path, da, nodata=nodata_value)
da.rio.to_raster(output_raster_path)

print(" DEM saved as IDF and GeoTIFF")

