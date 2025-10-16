import imod
import rioxarray
import xarray as xr
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt


# INPUT FILES
top_l1_file = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/TOP_L1.IDF"

# OUTPUT FILES
output_tif = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/Elevation_correction/DEM_model_2050.tif"


# Load data
top_l1 = imod.idf.open(top_l1_file)
top_l1 = top_l1.isel(layer=0)

# Set CRS for GeoTIFF output
top_l1 = top_l1.rio.write_crs("EPSG:28992", inplace=True)


# 5. Save outputs
top_l1.rio.to_raster(output_tif)

print(" DEM saved as IDF and GeoTIFF")
