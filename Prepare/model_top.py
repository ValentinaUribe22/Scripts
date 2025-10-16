import numpy as np
import xarray as xr
import rioxarray
import rasterio
import imod

# modelname = "terschelling_ref"

# Paths
path_ahn = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files/merged_tif.nc"
# = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/1-external/bathy_ahn/ahn2_20m_cmNAP.nc"
path_bathymetry = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/1-external/bathy_ahn/BATHY.idf"

# Interim paths
#path_template= "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_10x10_topfiles.nc"
#path_template_2d = "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_2d_10x10_topfiles.nc"
path_template= "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim/rch_50/template.nc"
path_template_2d = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim/rch_50/template_2d.nc"
path_template_IDF = r"P:\11207941-005-terschelling-model\TERSCHELLING_IMOD_MODEL_50X50\TOP\TOP_L1_merged.IDF"

# Output paths
path_modeltop = "P:/11209740-nbracer/Valentina_Uribe/scenarios_files/modeltop_3.nc"
path_modeltop_min = "P:/11209740-nbracer/Valentina_Uribe/scenarios_files/modeltop_min_py.nc"
path_modeltop_idf = "P:/11209740-nbracer/Valentina_Uribe/scenarios_files/modeltop5.IDF"
path_modeltop_min_idf = "P:/11209740-nbracer/Valentina_Uribe/scenarios_files/modeltop_min.IDF"

# Open templates
# like = xr.open_dataset(path_template)["template"]
# like_2d = xr.open_dataset(path_template_2d)["template"]
# dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
# zmin = like.zbot.min()
# zmax = like.ztop.max()
# dz = like.dz

# Use the IDF file as template
like = imod.idf.open(path_template_IDF)  # This is a 2D array if it's from the TOP folder
like_2d = like  # You can use the same for 2D template if it's only one layer

# Get grid and spatial extent
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)


# AHN
ahn = xr.open_dataset(path_ahn).sel(x=slice(xmin, xmax), y=slice(ymax, ymin))["Band1"]
# remove nodata, also casts to float64
nodata = ahn.rio.nodata
ahn = ahn.where(ahn != nodata)
# = ahn.where(ahn != ahn.attrs["nodatavals"])
ahn = ahn * 0.01


# bathymetrie
bathy = (
    imod.idf.open(path_bathymetry)
    .sel(y=slice(ymax, ymin))
    .sel(x=slice(xmin, xmax))
    .astype(np.float64)
)

# # Open dataset
# bathym = xr.open_dataset(path_bathymetry)
# bathy = bathym["BATHY_RDNEW.IDF"]


# # Rename dimensions
# bathy = bathy.rename({'xc': 'x', 'yc': 'y'})
# # Slice to model extent
# bathy = bathy.sel(x=slice(xmin, xmax), y=slice(ymax, ymin))

# Regrid AHN and the bathymetry
mean_2d_regridder = imod.prepare.Regridder(method="mean")
top = mean_2d_regridder.regrid(ahn, like_2d)
seafloor = mean_2d_regridder.regrid(bathy, like_2d)

# combine AHN and bathymetry
top = top.combine_first(seafloor)
top = imod.prepare.fill(top)
top = top.clip(max=10.0)
top.name = "modeltop"
top.to_netcdf(path_modeltop)
imod.idf.write(path_modeltop_idf, top)

# Create minimum modeltop for rivers
# Regrid AHN and the bathymetry
min_2d_regridder = imod.prepare.Regridder(method="minimum")
top_min = min_2d_regridder.regrid(ahn, like_2d)
seafloor_min = min_2d_regridder.regrid(bathy, like_2d)

# combine AHN and bathymetry
top_min = top_min.combine_first(seafloor_min)
top_min = imod.prepare.fill(top_min)

top_min.name = "modeltop_minimum"
top_min.to_netcdf(path_modeltop_min)
imod.idf.write(path_modeltop_min_idf, top_min)
