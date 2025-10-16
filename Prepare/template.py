import geopandas as gpd
import imod
import numpy as np
import xarray as xr


def get_extent(path_area, cellsize):
    if ".shp" in path_area:
        gdf = gpd.read_file(path_area)
        gdf = gdf.loc[gdf.id == 1]
        extent = imod.prepare.spatial.round_extent(gdf.bounds.values[0], cellsize)
    elif ".tif" in path_area:
        dem = imod.rasterio.open(path_area)
        dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(dem)
        extent = (
            xmin + 0.5 * abs(dx),
            ymin + 0.5 * abs(dy),
            xmax - 0.5 * abs(dx),
            ymax - 0.5 * abs(dy),
        )
    else:
        raise FileNotFoundError("No shapefile or DEM found to derive extent from")
    return extent


## Set snakemake variables
modelname = "reference"
cellsize = 10
zmin = -150
zmax = 28
dz = np.array([-5.0] * 4 + [-1.0] * 4 + [-0.5] * 8 + [-1.0] * 10 + [-10.0] * 4 + [-20.0] * 5)


## Paths
# Input
path_area = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/1-external/aoi/aoi_model_adj.shp"
# Output
path_template = "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_10x10.nc"
path_template_2d = "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_2d_10x10.nc"

# Open data
extent = get_extent(path_area, cellsize)
xmin, ymin, xmax, ymax = extent
zmin = zmin
zmax = zmax
size_top_dz = 0.5

# Discretization
cellsize = float(cellsize)
dx = cellsize
dy = -cellsize
dz = dz
layer = np.arange(1, 1 + dz.size)

# Check if discretization matches with zmin and zmax, throw error if it doesn't match up:
assert dz.sum() == (
    zmin - zmax
), "Take heed, your vertical discretization does not add up to (zmax - zmin)"

# define template
dims = ("z", "y", "x")
coords = {
    "z": zmax + dz.cumsum() - 0.5 * dz,
    "y": np.arange(ymax, ymin, dy) + 0.5 * dy,
    "x": np.arange(xmin, xmax, dx) + 0.5 * dx,
    "dz": ("z", dz),
    "layer": ("z", layer),
    "zbot": ("z", zmax + dz.cumsum()),
    "ztop": ("z", zmax + dz.cumsum() - dz),
}
nrow = coords["y"].size
ncol = coords["x"].size
nlay = coords["z"].size
like = xr.DataArray(np.full((nlay, nrow, ncol), np.nan), coords, dims)
like_2d = like.isel(z=0).drop(["z", "layer", "dz", "zbot", "ztop"])

like.name = "template_10x10"
like_2d.name = "template_2d_10x10"
like.to_netcdf(path_template)
like_2d.to_netcdf(path_template_2d)

