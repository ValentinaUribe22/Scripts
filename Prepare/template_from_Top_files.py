import xarray as xr
import numpy as np
import geopandas as gpd
import imod

# Load modeltop
path_modeltop = "P:/11209740-nbracer/Valentina_Uribe/scenarios_files/modeltop_py.nc"
modeltop = xr.open_dataarray(path_modeltop)

# Use coordinates from modeltop
x = modeltop.x
y = modeltop.y

# Set vertical domain
zmin = -150
zmax = 28
dz = np.array([-5.0]*4 + [-1.0]*4 + [-0.5]*8 + [-1.0]*10 + [-10.0]*4 + [-20.0]*5)

# Validate vertical discretization
assert dz.sum() == (zmin - zmax), "Vertical dz does not match (zmin - zmax)"

# Construct z-related coordinates
layer = np.arange(1, 1 + dz.size)
z = zmax + dz.cumsum() - 0.5 * dz

# Create 3D grid
dims = ("z", "y", "x")
coords = {
    "z": z,
    "y": y,
    "x": x,
    "dz": ("z", dz),
    "layer": ("z", layer),
    "zbot": ("z", zmax + dz.cumsum()),
    "ztop": ("z", zmax + dz.cumsum() - dz),
}

nlay = dz.size
nrow = y.size
ncol = x.size

like = xr.DataArray(
    np.full((nlay, nrow, ncol), np.nan),
    coords=coords,
    dims=dims,
    name="template_10x10"
)

like_2d = like.isel(z=0).drop_vars(["z", "layer", "dz", "zbot", "ztop"])
like_2d.name = "template_2d_10x10"


# Output
path_template = "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_10x10_nmt.nc"
path_template_2d = "P:/11209740-nbracer/Valentina_Uribe/Prepare/templates/template_2d_10x10_nmt.nc"

like.to_netcdf(path_template)
like_2d.to_netcdf(path_template_2d)

# To check if my modeltp matches the template
#1. Load both datasets
modeltop = imod.idf.open("P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/TOP_10x10/TOP_L1.IDF")

# 2. Check grid resolution
dx_like = like.x[1] - like.x[0]
dy_like = like.y[1] - like.y[0]

dx_top = modeltop.x[1] - modeltop.x[0]
dy_top = modeltop.y[1] - modeltop.y[0]

print("Resolution check:")
print(f"like dx: {dx_like.values}, modeltop dx: {dx_top.values}")
print(f"like dy: {dy_like.values}, modeltop dy: {dy_top.values}")

# 3. Check extent
print("Extent check:")
print(f"like extent: x({like.x.min().values}, {like.x.max().values}), y({like.y.min().values}, {like.y.max().values})")
print(f"modeltop extent: x({modeltop.x.min().values}, {modeltop.x.max().values}), y({modeltop.y.min().values}, {modeltop.y.max().values})")


print("like.x[:3] =", like.x[:3].values)
print("modeltop.x[:3] =", modeltop.x[:3].values)
