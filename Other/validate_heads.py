import pathlib

import contextily as ctx
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import shapely.geometry as sg
import tqdm
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Proj, transform

import imod

# params
params = snakemake.params
modelname = params["modelname"]

# Paths
aoi_shape = snakemake.input.aoi_shape
path_val_da_dino = snakemake.input.path_val_da_dino
path_val_da_vitens = snakemake.input.path_val_da_vitens
path_waterschap_data = snakemake.input.path_waterschap_data
path_template = snakemake.input.path_template

# modelname= "Nulmodel_Kalibratie3"
# aoi_shape = "data/1-external/aoi/aoi.shp"
# path_val_da_dino = "data/2-interim/validation_data/dino_validatie_data.csv"
# path_val_da_vitens = "data/2-interim/validation_data/vitens_validatie_data.csv"
# path_waterschap_data = "data/2-interim/validation_data/waterschap_validatie_data.csv"
# path_template = f"data/2-interim/{modelname}/template.nc"


# Turn interactive plotting off
plt.ioff()

# Define functions
@numba.njit
def iloc_indices(tops, bottoms):
    indices = []
    layers = []
    for ind, (t, b) in enumerate(zip(tops, bottoms)):
        for i in range(t, b + 1):
            layers.append(i)
            indices.append(ind)
    return np.array(indices), np.array(layers)

def well_layers(like, elevation):
    inds = imod.select.points_indices(like, z=elevation)["z"].values
    layers = like.coords["layer"].values[inds]
    return layers

# Location data paths
path_output_head = f"data/4-output/{modelname}/head/head*"

# opening template
like = xr.open_dataset(path_template)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = -250  # todo
zmax = 6.0  # todo
dz = like.dz

# Opening of validation data from DINOloket
val_da = pd.read_csv(path_val_da_dino)
val_da["time"] = pd.to_datetime(val_da["time"])

# Opening validation data from Vitens
val_da_vitens = pd.read_csv(path_val_da_vitens)
val_da_vitens["time"] = pd.to_datetime(val_da_vitens["time"])

# Opening validation data from the waterschap
val_da_waterschap = pd.read_csv(path_waterschap_data)
val_da_waterschap["time"] = pd.to_datetime(val_da_waterschap["time"])

# Combine datasets
val_da = pd.concat([val_da, val_da_vitens, val_da_waterschap])

# Opening output head data
head_out = xr.Dataset()
head_out['head'] = xr.open_dataarray(f"data/4-output/{modelname}/head/head.nc")

# Remove data outside area of interest
val_da = val_da.where((val_da["x"] > xmin) & (val_da["x"] < xmax)).dropna(
    axis=0, how="all"
)
val_da = val_da.where((val_da["y"] > ymin) & (val_da["y"] < ymax)).dropna(
    axis=0, how="all"
)
val_da = val_da.where(
    (val_da["filt_bot"] > zmin) & (val_da["filt_bot"] < zmax)
).dropna(axis=0, how="all")

val_da["filt_middle"] = (val_da["filt_top"] + val_da["filt_bot"]) / 2

# Add corresponding layer to validation data
val_da["layer"] = well_layers(like, val_da["filt_middle"])

# Select modeldata at measurement site
df_temp = pd.DataFrame()
dfs = []
for well_id, well_df in tqdm.tqdm(val_da.groupby("id")):
    x = well_df["x"].iloc[0]
    y = well_df["y"].iloc[0]
    layer = well_df["layer"].iloc[0]
    arrayhead = imod.select.points_values(head_out, x=x, y=y, layer=layer)
    df_temp = pd.DataFrame()
    df_temp["time"] = arrayhead.time
    df_temp["modelled_head"] = arrayhead['head'].values[0] # 0 to get first index
    df_temp["id"] = well_id
    dfs.append(df_temp)

vali_data = pd.concat(dfs)

df = val_da.groupby("id").first()[
    ["x", "y", "filt_top", "filt_bot", "filt_middle", "layer"]
]
df["modelled_head_mean"] = vali_data.groupby("id")["modelled_head"].mean()
df["modelled_head_p10"] = vali_data.groupby("id")["modelled_head"].quantile(0.1)
df["modelled_head_p90"] = vali_data.groupby("id")["modelled_head"].quantile(0.9)
df["head_mean"] = val_da.groupby("id")["head"].mean()
df["head_p10"] = val_da.groupby("id")["head"].quantile(0.1)
df["head_p90"] = val_da.groupby("id")["head"].quantile(0.9)

df["diff"] = df["modelled_head_mean"] - df["head_mean"]
df["absdiff"] = np.abs(df["diff"])

# Create folder to save to
pathlib.Path(f"data/5-visualization/{modelname}/validate_heads").mkdir(
    exist_ok=True, parents=True
)

df.to_csv(f"data/5-visualization/{modelname}/validate_heads/head-results.csv")
df[df["filt_top"] < -10.0].to_csv(f"data/5-visualization/{modelname}/validate_heads/head-deep.csv")
df[df["filt_top"] >= -10.0].to_csv(f"data/5-visualization/{modelname}/validate_heads/head-shallow.csv")

## plots
for flavour in ["mean", "p10", "p90"]:
    for zrange in [[0, -10], [-10, -50]]:  # [0,-10]=phreatic, [-10,-50]=1wvp

        in_layer = df[
            (df["filt_middle"] < zrange[0]) & (df["filt_middle"] > zrange[1])
        ]

        x = in_layer[f"modelled_head_{flavour}"]
        y = in_layer[f"head_{flavour}"]

        max_value = max([x.max(), y.max()])
        min_value = min([x.min(), y.min()])

        fig, ax = plt.subplots()
        ax.set_xlabel("Modelled head")
        ax.set_ylabel("Observed head")
        ax.set_title(f"{flavour} between {zrange[0]} and {zrange[1]}m NAP")
        ax.plot([max_value, min_value], [max_value, min_value], "k-")
        ax.plot(
            [max_value + 0.5, min_value + 0.5],
            [max_value, min_value],
            "k--",
            alpha=0.5,
        )
        ax.plot([max_value + 1.0, min_value + 1.0], [max_value, min_value], "k--")
        ax.plot(
            [max_value - 0.5, min_value - 0.5],
            [max_value, min_value],
            "k--",
            alpha=0.5,
        )
        ax.plot([max_value - 1.0, min_value - 1.0], [max_value, min_value], "k--")
        ax.scatter(x, y, zorder=20, s=1, c="b")

        fig.savefig(
            f"data/5-visualization/{modelname}/validate_heads/{flavour} between {zrange[0]} and {zrange[1]}m NAP.png",
            dpi=300,
        )

# Spatially plotting the absolute difference in head
geometry = [sg.Point(float(x), float(y)) for x, y in zip(df["x"], df["y"])]

gdf = gpd.GeoDataFrame({"geometry": geometry})
for column in df.columns:
    gdf[column] = df[column].values

# # Calculating absdiff
# gdf["absdiff"] = (gdf["modelled_head_mean"] - gdf["head_mean"]).abs()

# # Calculating relative difference, modelled-obs thus 
# gdf["diff"] = gdf["modelled_head_mean"] - gdf["head_mean"]

# Open overlay
aoi = gpd.read_file(aoi_shape)

gdf_rp = gdf.copy()

# converting coordinates epsg:28992 to epsg:3857
gdf.crs = "EPSG:28992"
gdfrp = gdf.to_crs("EPSG:3857")
aoi = aoi.to_crs("EPSG:3857")

# color bar settings
colors = "YlOrRd"
levels = [0, 0.25, 0.5, 1, 2, 5, 7]

# color bar magic. do not touch
nlevels = len(levels)
cmap = matplotlib.cm.get_cmap(colors)
colors = cmap(np.linspace(0, 1, nlevels + 1))
cmap = matplotlib.colors.ListedColormap(colors[1:-1])
# Make triangles white if data is not larger/smaller than legend_levels-range
cmap.set_under(colors[0])
cmap.set_over(colors[-1])
if gdf_rp["absdiff"].max() < levels[-1]:
    cmap.set_over("#FFFFFF")
if gdf_rp["absdiff"].min() > levels[0]:
    cmap.set_under("#FFFFFF")
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

# Plot data points
fig, ax = plt.subplots(figsize=(15, 10))
gdfrp.sort_values(by="absdiff").plot(
    column="absdiff", ax=ax, legend=False, cmap=cmap, norm=norm, edgecolor="k"
)

# Plot AOI
aoi.plot(edgecolor="black", color="none", ax=ax)

# Plot basemap
ctx.add_basemap(ax)

# Add colorbar
settings_cbar = {"ticks": levels, "extend": "both"}
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad="5%")
fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

# Plot settings
ax.set_title(f"Absolute difference between modelled and observed head")
fig.tight_layout()

# save fig
# fig.show()
fig.savefig(
    f"data/5-visualization/{modelname}/validate_heads/spatial absolute difference heads.png",
    dpi=300,
)

# Plotting relative difference
levels = [-3,-2,-1,-0.5,-0.25,-0.05,0.05, 0.25, 0.5, 1, 2, 3]
colors = "RdBu"

# color bar magic. do not touch
nlevels = len(levels)
cmap = matplotlib.cm.get_cmap(colors)
colors = cmap(np.linspace(0, 1, nlevels + 1))
cmap = matplotlib.colors.ListedColormap(colors[1:-1])
# Make triangles white if data is not larger/smaller than legend_levels-range
cmap.set_under(colors[0])
cmap.set_over(colors[-1])
if gdf_rp["diff"].max() < levels[-1]:
    cmap.set_over("#FFFFFF")
if gdf_rp["diff"].min() > levels[0]:
    cmap.set_under("#FFFFFF")
norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)
cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

# Plot data points
fig, ax = plt.subplots(figsize=(15, 10))
gdfrp.sort_values(by="diff").plot(
    column="diff", ax=ax, legend=False, cmap=cmap, norm=norm, edgecolor="k"
)

# Plot AOI
aoi.plot(edgecolor="black", color="none", ax=ax)

# Plot basemap
ctx.add_basemap(ax)

# Add colorbar
settings_cbar = {"ticks": levels, "extend": "both"}
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad="5%")
fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

# Plot settings
ax.set_title(f"Difference between modelled and observed head")
fig.tight_layout()

# save fig
# fig.show()
fig.savefig(
    f"data/5-visualization/{modelname}/validate_heads/spatial difference heads.png",
    dpi=300,
)