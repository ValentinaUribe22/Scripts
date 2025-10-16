import pathlib

import geopandas as gpd
import imod
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import xarray as xr

# params
params = snakemake.params
modelname = params["modelname"]
start_time = params["start_time"]
end_time = params["end_time"]

# modelname = "nulmodel_ongekalibreerd0"

# Paths
path_modeltop = snakemake.input.path_modeltop
aoi_shape = snakemake.input.aoi_shape

# path_modeltop = f"data/2-interim/{modelname}/modeltop.nc"
# aoi_shape = "data/1-external/aoi/aoi.shp"

# Open modeltop
top = xr.open_dataset(path_modeltop)["modeltop"]

# Get area
area = gpd.read_file(aoi_shape)
area_raster = imod.prepare.spatial.rasterize(area, like=top)

# Open overlay
overlays = [{"gdf": area, "edgecolor": "black", "color": "none"}]

# Set levels
levels_depth = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    3,
    4,
    5.0,
    7.5,
    10,
    15,
    20,
    30,
]

levels_gws = np.linspace(-1, 4, 20 + 1)

textstr = "\n".join((r"Grey indicates no data",))

# Create folder to save to
pathlib.Path(f"data/5-visualization/{modelname}/heads/").mkdir(
    exist_ok=True, parents=True
)

## Heads
# set paths
path_heads = f"data/4-output/{modelname}/head/head.nc"

# Open file
heads = (
    xr.Dataset()
)  # Trucje: eerst via dataset openen, zodat naamgeving van data-array klopt
heads["head"] = xr.open_dataarray(path_heads)
heads = heads["head"]


# Select first layer
upper_active_layer = imod.select.upper_active_layer(heads.isel(time=0), is_ibound=False)
heads = heads.where(heads.layer == upper_active_layer).where(area_raster.notnull())
heads = heads.max("layer").compute()

heads = heads.sel(time=slice(start_time, end_time))

depth = top - heads
depth = depth.where(area_raster.notnull())

# # Calculate GXG's
# gxg = imod.evaluate.calculate_gxg(heads)
percentile = {"mean": 1, "min": 2, "max": 3, "p90": 0.9, "p10": 0.1}

for percent, j in percentile.items():
    print(percent)
    if j == 1:
        head_mean = heads.mean("time")
    elif j == 2:
        head_mean = heads.min("time")
    elif j == 3:
        head_mean = heads.max("time")
    else:
        # First get 90 percentile for each year
        quantile_data = []
        for year, year_da in heads.groupby(heads.time.dt.year):
            print(year)
            p90 = year_da.compute().quantile(j, dim="time")
            p90 = p90.assign_coords(year=year)
            quantile_data.append(p90)

        quantile = xr.concat(quantile_data, dim="year")

        # Average all percentiles
        head_mean = quantile.mean("year")

    depth_ref = top - head_mean
    depth_ref = depth_ref.where(depth_ref >= 0, 0.0).where(~top.isnull())
    depth_ref = depth_ref.where(area_raster.notnull())
    depth_ref = depth_ref.rename("gw depth (m)")
    head_mean = (
        head_mean.rename("gw level (m)")
        .where(~top.isnull())
        .where(area_raster.notnull())
    )

    # Plot figures
    # Groundwater depth
    print("plotting...")
    plt.axis("scaled")
    fig, ax = imod.visualize.plot_map(
        head_mean,
        colors="jet",
        levels=levels_gws,
        overlays=overlays,
        figsize=[15, 10],
        kwargs_colorbar={"label": "gw level (m)"},
    )
    ax.set_facecolor("0.85")
    ax.set_title(f"Groundwater level, {percent}")

    fig.savefig(
        f"data/5-visualization/{modelname}/heads/{percent} groundwater level.png",
        dpi=300,
    )

    # Plot figures
    # Groundwater depth
    plt.axis("scaled")
    fig, ax = imod.visualize.plot_map(
        depth_ref,
        colors="jet",
        levels=levels_depth,
        overlays=overlays,
        figsize=[15, 10],
        kwargs_colorbar={"label": "depth (m)"},
    )
    ax.set_facecolor("0.85")
    ax.set_title(f"Depth groundwater, {percent}")

    fig.savefig(
        f"data/5-visualization/{modelname}/heads/{percent} Depth of groundwater.png",
        dpi=300,
    )

    # for scenario in scen:
    #     print("Plotting heads", mtype, scenario, j)
    #     # set paths
    #     path_heads = (
    #         f"data/4-output/{mtype}_{scenario}/GWF_1/MODELOUTPUT/HEAD/HEAD.hed"
    #     )
    #     path_grb = f"data/4-output/{mtype}_{scenario}/GWF_1/MODELINPUT/SINGAPORE_V7_DAY_{mtype}_{scenario}.DIS6.grb"

    #     # Open file
    #     heads = imod.mf6.open_hds(path_heads, path_grb)

    #     # Add right timestap to heads
    #     heads = heads[1:].assign_coords(time=rch.time)

    #     if time_period == "2017":
    #         heads = heads[heads["time"].dt.year == 2017]

    #     # Select first layer
    #     heads_data = heads.sel(layer=1, drop=True)

    #     # set name
    #     name = scenario.replace("_", " ")

    #     heads = heads_data.copy()

    #     # First get 90 percentile for each year
    #     quantile_data = []
    #     for year, year_da in heads.groupby(heads.time.dt.year):
    #         print(year)
    #         p90 = year_da.compute().quantile(j, dim="time")
    #         p90 = p90.assign_coords(year=year)
    #         quantile_data.append(p90)

    #     quantile = xr.concat(quantile_data, dim="year")

    #     # Average all percentiles
    #     head_mean = quantile.mean("year")

    #     depth = top - head_mean
    #     depth = depth.where(depth >= 0, 0.0).where(~top.isnull())
    #     depth = depth.where(area_raster.notnull())

    #     if "scen2" in scenario:
    #         depth = depth.sel(x=slice(11500, 33000), y=slice(44450, 33300))
    #         area = pd.concat([area, rch_zone])
    #         # Plot figures
    #         # Groundwater depth
    #         plt.axis("scaled")
    #         fig, ax = imod.visualize.plot_map(
    #             depth.sel(x=slice(11500, 33000), y=slice(44450, 33300)),
    #             colors="jet",
    #             levels=levels_depth,
    #             overlays=overlays,
    #             figsize=[15, 10],
    #             kwargs_colorbar={"label": "gw depth (m)"},
    #         )
    #     else:
    #         depth = depth.sel(x=slice(0, 51000), y=slice(51000, 22000))
    #         fig, ax = imod.visualize.plot_map(
    #             depth.sel(x=slice(0, 51000), y=slice(51000, 22000)),
    #             colors="jet",
    #             levels=levels_depth,
    #             overlays=overlays,
    #             figsize=[15, 10],
    #             kwargs_colorbar={"label": "gw depth (m)"},
    #         )
    #     ax.set_facecolor("0.85")
    #     ax.set_title(
    #         f"Depth of groundwater {mtype} {name}, {percent}, {time_period}"
    #     )

    #     #ax.text(
    #     #    1.05,
    #     #    -0.05,
    #     #    textstr,
    #     #    transform=ax.transAxes,
    #     #    fontsize=11,
    #     #    verticalalignment="bottom",
    #     #)

    #     fig.savefig(
    #         f"data/5-visualization/heads/reference/{time_period} {mtype} {scenario} {percent} Depth of groundwater.png",
    #         dpi=300,
    #     )

    #     plt.close()
    #     # if scen != "ref":
    #     #     # Calculate depth
    #     #     depth_diff = depth - depth_ref
    #     #     depth_diff = depth_diff.rename("Difference gw depth (m)")

    #     #     mask = depth_diff.where((depth_diff < 0.05) & (depth_diff > -0.05))
    #     #     #depth_diff = depth_diff.where(mask.isnull())

    #     #     if "scen2" in scenario:
    #     #         depth_diff = depth_diff.sel(
    #     #             x=slice(11500, 33000), y=slice(44450, 33300)
    #     #         )
    #     #         top_plot = top.where(area_raster.notnull())
    #     #         top_plot = top_plot.sel(
    #     #             x=slice(11500, 33000), y=slice(44450, 33300)
    #     #         )
    #     #         depth_diff = depth_diff * -1
    #     #         # Plot figures
    #     #         # Groundwater depth
    #     #         plt.axis("scaled")
    #     #         fig, ax = imod.visualize.plot_map(
    #     #             depth_diff.sel(x=slice(11500, 33000), y=slice(44450, 33300)),
    #     #             colors="RdBu_r",
    #     #             levels=levels_diff,
    #     #             overlays=overlays,
    #     #             figsize=[15, 10],
    #     #             kwargs_colorbar={"label": "Effect (m)"},
    #     #         )
    #     #     else:
    #     #         depth_diff = depth_diff.sel(
    #     #             x=slice(0, 51000), y=slice(51000, 22000)
    #     #         )
    #     #         top_plot = top.where(area_raster.notnull())
    #     #         top_plot = top_plot.sel(x=slice(0, 51000), y=slice(51000, 22000))
    #     #         depth_diff = depth_diff * -1
    #     #         # Plot figures
    #     #         # Groundwater depth
    #     #         plt.axis("scaled")
    #     #         fig, ax = imod.visualize.plot_map(
    #     #             depth_diff.sel(x=slice(0, 51000), y=slice(51000, 22000)),
    #     #             colors="RdBu_r",
    #     #             levels=levels_diff,
    #     #             overlays=overlays,
    #     #             figsize=[15, 10],
    #     #             kwargs_colorbar={"label": "Effect (m)"},
    #     #         )

    #     #     # im = top_plot.plot.imshow(
    #     #     #    ax=ax, cmap="terrain", add_colorbar=False, alpha=0.75
    #     #     # )
    #     #     ax.set_facecolor("0.85")
    #     #     ax.set_aspect("equal")
    #     #     ax.set_title(
    #     #         f"Effect on groundwater level {mtype} {name} and reference, {percent}, {time_period}"
    #     #     )

    #     #     #ax.text(
    #     #     #    1.05,
    #     #     #    -0.05,
    #     #     #    textstr,
    #     #     #    transform=ax.transAxes,
    #     #     #    fontsize=11,
    #     #     #    verticalalignment="bottom",
    #     #     #)

    #     #     fig.savefig(
    #     #         f"data/5-visualization/heads/{time_period} {mtype} {scenario} {percent} Difference depth of groundwater with reference.png",
    #     #         dpi=300,
    #     #     )

    #     #     plt.close()

    #     #     ## Additional figures
    #     #     if percent == "p90":
    #     #         add_diff = depth.where(depth_diff >= 0.05)

    #     #         add_diff_red = add_diff.where(depth <= 0.02)
    #     #         add_diff_red = add_diff_red.where(add_diff_red.isnull(), 1)

    #     #         add_diff_oranje = add_diff.where(depth >= 0.02)
    #     #         add_diff_oranje = add_diff_oranje.where(add_diff_oranje.isnull(), 2)

    #     #         add_diff_green = add_diff.where(depth >= 1.0)
    #     #         add_diff_green = add_diff_green.where(add_diff_green.isnull(), 3)

    #     #         add_diff_plot = (
    #     #             add_diff_red.combine_first(add_diff_green)
    #     #             .combine_first(add_diff_oranje)
    #     #         )
    #     #         add_diff_plot = add_diff_plot.fillna(0.)
    #     #         add_diff_plot = add_diff_plot.where(area_raster.notnull())

    #     #         to_tiff = add_diff_plot

    #     #         to_tiff.attrs["crs"] = "EPSG:3414"
    #     #         # Save to geotiff
    #     #         imod.rasterio.save(f"data/5-visualization/heads/risk_maps/risk_tif_{scenario}_{percent}_{time_period}.tif", to_tiff, driver="GTIFF")

    #     #         plt.axis("scaled")
    #     #         if "scen2" in scenario:
    #     #             fig, ax = imod.visualize.plot_map(
    #     #                 add_diff_plot.sel(x=slice(11500, 33000), y=slice(44450, 33300)),
    #     #                 colors=["w", "r", "xkcd:mango", "xkcd:leafy green"],
    #     #                 levels=[1, 2, 3],
    #     #                 overlays=overlays,
    #     #                 figsize=[15, 10],
    #     #             )
    #     #         else:
    #     #             fig, ax = imod.visualize.plot_map(
    #     #                 add_diff_plot.sel(x=slice(0, 51000), y=slice(51000, 22000)),
    #     #                 colors=["w", "r", "xkcd:mango", "xkcd:leafy green"],
    #     #                 levels=[1, 2, 3],
    #     #                 overlays=overlays,
    #     #                 figsize=[15, 10],
    #     #             )
    #     #         # im = top_plot.plot.imshow(
    #     #         #    ax=ax, cmap="terrain", add_colorbar=False, alpha=0.75
    #     #         # )
    #     #         ax.set_facecolor("0.85")
    #     #         ax.set_aspect("equal")
    #     #         ax.set_title(
    #     #             f"Risk of new groundwater depth as a result of {mtype} {name}, {percent}, {time_period}"
    #     #         )

    #     #         # set legend
    #     #         red = mlines.Line2D(
    #     #             [], [], color="red", label="Groundwater at surface"
    #     #         )
    #     #         orange = mlines.Line2D(
    #     #             [],
    #     #             [],
    #     #             color="xkcd:mango",
    #     #             label="Groundwater between surface and 1 m below surface",
    #     #         )
    #     #         green = mlines.Line2D(
    #     #             [], [], color="xkcd:leafy green", label="Groundwater deeper than 1 m below surface"
    #     #         )

    #     #         ax.legend(handles=[red, orange, green], loc="upper right")
    #     #         fig.delaxes(fig.axes[1])

    #     #         #ax.text(
    #     #         #    0.82,
    #     #         #    0.05,
    #     #         #    textstr,
    #     #         #    transform=ax.transAxes,
    #     #         #    fontsize=11,
    #     #         #    verticalalignment="bottom",
    #     #         #)

    #     #         fig.savefig(
    #     #             f"data/5-visualization/heads/risk_maps/risk_map_{scenario}_{percent}_{time_period}.png",
    #     #             dpi=300,
    #     #         )

    #     #         ## calculate percentages
    #     #         red = add_diff_plot.where(add_diff_plot == 1).count()
    #     #         orange = add_diff_plot.where(add_diff_plot == 2).count()
    #     #         green = add_diff_plot.where(add_diff_plot == 3).count()
    #     #         total = red + orange + green

    #     #         perc_red = red / total * 100
    #     #         perc_orange = orange / total * 100
    #     #         perc_green = green / total * 100

    #     #         data_perc = {
    #     #             "Total": [total.values],
    #     #             "red": [red.values],
    #     #             "orange": [orange.values],
    #     #             "green": [green.values],
    #     #             "Percentage red": [perc_red.values],
    #     #             "Percentage orange": [perc_orange.values],
    #     #             "Percentage green": [perc_green.values],
    #     #         }

    #     #         df_perc = pd.DataFrame(data=data_perc)
    #     #         df_perc.to_csv(f"data/5-visualization/heads/risk_maps/risk_{scenario}_{percent}_{time_period}.csv")

    #     #         plt.close()
