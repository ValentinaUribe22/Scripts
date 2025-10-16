import pathlib
import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import shapely.geometry as sg
import matplotlib
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

rulename = "heads_timeseries_validate"
path_snakefile = (
    r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model\snakefile"
)


def read_snakemake_rule(path, rule: str) -> "snakemake.rules.Rule":
    import snakemake as sm

    workflow = sm.Workflow(snakefile="snakefile")
    workflow.include(path)
    return workflow.get_rule(rule)


if "snakemake" not in globals():
    os.chdir(r"c:\Users\kelde_ts\data\3_projects\Terschelling\terschelling-gw-model")
    snakemake = read_snakemake_rule(path_snakefile, rule=rulename)

## Input
path_template = snakemake.input.path_template
path_obs_wells = snakemake.input.path_obs_wells
path_dem = snakemake.input.path_dem
path_val_da_vitens = snakemake.input.path_val_da_vitens
path_waterschap_data = snakemake.input.path_waterschap_data
aoi_shape = snakemake.input.aoi_shape
path_legend_residu = snakemake.input.path_legend_residu
## Parameters
modelname = snakemake.params["modelname"]
frequency = snakemake.params["frequency"]
compare_measurements = snakemake.params["compare_measurements"]
compare_other_run = snakemake.params["compare_other_run"]
p_drive = snakemake.params["p_drive"]
modelname_compare = snakemake.params["modelname_compare"]
start_time = snakemake.params["start_time"]
end_time = snakemake.params["end_time"]

# Open template
like = xr.open_dataset(path_template)["template"]
aoi = gpd.read_file(aoi_shape)

# Open dataset to plot
# dino data
ds_obs = pd.read_csv(path_obs_wells)
# Vitens data
val_da_vitens = pd.read_csv(path_val_da_vitens)
# waterschap data
val_da_waterschap = pd.read_csv(path_waterschap_data)

# Combine datasets
ds_obs = pd.concat([ds_obs, val_da_vitens, val_da_waterschap])
ds_obs["time"] = pd.to_datetime(ds_obs["time"])

# Make sure only points in bounds are included
in_bounds = imod.select.points_in_bounds(like, x=ds_obs["x"], y=ds_obs["y"])
ds_obs = ds_obs[in_bounds]

ds_obs["filt_depth"] = (ds_obs["filt_top"] + ds_obs["filt_bot"]) / 2
ds_obs["layer"] = (
    like.sel(z=ds_obs["filt_depth"].values, method="Nearest").coords["layer"].values
)

## Scheidt info van metingen
ds_obs = ds_obs.replace(np.nan, float("NaN"))
obs_info = ds_obs[
    [
        "id",
        "Filternummer",
        "filt_top",
        "filt_bot",
        "filt_depth",
        "Meetpunt tov m NAP",
        "x",
        "y",
        "layer",
    ]
]
obs_info = obs_info.round(3)
obs_info = obs_info.drop_duplicates()
obs_info.set_index(obs_info["id"], inplace=True)

## Resample metingen naar gewenste frequentie
ds_obs2 = ds_obs.set_index("time")
ds_obs_resampled = ds_obs2.groupby("id").resample(frequency).mean().reset_index()


# Create GDF for plotting purposes
gdf = gpd.GeoDataFrame(
    ds_obs_resampled,
    geometry=gpd.points_from_xy(ds_obs_resampled["x"], ds_obs_resampled["y"]),
)
gdf["name"] = gdf["id"]

# Open DEM
dem = xr.open_dataarray(path_dem)

# PLot observation locations
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    dem,
    colors="terrain",
    levels=np.linspace(-10, 10, 20),
    figsize=[20, 12],
)
gdf.plot(column="name", legend=False, color="k", ax=ax, zorder=2.5)
fig.delaxes(fig.axes[1])
ax.set_title("Location observations")
pathlib.Path(f"data/5-visualization/{modelname}/timeseries/").mkdir(
    exist_ok=True, parents=True
)
fig.savefig(f"data/5-visualization/{modelname}/timeseries/map.png", dpi=300)

for mtype in ["head"]:
    print(mtype)
    # Read modelled data
    output_ds = xr.Dataset()
    output_ds[mtype] = xr.open_dataarray(
        f"data/4-output/{modelname}/{mtype}/{mtype}.nc"
    ).sel(time=slice(start_time, end_time))

    # Select start and end times for plotting
    sdate = output_ds.time[0].values
    edate = output_ds.time[-1].values

    # Extract modelled data
    model_ds = imod.select.points_values(
        output_ds, x=obs_info["x"], y=obs_info["y"], layer=obs_info["layer"]
    ).to_dataframe()
    model_ds = model_ds.reset_index()
    model_ds = model_ds.rename(columns={"index": "id"})
    model_ds = model_ds.drop(columns=["x", "y"])

    # if mtype == "head":
    # ## Calculate GXG's
    # pathlib.Path(f"data/5-visualization/gxg/").mkdir(
    #     exist_ok=True, parents=True
    # )
    # out_ds = output_ds.sel(layer=ds_obs["layer"]).drop_duplicates()[0])
    # gxg_ds = imod.evaluate.calculate_gxg(out_ds)

    # # Extract modelled data
    # gxg = imod.select.points_values(
    #     gxg_ds, x=ds_obs["x"], y=ds_obs["y"]
    # ).to_dataframe()
    # gxg = gxg.reset_index()
    # gxg = gxg.rename(columns={"index": "id"})
    # gxg = gxg.drop(columns=["dx", "dy", "layer", "x", "y"])
    # pathlib.Path(f"data/5-visualization/{modelname}/gxg/").mkdir(
    #     exist_ok=True, parents=True
    # )
    # gxg.to_csv(f"data/5-visualization/{modelname}/gxg/{modelname}_gxg.csv")

    pathlib.Path(f"data/5-visualization/{modelname}/timeseries/{mtype}/").mkdir(
        exist_ok=True, parents=True
    )

    model_ds.to_csv(
        f"data/5-visualization/{modelname}/timeseries/{mtype}_timeseries.csv"
    )

    ids = model_ds.drop_duplicates(subset="id").reset_index()
    id_list = ids["id"]

    # df_validate =
    """ SAVE GROUNDWATER HEAD"""
    pathlib.Path(f"data/5-visualization/{modelname}/grondwaterstand/").mkdir(
        exist_ok=True, parents=True
    )

    ## Specify groundwater
    test = output_ds.isel(time=0)
    is_topcel = test[mtype]["layer"] == test["head"]["layer"].where(
        test["head"].notnull()
    ).min("layer")
    groundwater = output_ds[mtype].where(is_topcel == True).mean("layer")
    groundwater = groundwater.transpose("time", "y", "x")

    ## Save timeseries and mean
    imod.idf.save(
        f"data/5-visualization/{modelname}/grondwaterstand/grondwater.idf", groundwater
    )
    imod.idf.write(
        f"data/5-visualization/{modelname}/grondwaterstand/grondwater_gemiddeld.idf",
        groundwater.mean("time"),
    )

    ## Save head at -5m NAP
    gws_min5 = output_ds[mtype].sel(layer=21).drop("layer").mean("time")
    imod.idf.write(
        f"data/5-visualization/{modelname}/grondwaterstand/head_min5mNAP.idf", gws_min5
    )

    ## Calculate difference
    if compare_other_run == True:
        if os.path.isfile(
            f"{p_drive}/data/5-visualization/{modelname_compare}/grondwaterstand/grondwater_gemiddeld.idf"
        ):
            groundwater_reference = imod.idf.open(
                f"{p_drive}/data/5-visualization/{modelname_compare}/grondwaterstand/grondwater_gemiddeld.idf"
            )
            gws_min5_reference = imod.idf.open(
                f"{p_drive}/data/5-visualization/{modelname_compare}/grondwaterstand/head_min5mNAP.idf"
            )
            diff_groundwater = groundwater.mean("time") - groundwater_reference
            diff_gws_min5 = gws_min5 - gws_min5_reference

            ## Plotten
            legend = imod.visualize.spatial.read_imod_legend(path_legend_residu)
            overlays = [{"gdf": aoi, "edgecolor": "black", "color": "none"}]
            for diff, name in zip(
                [diff_groundwater, diff_gws_min5],
                ["verschil_grondwaterstand", "verschil_head_min5mNAP"],
            ):
                fig, ax = imod.visualize.plot_map(
                    diff,
                    legend[0],
                    legend[1],
                    overlays=overlays,
                    figsize=[15, 10],
                    kwargs_colorbar={"label": "verschil (m)"},
                )
                ax.set_facecolor("0.85")
                ax.set_title(name)
                fig.savefig(
                    f"data/5-visualization/{modelname}/grondwaterstand/diff_{modelname}_{modelname_compare}_{name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
    """ MAKE IPF """
    ds_obs2 = ds_obs[["id", "time", "head"]]
    ds_obs_resampled2 = ds_obs_resampled[["id", "time", "head"]]
    ## Combine datasets
    model = model_ds.copy()
    model.rename(columns={"head": "head_modelled"}, inplace=True)
    ds_obs_resampled2.rename(columns={"head": "head_resampled"}, inplace=True)
    timeseries_all = pd.merge(ds_obs_resampled2, model, how="outer", on=["id", "time"])
    timeseries_all = timeseries_all[["id", "time", "head_resampled", "head_modelled"]]

    ## Calcate difference and add information
    timeseries_all["difference"] = (
        timeseries_all["head_modelled"] - timeseries_all["head_resampled"]
    )
    mean = timeseries_all[["id", "difference"]].groupby("id").mean().reset_index()
    median = timeseries_all[["id", "difference"]].groupby("id").median().reset_index()
    mean.columns = ["id", "mean_difference"]
    median.columns = ["id", "median_difference"]
    obs_info2 = obs_info.reset_index(drop=True).drop(columns=["layer"])
    obs_info2 = pd.merge(obs_info2, mean, how="left", on="id")
    obs_info2 = pd.merge(obs_info2, median, how="left", on="id")

    final_dataset = pd.merge(timeseries_all, obs_info2, how="inner", on="id")
    final_dataset.sort_values(by=["time"], inplace=True)
    final_dataset["id"] = "combined/" + final_dataset["id"]

    ## Add info from reference run
    if compare_other_run == True:
        if os.path.isfile(
            rf"{p_drive}\data\5-visualization\{modelname_compare}\timeseries\{modelname_compare}_gws.ipf"
        ):
            tijdreeks_ref = imod.ipf.read(
                rf"{p_drive}\data\5-visualization\{modelname_compare}\timeseries\{modelname_compare}_gws.ipf"
            )
            ## Kolom verwijderen
            if "head_refmodel" in tijdreeks_ref.columns:
                tijdreeks_ref = tijdreeks_ref.drop(["head_refmodel"], axis=1)
            ## Kolommen herbepalen
            tijdreeks_ref = tijdreeks_ref.rename(
                {"head_modelled": "head_refmodel"}, axis="columns"
            )
            tijdreeks_ref = tijdreeks_ref[
                [
                    "time",
                    "head_refmodel",
                    "x",
                    "y",
                    "id",
                    "filternummer",
                    "filt_top",
                    "filt_bot",
                    "filt_depth",
                    "meetpunt tov m nap",
                ]
            ]
            tijdreeks_ref.rename(columns={"filternummer": "Filternummer"}, inplace=True)
            tijdreeks_ref.rename(
                columns={"meetpunt tov m nap": "Meetpunt tov m NAP"}, inplace=True
            )
            tijdreeks_ref = tijdreeks_ref.dropna(subset=["head_refmodel"])
            ## Samenvoegen
            final_dataset = pd.merge(
                final_dataset,
                tijdreeks_ref,
                how="left",
                on=[
                    "time",
                    "x",
                    "y",
                    "id",
                    "Filternummer",
                    "filt_top",
                    "filt_bot",
                    "filt_depth",
                    "Meetpunt tov m NAP",
                ],
            )
            final_dataset = final_dataset[
                [
                    "id",
                    "time",
                    "head_resampled",
                    "head_modelled",
                    "difference",
                    "head_refmodel",
                    "Filternummer",
                    "filt_top",
                    "filt_bot",
                    "filt_depth",
                    "Meetpunt tov m NAP",
                    "x",
                    "y",
                    "mean_difference",
                    "median_difference",
                ]
            ]

    ## Make dataset with raw (daily) measurements
    measurements = pd.merge(ds_obs2, obs_info2, how="inner", on="id")
    measurements.sort_values(by=["time"], inplace=True)
    measurements["id"] = "raw_measurements/" + measurements["id"]

    ## Verwijder meetreeksen met minder dan 10 metingen
    tijdreeks_modelperiod = final_dataset.dropna(
        subset=["head_modelled"]
    )  # Alleen records binnen modelperiode
    subset = tijdreeks_modelperiod[["id", "head_resampled"]].dropna(
        subset=["head_resampled"]
    )  # alleen records met metingen
    count_metingen = subset.groupby(["id"]).count()
    genoeg_metingen = list(
        count_metingen[count_metingen["head_resampled"] > 9].reset_index()["id"]
    )
    final_dataset2 = final_dataset[final_dataset["id"].isin(genoeg_metingen)]

    ## To ipf
    if compare_other_run == True:
        if "head_refmodel" in tijdreeks_ref.columns:
            imod.ipf.save(
                f"data/5-visualization/{modelname}/timeseries/{modelname}_gws.ipf",
                final_dataset2,
                itype="timeseries",
                assoc_columns=[
                    "time",
                    "head_resampled",
                    "head_modelled",
                    "head_refmodel",
                    "difference",
                ],
            )
    else:
        imod.ipf.save(
            f"data/5-visualization/{modelname}/timeseries/{modelname}_gws.ipf",
            final_dataset2,
            itype="timeseries",
            assoc_columns=["time", "head_resampled", "head_modelled", "difference"],
        )
    imod.ipf.save(
        f"data/5-visualization/{modelname}/timeseries/raw_measurements.ipf",
        measurements,
        itype="timeseries",
        assoc_columns=["time", "head"],
    )

    """ PLOT TIMESERIES """
    for j in id_list:
        if model_ds.loc[model_ds["id"] == j][mtype].dropna().empty == False:
            print(j)

            # Figuur gereed maken
            row = 1
            col = 2
            fig, axs = plt.subplots(
                col, row, figsize=(12, 14), sharey=False, sharex=False, squeeze=False
            )
            # filterpositie inladen
            layer = obs_info["layer"][j]
            depth = obs_info["filt_depth"][j]
            x_value = obs_info["x"][j]
            y_value = obs_info["y"][j]

            # Plot modelled data
            model_ds.loc[model_ds["id"] == j].where(
                model_ds.loc[model_ds["id"] == j].notnull()
            ).plot(
                x="time",
                y=mtype,
                color="b",
                linestyle="solid",
                linewidth=1,
                legend=False,
                ax=axs[0][0],
            )
            ## plot measurement data
            if compare_measurements == True:
                ds_obs_resampled.loc[ds_obs_resampled["id"] == j].where(
                    ds_obs_resampled.loc[ds_obs_resampled["id"] == j].notnull()
                ).plot(
                    x="time",
                    y="head",
                    legend=False,
                    ax=axs[0][0],
                    color="r",
                    linestyle="solid",
                    linewidth=1,
                )

            # layout
            title_name = j
            axs[0][0].set_title(title_name)
            axs[0][0].set_xlim([sdate, edate])

            # add text box
            textstr = "\n".join(
                (
                    r"$\mathrm{Layer:}%d$" % (layer,),
                    r"$\mathrm{Depth(mMSL):}%.2f$" % (depth,),
                )
            )

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            # place a text box in upper left in axes coords
            axs[0][0].text(
                1.05,
                0.98,
                textstr,
                transform=axs[0][0].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

            ## Plot map on different axis
            imod.visualize.plot_map(
                dem,
                colors="terrain",
                levels=np.linspace(-10, 10, 20),
                figsize=[12, 7],
                ax=axs[1, 0],
                fig=fig,
            )
            axs[1, 0].scatter(x=x_value, y=y_value, c="k", s=20, zorder=5)

            # fig.tight_layout()
            if mtype == "head":
                fig.suptitle("Modelled and measured heads (mMSL)")
                l = mlines.Line2D([], [], color="blue", label="Modelled head")
                m = mlines.Line2D([], [], color="red", label="Measured head")
            if mtype == "conc":
                fig.suptitle("Modelled chloride concentration (g/L)")
                l = mlines.Line2D([], [], color="blue", label="Modelled concentration")
                m = mlines.Line2D([], [], color="red", label="Measured concentration")
            fig.tight_layout(pad=5.0)
            fig.legend(handles=[l, m], loc="upper right")
            fig.savefig(
                f"data/5-visualization/{modelname}/timeseries/{mtype}/modelled_obs_{j}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


""" VALIDATE HEADS """
if compare_measurements == True:
    # Maak dataframe met measurements en modelresults
    df_model = pd.DataFrame(index=model_ds["id"].unique())
    df_model["modelled_head_mean"] = model_ds.groupby("id")["head"].mean()
    df_model["modelled_head_p10"] = model_ds.groupby("id")["head"].quantile(0.1)
    df_model["modelled_head_p90"] = model_ds.groupby("id")["head"].quantile(0.9)
    df_model = df_model.reset_index()
    df_measu = pd.DataFrame(index=gdf["id"].unique())
    df_measu["head_mean"] = gdf.reset_index(drop=True).groupby("id")["head"].mean()
    df_measu["head_p10"] = (
        gdf.reset_index(drop=True).groupby("id")["head"].quantile(0.1)
    )
    df_measu["head_p90"] = (
        gdf.reset_index(drop=True).groupby("id")["head"].quantile(0.9)
    )
    df_measu["filt_top"] = gdf.reset_index(drop=True).groupby("id")["filt_top"].mean()
    df_measu["filt_middle"] = (
        gdf.reset_index(drop=True).groupby("id")["filt_depth"].mean()
    )
    df_measu["x"] = gdf.reset_index(drop=True).groupby("id")["x"].mean()
    df_measu["y"] = gdf.reset_index(drop=True).groupby("id")["y"].mean()
    df_measu = df_measu.reset_index()
    # df_measu['y'] = gdf.reset_index(drop=True).groupby("id")["y"].mean()
    df = pd.merge(df_model, df_measu, on="index", how="inner")

    df["diff"] = df["modelled_head_mean"] - df["head_mean"]
    df["absdiff"] = np.abs(df["diff"])

    # Create folder to save to
    pathlib.Path(f"data/5-visualization/{modelname}/validate_heads").mkdir(
        exist_ok=True, parents=True
    )

    df.to_csv(f"data/5-visualization/{modelname}/validate_heads/head-results.csv")
    df[df["filt_top"] < -10.0].to_csv(
        f"data/5-visualization/{modelname}/validate_heads/head-deep.csv"
    )
    df[df["filt_top"] >= -10.0].to_csv(
        f"data/5-visualization/{modelname}/validate_heads/head-shallow.csv"
    )

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
    cmap = matplotlib.colormaps.get_cmap(colors)
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
    # ctx.add_basemap(ax) ## basemap werkt niet meer. heb geen tijd/prioriteit om te debuggen

    # Add colorbar
    settings_cbar = {"ticks": levels, "extend": "both"}
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    # Plot settings
    ax.set_title("Absolute difference between modelled and observed head")
    fig.tight_layout()

    # save fig
    # fig.show()
    fig.savefig(
        f"data/5-visualization/{modelname}/validate_heads/spatial absolute difference heads.png",
        dpi=300,
    )

    # Plotting relative difference
    levels = [-3, -2, -1, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 1, 2, 3]
    colors = "RdBu"

    # color bar magic. do not touch
    nlevels = len(levels)
    cmap = matplotlib.colormaps.get_cmap(colors)
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
    # ctx.add_basemap(ax) ## basemap werkt niet meer. heb geen tijd/prioriteit om te debuggen

    # Add colorbar
    settings_cbar = {"ticks": levels, "extend": "both"}
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad="5%")
    fig.colorbar(cbar, cmap=cmap, norm=norm, cax=cbar_ax, **settings_cbar)

    # Plot settings
    ax.set_title("Difference between modelled and observed head")
    fig.tight_layout()

    # save fig
    # fig.show()
    fig.savefig(
        f"data/5-visualization/{modelname}/validate_heads/spatial difference heads.png",
        dpi=300,
    )
