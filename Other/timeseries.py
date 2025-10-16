import pathlib

import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Paths
path_template = snakemake.input.path_template
path_obs_wells = snakemake.input.path_obs_wells
path_dem = snakemake.input.path_dem
path_val_da_vitens = snakemake.input.path_val_da_vitens
path_waterschap_data = snakemake.input.path_waterschap_data

# modelname= "nulmodel_ongekalibreerd0"
# frequency = "W"

# path_template = f"data/2-interim/{modelname}/template.nc"
# path_obs_wells = "data/2-interim/validation_data/dino_validatie_data.csv"
# path_val_da_vitens = "data/2-interim/validation_data/vitens_validatie_data.csv"
# path_waterschap_data = "data/2-interim/validation_data/waterschap_validatie_data.csv"
# path_dem = f"data/2-interim/{modelname}/modeltop.nc"

# ## params
params = snakemake.params
modelname = params["modelname"]
frequency =  params["frequency"]
# modelname= "terschelling_rivs"

# Open template
like = xr.open_dataset(path_template)["template"]

# Open dataset to plot
# dino data
ds_obs = pd.read_csv(path_obs_wells)
# Vitens data
val_da_vitens = pd.read_csv(path_val_da_vitens)
# waterschap data
val_da_waterschap = pd.read_csv(path_waterschap_data)

# Combine datasets
ds_obs = pd.concat([ds_obs, val_da_vitens, val_da_waterschap])

# resample to model frequency
ds_obs["time"] = pd.to_datetime(ds_obs["time"])
ds_obs = ds_obs.set_index("time")
ds_obs = ds_obs.groupby("id").resample(frequency).mean().reset_index()

# Make sure only points in bounds are included
in_bounds = imod.select.points_in_bounds(like, x=ds_obs["x"], y=ds_obs["y"])
ds_obs = ds_obs[in_bounds]

ds_obs.set_index(ds_obs["id"], inplace=True)

ds_obs["filt_depth"] = (ds_obs["filt_top"] + ds_obs["filt_bot"]) / 2
ds_obs["layer"] = like.sel(z=ds_obs["filt_depth"].values, method="Nearest").coords['layer'].values

# Create GDF for plotting purposes
gdf = gpd.GeoDataFrame(ds_obs, geometry=gpd.points_from_xy(ds_obs["x"], ds_obs["y"]))
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
    output_ds[mtype] = xr.open_dataarray(f"data/4-output/{modelname}/{mtype}/{mtype}.nc")

    # Select start and end times for plotting
    sdate = output_ds.time[0].values
    edate = output_ds.time[-1].values

    # Extract modelled data
    model_ds = imod.select.points_values(
        output_ds, x=ds_obs["x"], y=ds_obs["y"], layer=ds_obs["layer"]
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

    model_ds.to_csv(f"data/5-visualization/{modelname}/timeseries/{mtype}_timeseries.csv")

    ids = model_ds.drop_duplicates(subset="id").reset_index()
    id_list = ids["id"]

    for j in id_list:
        row = 1
        col = 2

        fig, axs = plt.subplots(
            col, row, figsize=(12, 14), sharey=False, sharex=False, squeeze=False
        )
        layer = ds_obs.loc[ds_obs["id"] == j]["layer"][0]
        depth = ds_obs.loc[ds_obs["id"] == j]["filt_depth"][0]
        x_value = ds_obs.loc[ds_obs["id"] == j].x[0]
        y_value = ds_obs.loc[ds_obs["id"] == j].y[0]


        print(j)
        if model_ds.loc[model_ds["id"] == j][mtype].dropna().empty == False:
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
            ds_obs.loc[ds_obs["id"] == j].where(ds_obs.loc[ds_obs["id"] == j].notnull()).plot(
                x="time",
                y="head",
                legend=False,
                ax=axs[0][0],
                color="r",
                linestyle="solid",
                linewidth=1,
            )

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
                ax = axs[1,0],
                fig = fig,
            )
            axs[1, 0].scatter(
                x=x_value, y=y_value, c="k", s=20, zorder=5
            )


            #fig.tight_layout()
            if mtype == "head":
                fig.suptitle(f"Modelled and measured heads (mMSL)")
                l = mlines.Line2D([], [], color="blue", label="Modelled head")
                m = mlines.Line2D([], [], color="red", label="Measured head")
            if mtype == "conc":
                fig.suptitle(f"Modelled chloride concentration (g/L)")
                l = mlines.Line2D([], [], color="blue", label="Modelled concentration")
                m = mlines.Line2D([], [], color="red", label="Measured concentration")            
            fig.tight_layout(pad=5.)
            fig.legend(handles=[l,m], loc="upper right")
            fig.savefig(
                f"data/5-visualization/{modelname}/timeseries/{mtype}/modelled_obs_{j}.png",
                dpi=300,
            )
            plt.close()

