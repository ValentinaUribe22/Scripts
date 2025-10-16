# This script compares time series of groundwater heads from different model scenarios
# at specified observation well locations. It generates plots and saves the data to CSV files.

# --- CONFIGURATION SECTION ---

# Models/scenarios to compare (tuples of (model1, model2, model3, ...))
modelnames = [("reference","reference_s0b"),
              ]  # Add more tuples as needed
model_set = "reference_s0b"
# Years to process
years = ["2050", "2100"]

# Plot colors for each model in the tuple (must match order)
plot_colors = ["black", "red", "blue", "green", "orange"]  # Add more colors as needed

# Well IDs to plot (set USE_SELECTED_IDS = False to use all wells)
USE_SELECTED_IDS = True
selected_ids = [
    "B01C0049001",# In the middle of the dunes layer 9
    "B01C0168001",# In the nothern part of the dunes Layer 14
    "B01D0039001", #In the eastern part of the dunes Layer 16
    "B01D0079001",# Polder area layer 15
    "B01D0091001",# Polder near the dike  layer 17
    "B01C0046003",# Dunes edge area Layer 21
    "B05A0184001"# Coast in the west,
]

# Paths (change these if you move your data)
external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim"
results_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"
path_template = f"{external_path}/rch_50/template.nc"
path_obs_wells = f"{external_path}/validation_data/dino_validatie_data.csv"
path_val_da_vitens = f"{external_path}/validation_data/vitens_validatie_data.csv"
path_waterschap_data = f"{external_path}/validation_data/waterschap_validatie_data.csv"
path_dem = f"{external_path}/rch_50/modeltop.nc"
output_folder_template = f"{visualizations_path}/{model_set}/timeseries"

# --- END CONFIGURATION SECTION ---

import pathlib
import os
import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Open template
like = xr.open_dataset(path_template)["template"]

# Open dataset to plot
print("opening obs wells data")
ds_obs = pd.read_csv(path_obs_wells)

# Vitens data
print("opening vitens data")
val_da_vitens = pd.read_csv(path_val_da_vitens)

# waterschap data
print("opening waterschap data")
val_da_waterschap = pd.read_csv(path_waterschap_data)

# Combine all data points in one dataset
ds_obs = pd.concat([ds_obs, val_da_vitens, val_da_waterschap])

# Make sure only points in the model grid are included
in_bounds = imod.select.points_in_bounds(like, x=ds_obs["x"], y=ds_obs["y"])
ds_obs = ds_obs[in_bounds]

# Computes the average filter depth for each well.
ds_obs = ds_obs.dropna(subset=['filt_top', 'filt_bot'])
ds_obs["filt_depth"] = (ds_obs["filt_top"] + ds_obs["filt_bot"]) / 2

# find the closest model layer to each well’s filter depth.
ds_obs["layer"] = (like.sel(z=ds_obs["filt_depth"].values, method="Nearest").coords["layer"].values)

# Cleans the dataset for plotting
ds_obs = ds_obs[["id", "x", "y", "layer","filt_depth"]].drop_duplicates()
ds_obs = ds_obs.reset_index()[["id", "x", "y", "layer","filt_depth"]].reset_index(drop=True)

# Save shapefile and map ONCE (not per model)
pathlib.Path(output_folder_template).mkdir(exist_ok=True, parents=True)
gdf = gpd.GeoDataFrame(ds_obs, geometry=gpd.points_from_xy(ds_obs["x"], ds_obs["y"]))
gdf["name"] = gdf["id"]
gdf.to_file(f"{output_folder_template}/observation_points.shp")
print(f"Shapefile saved")

# Open DEM
dem = xr.open_dataarray(path_dem)

# Plot observation locations
plt.axis("scaled")
fig, ax = imod.visualize.plot_map(
    dem,
    colors="terrain",
    levels=np.linspace(-10, 10, 20),
    figsize=[20, 12],
)
gdf.plot(column="name", legend=False, color="k", ax=ax, zorder=2.5)
if len(fig.axes) > 1:
    fig.delaxes(fig.axes[1])
ax.set_title("Location observations")
fig.savefig(f"{output_folder_template}/map_wells.png", dpi=300)
plt.close(fig)

# --- Use selected_ids only if flag is set ---
if USE_SELECTED_IDS:
    ds_obs = ds_obs[ds_obs["id"].isin(selected_ids)]
all_data = {}

# Now checks the modelled data
for model_tuple in modelnames:
    scenarios_str = "_".join(model_tuple)
    pathlib.Path(output_folder_template).mkdir(exist_ok=True, parents=True)

    for year in years:

        yearly_data = []
        for mtype in ["head"]:
            print(mtype)
            for month in range(1, 13):
                # Select 2nd, 15th, and 28th day of each month for all years
                day_range = [2, 15, 28]
                for day in day_range:
                    file_path = f"{mtype}_{year}{month:02d}{day:02d}_*.idf"
                    print(file_path)

                    # Open all scenarios for this tuple
                    ds_model = xr.Dataset()
                    for scenario in model_tuple:
                        print(f"Opening head files for {scenario}, {year}")
                        ds_model[scenario] = imod.idf.open(f"{results_path}/{scenario}/head/{file_path}")

                    # Extract modelled data at observation points
                    ds_model_points = imod.select.points_values(
                        ds_model,
                        x=ds_obs["x"],
                        y=ds_obs["y"],
                        layer=ds_obs["layer"],
                    ).to_dataframe().reset_index()

                    ds_model_points["well"] = ds_obs["id"].values
                    ds_model_points = ds_model_points.drop(columns=["index", "dx", "dy", "dz", "z"], errors="ignore")
                    ds_model_points = ds_model_points.reset_index(drop=True)
                    yearly_data.append(ds_model_points)

        if yearly_data:
            full_df_year = pd.concat(yearly_data).reset_index(drop=True)
            full_df_year["time"] = pd.to_datetime(full_df_year["time"], dayfirst=True)
            full_df_year = full_df_year.sort_values(by=["well", "time"])
            output_csv = f"{output_folder_template}/{scenarios_str}_timeseries_{year}.csv"
            full_df_year.to_csv(output_csv, index=False, decimal=',', sep=';')
            all_data[year] = full_df_year

    # Loops through each unique observation well to generate individual plots.
    for year in years:
        for j in selected_ids if USE_SELECTED_IDS else ds_obs["id"].unique():
            ds_model = all_data[year]

            row = 1
            col = 2
            fig, axs = plt.subplots(col, row, figsize=(10, 12), sharey=False, sharex=False, squeeze=False)

            obs_row = ds_obs.loc[ds_obs["id"] == j]
            if obs_row.empty:
                print(f"Warning: Well ID {j} not found in ds_obs, skipping.")
                continue
            layer = obs_row["layer"].values[0]
            depth = obs_row["filt_depth"].values[0]
            x_value = obs_row.x.values[0]
            y_value = obs_row.y.values[0]

            print(j)

            legend_lines = []

            # Plot each scenario conditionally
            for scenario, color in zip(model_tuple, plot_colors):
                if scenario in ds_model.columns:
                    data = ds_model.loc[ds_model["well"] == j][scenario]
                    if not data.dropna().empty:
                        ds_model.loc[ds_model["well"] == j].where(data.notnull()).plot(
                            x="time",
                            y=scenario,
                            color=color,
                            linestyle="solid",
                            linewidth=1,
                            legend=False,
                            ax=axs[0][0],
                        )
                        legend_lines.append(mlines.Line2D([], [], color=color, label=scenario))

            # Compute mean absolute difference between first two scenarios (if at least 2)
            mean_diff = np.nan
            if len(model_tuple) >= 2:
                data1 = ds_model.loc[ds_model["well"] == j, model_tuple[0]]
                data2 = ds_model.loc[ds_model["well"] == j, model_tuple[1]]
                diff_series = (data1 - data2).abs().dropna()
                mean_diff = diff_series.mean()

            # add text box — outside the scenario loop!
            textstr = "\n".join(
                (
                    r"$\mathrm{Layer:}%d$" % (layer,),
                    r"$\mathrm{Depth(mMSL):}%.2f$" % (depth,),
                    f"Mean abs diff: {mean_diff:.3f}" if not np.isnan(mean_diff) else "",
                )
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            axs[0][0].text(
                1.05,
                0.98,
                textstr,
                transform=axs[0][0].transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

            # Plot map on different axis
            imod.visualize.plot_map(
                dem,
                colors="terrain",
                levels=np.linspace(-10, 10, 20),
                figsize=[12, 7],
                ax=axs[1, 0],
                fig=fig,
            )
            axs[1, 0].scatter(x=x_value, y=y_value, c="k", s=20, zorder=5)

            # Titles and legends — also outside the scenario loop!
            fig.suptitle(f"Heads scenarios {scenarios_str} - {year}")
            axs[0][0].legend(
                handles=legend_lines,
                loc="lower left",
                frameon=True,
                facecolor="white",
                fontsize=10
            )

            axs[0][0].set_xlabel("Month")
            axs[0][0].set_ylabel("Groundwater head (m MSL)")

            # Format x-axis to show month names for one-year plots
            import matplotlib.dates as mdates
            axs[0][0].xaxis.set_major_locator(mdates.MonthLocator())
            axs[0][0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

            fig.tight_layout(pad=2.5)
            output_fig = f"{output_folder_template}/{j}_{year}.png"
            fig.savefig(output_fig, dpi=300)
            plt.close(fig)
