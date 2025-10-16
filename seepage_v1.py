# This script processes seepage and salt load data from the Terschelling groundwater model.
# It visualizes data for three different time periods.

# --- CONFIGURATION SECTION ---

# Set to True to run full script, False to only run comparison beetween scenarios that are done
RUN_FULL_PROCESSING = False

# Set to True to run the comparison function
RUN_COMPARISON = True

# Model scenarios
modelnames = ["reference","hd_s1"]

# Years to process (start, end, skip)
year_sets = [
    #[2016, 2023, 2021],
    [2047, 2054, 2050],
    [2097, 2104, 2100]
]

# Paths (change these if you move your data)
external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
results_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"
template_path = f"{external_path}/2-interim/rch_50/template.nc"
aoi_shape = f"{external_path}/1-external/aoi/aoi.shp"
top_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
legend_seepage_path = f"{external_path}/1-external/legends/kwel_mmd.leg"
legend_diff_path = f"{external_path}/1-external/legends/residu_detail_salt.leg"

subareaen = {
    #"Recreational Areas": f"{external_path}/1-external/aoi/Recreatie_Duinrand.shp",
    "All_Terschelling": f"{external_path}/1-external/aoi/aoi.shp",
}

COMPARE_SCENARIOS = [
    ("hd_s1", "reference"),
    ("hn_s1", "reference"),
    ("hd_s3", "hd_s1"),
    # Add more tuples as needed
]
COMPARE_VARIABLES = [
    "seepage_-0.5",   # for seepage
    "salt_load_-0.5", # for salt load
]
SEEPAGE_PATTERNS = [
    "upward_downward_seepage",
    "upward_seepage",           # Exfiltration
    #"downward_seepage",        # Infiltration
]
SEEPAGE_PERIODS = [
    "Average",
    "Average summer",
    "Average winter"
]
SALTLOAD_PERIODS = [
    "Average salt_load",
    "Average salt_load summer",
    "Average salt_load winter"
]

# --- END CONFIGURATION SECTION ---

import pathlib
from pathlib import Path
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import contextily as cx
import imod


def open_data(data_type, scenario, start_year, end_year, skip_year):
    data_list = []
    for year in range(start_year, end_year+1):
        print(year)
        if year != skip_year:
            file_path = f"{data_type}_{year}*.idf"
            path = f"{results_path}/{scenario}/{data_type}/{file_path}"
            data = imod.idf.open(path)
            data_list.append(data)
        else:
            for month in range(1, 13):
                for day in [15, 29]:
                    file_path = f"{data_type}_{year}{month:02d}{day:02d}_*.idf"
                    if month == 2:
                        file_path = f"{data_type}_{year}0301_*.idf"

                    path = f"{results_path}/{scenario}/{data_type}/{file_path}"
                    data = imod.idf.open(path)
                    data_list.append(data)
    concated_data = xr.concat(data_list, dim="time")
    return concated_data


# Open templates
like = xr.open_dataset(template_path)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()

top = imod.idf.open(top_idf_path).isel(layer=0)

def run_full_processing():
    for subarea, path_shp in subareaen.items():
        print(subarea)
        gdf_area = gpd.read_file(path_shp)
        area_mask = imod.prepare.spatial.rasterize(gdf_area, like=top)
        data_present = ~area_mask.isnull()
        x_indices, y_indices = data_present.where(data_present, drop=True).indexes.values()
        padding = 500
        xmin, xmax = x_indices.min() - padding, x_indices.max() + padding
        ymin, ymax = y_indices.max() + padding, y_indices.min() - padding
        zoomed_area = area_mask.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

        for modelname in modelnames:
            print(modelname)
            # Create folder to save to
            outdir = Path(f"{visualizations_path}/{modelname}/seepage_-0.5_{subarea}")
            outdir.mkdir(parents=True, exist_ok=True)

            for years in year_sets:
                start_year = years[0]
                end_year = years[1]
                skip_year = years[2]
                print(start_year, end_year)

                bdgflf = open_data("BDGFLF", modelname, start_year, end_year, skip_year).where(zoomed_area.notnull())
                conc = open_data("conc", modelname, start_year, end_year, skip_year).where(zoomed_area.notnull())

                # m3/day -> mm/day
                seepage = bdgflf / abs(bdgflf["dx"] * bdgflf["dy"]) * 1000.0
                seepage_top = seepage.isel(layer=16)

                # Select time step
                time = seepage_top.coords["time"][-1]
                mean_total = seepage_top.mean("time")

                # Select summer (april-september) and winter (oktober-maart) and average last ten years
                jan = seepage_top.where(seepage_top.time.dt.month == 1, np.nan)
                feb = seepage_top.where(seepage_top.time.dt.month == 2, np.nan)
                mar = seepage_top.where(seepage_top.time.dt.month == 3, np.nan)
                apr = seepage_top.where(seepage_top.time.dt.month == 4, np.nan)
                may = seepage_top.where(seepage_top.time.dt.month == 5, np.nan)
                jun = seepage_top.where(seepage_top.time.dt.month == 6, np.nan)
                jul = seepage_top.where(seepage_top.time.dt.month == 7, np.nan)
                aug = seepage_top.where(seepage_top.time.dt.month == 8, np.nan)
                sep = seepage_top.where(seepage_top.time.dt.month == 9, np.nan)
                oct = seepage_top.where(seepage_top.time.dt.month == 10, np.nan)
                nov = seepage_top.where(seepage_top.time.dt.month == 11, np.nan)
                dec = seepage_top.where(seepage_top.time.dt.month == 12, np.nan)

                summer = (
                    apr.combine_first(may)
                    .combine_first(jun)
                    .combine_first(jul)
                    .combine_first(aug)
                    .combine_first(sep)
                ).mean("time")

                winter = (
                    oct.combine_first(nov)
                    .combine_first(dec)
                    .combine_first(jan)
                    .combine_first(feb)
                    .combine_first(mar)
                ).mean("time")

                to_plot = {
                    f"Average seepage {start_year}-{end_year}": mean_total,
                    f"Average summer seepage {start_year}-{end_year}": summer,
                    f"Average winter seepage {start_year}-{end_year}": winter,
                }

                to_stat = {
                    f"Seepage {start_year}-{end_year}": mean_total,
                    f"Summer seepage {start_year}-{end_year}": summer,
                    f"Winter seepage {start_year}-{end_year}": winter,
                }

                results = {}
                for name, data in to_stat.items():
                    results[name] = {
                        "min": data.min().values,
                        "mean": data.mean().values,
                        "max": data.max().values
                    }
                    df = pd.DataFrame(results).T
                    df.reset_index(inplace=True)
                    df.rename(columns={"index": "Variable"}, inplace=True)
                    df.to_csv(outdir/f"{name}_stats_{modelname}_{subarea}_{start_year}_{end_year}.csv")

                overlays = [
                    {"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none", "linewidth": 0.7}
                ]
                area = gpd.read_file(aoi_shape)
                area_raster = imod.prepare.spatial.rasterize(area, like=seepage_top.isel(time=0))

                # Seepage plots
                for name, data in to_plot.items():
                    print (name)
                    # Only plot if pattern and period are in config
                    for pattern in SEEPAGE_PATTERNS:
                        for period in SEEPAGE_PERIODS:
                            outname = f"{subarea}_{pattern}_{name}.png"
                            outidf = f"{subarea}_{pattern}_{name}.idf"
                            colors, levels = imod.visualize.read_imod_legend(legend_seepage_path)

                            if pattern == "upward_downward_seepage":
                                data_to_save = data.where(area_raster.notnull())
                                title = f"Upward and downward seepage at -0.5 NAP, {name}"
                                plot_colors = list(reversed(colors))
                                plot_levels = levels
                            elif pattern == "upward_seepage":
                                data_to_save = data.where(data > 0)
                                title = f"Seepage at -0.5 NAP, {name}"
                                plot_colors = "Blues"
                                plot_levels = np.linspace(0, 2, 10)
                            else:  # downward seepage
                                data_to_save = data.where(data < 0)
                                title = f"Downward seepage at -0.5 NAP, {name}"
                                plot_colors = "Blues"
                                plot_levels = np.linspace(0, 2, 10)

                            data.attrs["unit"] = "mm/day"

                            fig, ax = imod.visualize.plot_map(
                                data_to_save,
                                plot_colors,
                                plot_levels,
                                overlays=overlays,
                                figsize=[12, 6],
                                kwargs_colorbar={"label": "mm/day"},
                            )
                            ax.set_title(title)
                            ax.tick_params(labelsize=7)
                            fig.tight_layout()
                            fig.savefig(
                                outdir / outname,
                                dpi=300,
                                bbox_inches="tight",
                                pad_inches=0,
                            )

                            plt.close(fig)  # Use plt.close(fig) instead of fig.clf()
                            imod.idf.write(
                                outdir / outidf,
                                data_to_save
                            )

                # Salt load
                # Create folder to save to
                outdirsl = Path(f"{visualizations_path}/{modelname}/salt_load_-0.5")
                outdirsl.mkdir(parents=True, exist_ok=True)

                salt_load_m = (bdgflf * conc / abs(bdgflf["dx"] * bdgflf["dy"])).sel(layer=16)
                conversion_factor = 365 * 10000  # kg/m²/day → kg/ha/year
                salt_load = salt_load_m * conversion_factor
                salt_load = salt_load.clip(min=0)
                salt_load.attrs["unit"] = r"$kg\,ha^{-1}\,yr^{-1}$"
                mean_total_kg = salt_load.mean("time")
                jan = salt_load.where(salt_load.time.dt.month == 1, np.nan)
                feb = salt_load.where(salt_load.time.dt.month == 2, np.nan)
                mar = salt_load.where(salt_load.time.dt.month == 3, np.nan)
                apr = salt_load.where(salt_load.time.dt.month == 4, np.nan)
                may = salt_load.where(salt_load.time.dt.month == 5, np.nan)
                jun = salt_load.where(salt_load.time.dt.month == 6, np.nan)
                jul = salt_load.where(salt_load.time.dt.month == 7, np.nan)
                aug = salt_load.where(salt_load.time.dt.month == 8, np.nan)
                sep = salt_load.where(salt_load.time.dt.month == 9, np.nan)
                oct = salt_load.where(salt_load.time.dt.month == 10, np.nan)
                nov = salt_load.where(salt_load.time.dt.month == 11, np.nan)
                dec = salt_load.where(salt_load.time.dt.month == 12, np.nan)
                summer_kg = (
                    apr.combine_first(may)
                    .combine_first(jun)
                    .combine_first(jul)
                    .combine_first(aug)
                    .combine_first(sep)
                ).mean("time")
                winter_kg = (
                    oct.combine_first(nov)
                    .combine_first(dec)
                    .combine_first(jan)
                    .combine_first(feb)
                    .combine_first(mar)
                ).mean("time")
                to_plot = {
                    f"Average salt_load {start_year}-{end_year}": mean_total_kg,
                    f"Average salt_load summer {start_year}-{end_year}": summer_kg,
                    f"Average salt_load winter {start_year}-{end_year}": winter_kg,
                }


                for name, data in to_plot.items():
                    # 10 intervals between 1 and 500000, so 10 bins
                    levels = np.linspace(1, 50000, 11)
                    cmap = plt.cm.get_cmap("jet", len(levels) - 1)
                    cmap.set_under(color='none')

                    # Mask only values exactly equal to 0 (transparent)
                    plotdata = data.where((area_raster.notnull()) & (data > 0))
                    plotdata = plotdata.where(~np.isnan(plotdata))

                    data.attrs["unit"] = r"$kg\,ha^{-1}\,yr^{-1}$"
                    plt.axis("scaled")
                    fig, ax = imod.visualize.plot_map(
                        plotdata,
                        levels=levels,
                        colors=cmap,
                        overlays=overlays,
                        figsize=[12, 6],
                        kwargs_colorbar={"label": "kg/ha/year"},
                    )

                    ax.set_title(f"{name}, -0.5m NAP")
                    ax.tick_params(labelsize=8)
                    fig.tight_layout()

                    fig.savefig(
                            outdirsl / f"{name}.png",
                            dpi=300,
                            bbox_inches="tight",
                            pad_inches=0,)

                    fig.clf()
                    imod.idf.write(outdirsl/f"{name}.idf", data.where(area_raster.notnull()))

def compare_scenarios(compare_scenarios, compare_variables, subareaen, year_sets, visualizations_path):
    """
    Compare variables (seepage, salt load) between model pairs for each subarea and period,
    only for the patterns and periods specified in SEEPAGE_PATTERNS and SEEPAGE_PERIODS.
    """
    for model1, model0 in compare_scenarios:
        for subarea in subareaen:
            for years in year_sets:
                start_year, end_year, _ = years
                for var in compare_variables:
                    patterns = []
                    if var == "seepage_-0.5":
                        # Only generate patterns that match config
                        for pattern in SEEPAGE_PATTERNS:
                            for period in SEEPAGE_PERIODS:
                                patterns.append(
                                    f"{subarea}_{pattern}_{period} seepage {start_year}-{end_year}.idf"
                                )
                    elif var == "salt_load_-0.5":
                        for period in SALTLOAD_PERIODS:
                            patterns.append(
                                f"{period} {start_year}-{end_year}.idf"
                            )
                    else:
                        continue

                    for pattern in patterns:
                        if var == "salt_load_-0.5":
                            file1 = f"{visualizations_path}/{model1}/{var}/{pattern}"
                            file0 = f"{visualizations_path}/{model0}/{var}/{pattern}"
                            outname = pattern.replace(".idf", f"_diff_{model1}_minus_{model0}.idf")
                            outpng = pattern.replace(".idf", f"_diff_{model1}_minus_{model0}.png")
                        else:
                            file1 = f"{visualizations_path}/{model1}/{var}_{subarea}/{pattern}"
                            file0 = f"{visualizations_path}/{model0}/{var}_{subarea}/{pattern}"
                            outname = pattern.replace(".idf", f"_diff_{model1}_minus_{model0}.idf")
                            outpng = pattern.replace(".idf", f"_diff_{model1}_minus_{model0}.png")

                        if not (os.path.exists(file1) and os.path.exists(file0)):
                            print(f"Skipping {file1} or {file0} (not found)")
                            continue

                        arr1 = imod.idf.open(file1)
                        arr0 = imod.idf.open(file0)
                        diff = arr1 - arr0

                        # Use legend for difference plots
                        colors, levels = imod.visualize.read_imod_legend(legend_diff_path)
                        overlays = [
                           {"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none", "linewidth": 0.7}
                        ]
                        if "salt_load" in pattern:
                            label = "Δ kg/ha/year"
                        else:
                            label = "Δ mm/day"

                        plt.axis("scaled")
                        fig, ax = imod.visualize.plot_map(
                            diff,
                            levels=levels,
                            colors=colors,
                            figsize=[12, 6],
                            kwargs_colorbar={"label": label},
                            overlays=overlays,
                        )
                        ax.set_title(f"Difference {pattern.replace('.idf','')}\n{model1} - {model0}")
                        fig.tight_layout()
                        fig.savefig(
                            f"{visualizations_path}/{model1}/{var}_{subarea}/{outpng}" if var != "salt_load_-0.5"
                            else f"{visualizations_path}/{model1}/{var}/{outpng}",
                            dpi=300, bbox_inches="tight", pad_inches=0
                        )
                        plt.close()
                        imod.idf.write(
                            f"{visualizations_path}/{model1}/{var}_{subarea}/{outname}" if var != "salt_load_-0.5"
                            else f"{visualizations_path}/{model1}/{var}/{outname}",
                            diff
                        )
                        print(f"Saved comparison: {outname}")

if __name__ == "__main__":
    if RUN_FULL_PROCESSING:
        run_full_processing()
    if RUN_COMPARISON:
        compare_scenarios(
            COMPARE_SCENARIOS,
            COMPARE_VARIABLES,
            subareaen,
            year_sets,
            visualizations_path
        )

