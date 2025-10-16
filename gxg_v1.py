# This script is used to calculate and visualize groundwater levels (GHG, GLG, GVG) for different scenarios in the Terschelling groundwater model.
# It includes functions to open data, calculate groundwater levels, and generate visualizations.
# It calculates glg, ghg and gvg for different scenarios and compares them.

# --- CONFIGURATION SECTION ---
# 1.List your model scenarios here
modelnames = [
        "reference",
        ]

# 2. Set which extra analyses to run
RUN_COMPARE_SCENARIOS = False    # Compare two scenarios (comp_scen_gxg) different than ref
RUN_COMPARE_GHG_GLG = False        # Compare GHG and GLG for each scenario (comp_ghg_glg)

# 3. or scenario comparison, set which models to compare
COMPARE_SCENARIOS = ("hd_s1", "hd_s3")  # (model_0, model_1) Model1 - Model0

# 4. For GHG-GLG comparison from the same scenario: glg - ghg
COMPARE_GHG_GLG_MODELS = ["reference"]    # List of model names

# 5. PATHS (change these if you move your data)
#base_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
base_path = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS" ##ref from tess
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations/ Tess_ref"
external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
top_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
aoi_path = f"{external_path}/1-external/aoi/aoi.shp"
legend_path = f"{external_path}/1-external/legends/grondwaterstand_tov_mv.leg"
legend_diff_path = f"{external_path}/1-external/legends/residu_detail.leg"

# --- END CONFIGURATION SECTION ---
import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import imod

top = imod.idf.open(top_idf_path).isel(layer=0)
area = gpd.read_file(aoi_path)
area_raster = imod.prepare.spatial.rasterize(area, like=top)
overlays = [{"gdf": gpd.read_file(aoi_path), "edgecolor": "black", "color": "none"}]

# Turn interactive plotting off
plt.ioff()

year_sets = [[2016, 2023, 2021], [2047, 2054, 2050], [2097, 2104, 2100]]


def open_data(data_type, scenario, start_year, end_year, skip_year):
    data_list = []
    print(f"Loading {data_type} for {scenario}: {start_year}-{end_year} (skip {skip_year})")

    for year in range(start_year, end_year+1):
        print(f"Processing year {year}")

        if year != skip_year:
            file_path = f"{data_type}_{year}*.idf"
            full_path = f"{base_path}/{scenario}/{data_type}/{file_path}"
            print(f"  Regular year: {full_path}")
            try:
                data = imod.idf.open(full_path)
                print(f"Loaded {year} - Shape: {data.shape}")
                data_list.append(data)
            except Exception as e:
                print(f"Failed to load {year}: {e}")
                continue
        else:
            print(f"Skip year {year} - loading monthly data")
            monthly_count = 0
            for month in range(1, 13):
                for day in [15, 29]:
                    file_path = f"{data_type}_{year}{month:02d}{day:02d}_*.idf"
                    if month == 2 and day == 29:  # February doesn't have 29th, use March 1st
                        file_path = f"{data_type}_{year}0301_*.idf"
                    full_path = f"{base_path}/{scenario}/{data_type}/{file_path}"
                    try:
                        data = imod.idf.open(full_path)
                        data_list.append(data)
                        monthly_count += 1
                    except Exception as e:
                        print(f"    ✗ Failed to load {file_path}: {e}")
                        continue
            print(f"  ✓ Loaded {monthly_count} monthly files for {year}")

    print(f"Total files loaded: {len(data_list)}")

    if not data_list:
        print(f"Warning: No data loaded for {scenario} {data_type}")
        return None

    # Debug: Check time coordinates before concatenation
    print("Time coordinates check:")
    for i, data in enumerate(data_list[:3]):  # Check first 3 files
        print(f"  File {i}: time shape {data.time.shape}, first time: {data.time.values[0]}")

    concated_data = xr.concat(data_list, dim="time")
    print(f"Concatenated data shape: {concated_data.shape}")

    return concated_data


for modelname in modelnames:
    print(modelname)
    for years in year_sets:
        start_year = years[0]
        end_year = years[1]
        skip_year = years[2]
        print(start_year, end_year)

        head = open_data(
            "head",
            modelname,
            start_year=start_year,
            end_year=end_year,
            skip_year=skip_year,
        )

        data = head.compute()
        print(f"After compute: {data.shape}")

        # Debug duplicate removal
        time_series = pd.Series(data['time'].values)
        duplicates_before = len(time_series)
        unique_times_index = ~time_series.duplicated(keep='first')
        duplicates_count = (~unique_times_index).sum()
        data = data.isel(time=unique_times_index)

        pathlib.Path(f"{visualizations_path}/{modelname}/gxg/").mkdir(
            exist_ok=True, parents=True
        )

        # select upper active layer to plot
        # Select first layer
        upper_active_layer = imod.select.upper_active_layer(data.isel(time=0), is_ibound=False)
        data = data.where(data.layer == upper_active_layer)
        data = data.max("layer")

        #calculate gxg
        gxg_ds = imod.evaluate.calculate_gxg(data)

        # Check if GXG calculation was successful - Fix: GXG doesn't have 'time' dimension
        if gxg_ds is None or len(gxg_ds.data_vars) == 0:
            print(f"Warning: GXG calculation failed for {modelname} {start_year}-{end_year}")
            continue

        gxg = top - gxg_ds

        colors, levels = imod.visualize.read_imod_legend(legend_path)

        for gxg_type in ["ghg", "glg"]:
            print(gxg_type)

            # Check if this GXG type exists in the dataset
            if gxg_type not in gxg:
                print(f"Warning: {gxg_type} not found in GXG dataset")
                continue

            colors, levels = imod.visualize.read_imod_legend(legend_path)
            plt.axis("scaled")
            fig, ax = imod.visualize.plot_map(
                gxg[gxg_type].where(area_raster.notnull()),
                colors=colors,
                levels=levels,
                figsize=[15, 10],
                kwargs_colorbar={"label": "(m-bgl)"},
                overlays=overlays,
            )

            # Apply zoom (adjust these values to your desired zoom level)
            xmin, xmax = 135275 + 2500, 170275 - 2500
            ymin, ymax = 591725 + 2500, 611725 - 2500
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)


            # Calculate max and min values for the color bar and text box
            maxi = gxg[gxg_type].where(area_raster.notnull()).max().values
            mini = gxg[gxg_type].where(area_raster.notnull()).min().values
            print(maxi, "max")
            print(mini, "min")

            # Add text box with maximum and minimum values
            textstr = "\n".join(
            (
            r"$\mathrm{Maximum:}%.2f$" % (maxi,),
            r"$\mathrm{Minimum:}%.2f$" % (mini,),
            )
            )

            # Text box properties
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            # Place the text box on the plot
            ax.text(
            0.83,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
            )
            # Plot AOI
            ax.set_title(f"{gxg_type}, {modelname}, {start_year} - {end_year}")
            fig.savefig(
                f"{visualizations_path}/{modelname}/gxg/{gxg_type}_{modelname}_{start_year}_{end_year}.png", dpi=300,
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close()
            imod.idf.write(
                f"{visualizations_path}/{modelname}/gxg/{gxg_type}_{modelname}_{start_year}_{end_year}.idf", gxg[gxg_type]
            )
            if modelname != "reference":
                try:
                    reference = imod.idf.open(f"{visualizations_path}/reference/gxg/{gxg_type}_reference_{start_year}_{end_year}.idf")
                    diff = (gxg[gxg_type] - reference)*-1

                    # Continue with difference plotting...
                    colors, levels = imod.visualize.read_imod_legend(legend_diff_path)

                    plt.axis("scaled")
                    xlim = (xmin, xmax)
                    ylim = (ymin, ymax)
                    fig, ax = imod.visualize.plot_map(
                            diff.where(area_raster.notnull()),
                            colors=colors,
                            levels=levels,
                            figsize=[15, 10],
                            kwargs_colorbar={"label": "Difference (m)"},
                            overlays=overlays,
                    )

                    # Apply zoom (adjust these values to your desired zoom level)
                    xmin, xmax = 135275 + 2500, 170275 - 2500
                    ymin, ymax = 591725 + 2500, 611725 - 2500
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)

                    # Calculate max and min values for the color bar and text box
                    maxi = diff.where(area_raster.notnull()).max().values
                    mini = diff.where(area_raster.notnull()).min().values
                    print(maxi, "max")
                    print(mini, "min")

                    # Add text box with maximum and minimum values
                    textstr = "\n".join(
                    (
                    r"$\mathrm{Maximum:}%.2f$" % (maxi,),
                    r"$\mathrm{Minimum:}%.2f$" % (mini,),
                    )
                    )

                    # Text box properties
                    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

                    # Place the text box on the plot
                    ax.text(
                    0.83,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,
                    )
                    # Plot AOI
                    ax.set_title(f"{gxg_type} difference {modelname} and Reference, ({start_year} - {end_year})")
                    fig.savefig(
                            f"{visualizations_path}/{modelname}/gxg/{gxg_type}_{modelname}_diff_ref_{start_year}_{end_year}.png", dpi=300,
                    bbox_inches="tight",
                    pad_inches=0
                    )
                    plt.close()
                    imod.idf.write(
                            f"{visualizations_path}/{modelname}/gxg/{gxg_type}_{modelname}_diff_ref_{start_year}_{end_year}.idf", diff
                    )
                except Exception as e:
                    print(f"Warning: Could not load reference data for comparison: {e}")
                    continue

# Figure to compare Gxg of two models
def comp_scen_gxg(model_0, model_1):
    for years in year_sets:
        year =  f"{years[0]}_{years[1]}"
        print (year)

        for gxg_type in ["ghg", "glg"]:
               print(gxg_type)

               gxg_0 = imod.idf.open(f"{visualizations_path}/{model_0}/gxg/{gxg_type}_{model_0}_{year}.idf")
               gxg_1 = imod.idf.open(f"{visualizations_path}/{model_1}/gxg/{gxg_type}_{model_1}_{year}.idf")

               diff = (gxg_1 - gxg_0)*-1

               colors, levels = imod.visualize.read_imod_legend(legend_diff_path)

               plt.axis("scaled")
               fig, ax = imod.visualize.plot_map(
                       diff.where(area_raster.notnull()),
                       colors=colors,
                       levels=levels,
                       figsize=[15, 10],
                       kwargs_colorbar={"label": "Difference (m)"},
                       overlays=overlays,
               )

               # Apply zoom (adjust these values to your desired zoom level)
               xmin, xmax = 135275 + 2500, 170275 - 2500
               ymin, ymax = 591725 + 2500, 611725 - 2500
               ax.set_xlim(xmin, xmax)
               ax.set_ylim(ymin, ymax)

               # Calculate max and min values for the color bar and text box
               maxi = diff.where(area_raster.notnull()).max().values
               mini = diff.where(area_raster.notnull()).min().values
               print(maxi, "max")
               print(mini, "min")

               # Add text box with maximum and minimum values
               textstr = "\n".join(
               (
               r"$\mathrm{Maximum:}%.2f$" % (maxi,),
               r"$\mathrm{Minimum:}%.2f$" % (mini,),
               )
               )

               # Text box properties
               props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

               # Place the text box on the plot
               ax.text(
               0.83,
               0.98,
               textstr,
               transform=ax.transAxes,
               fontsize=12,
               verticalalignment="top",
               bbox=props,
               )

               # Plot AOI
               ax.set_title(f"Difference {gxg_type}, {model_1} - {model_0}, {year}")


               fig.savefig(f"{visualizations_path}/{model_1}/gxg/Diff_{gxg_type}_{model_1}_{model_0}_{year}.png",
                   dpi=300,
                   bbox_inches="tight",
                   pad_inches=0)
               plt.close()


               imod.idf.write(
                   f"{visualizations_path}/{model_1}/gxg/Diff_{gxg_type}_{model_1}_{model_0}_{year}.idf",
                   diff,)

# Function to compare GHG and GLG of the same model (reference) for each year
def comp_ghg_glg(model):
    for years in year_sets:
        year = f"{years[0]}_{years[1]}"
        print(year)

        gxg_0 = "ghg"
        gxg_1 = "glg"

        print(f"Comparing {gxg_1} - {gxg_0}")

        ghg = imod.idf.open(
            f"{visualizations_path}/{model}/gxg/{gxg_0}_{model}_{year}.idf"
        )
        glg = imod.idf.open(
            f"{visualizations_path}/{model}/gxg/{gxg_1}_{model}_{year}.idf"
        )

        diff = glg - ghg

        # Load legend
        colors, levels = imod.visualize.read_imod_legend(legend_diff_path)

        plt.axis("scaled")
        fig, ax = imod.visualize.plot_map(
            diff.where(area_raster.notnull()),
            colors=colors,
            levels=levels,
            figsize=[15, 10],
            kwargs_colorbar={"label": "Difference (m)"},
            overlays=overlays,
        )

        # Apply zoom (adjust these values to your desired zoom level)
        xmin, xmax = 135275 + 2500, 170275 - 2500
        ymin, ymax = 591725 + 2500, 611725 - 2500
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Calculate max and min values for the color bar and text box
        maxi = diff.where(area_raster.notnull()).max().values
        mini = diff.where(area_raster.notnull()).min().values
        print(maxi, "max")
        print(mini, "min")

        # Add text box with maximum and minimum values
        textstr = "\n".join(
            (
                r"$\mathrm{Maximum:}%.2f$" % (maxi,),
                r"$\mathrm{Minimum:}%.2f$" % (mini,),
            )
        )

        # Text box properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        ax.text(
            0.83,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

        # Plot AOI
        ax.set_title(f"Difference GLG - GHG, {model}, {year}")

        # Save figure
        fig.savefig(
            f"{visualizations_path}/{model}/gxg/Diff_glg_ghg_{model}_{year}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

# Run the optional comparisons
if RUN_COMPARE_SCENARIOS:
    try:
        comp_scen_gxg(*COMPARE_SCENARIOS)
    except Exception as e:
        print(f"Error in scenario comparison: {e}")

if RUN_COMPARE_GHG_GLG:
    for model in COMPARE_GHG_GLG_MODELS:
        try:
            comp_ghg_glg(model)
        except Exception as e:
            print(f"Error in GHG-GLG comparison for {model}: {e}")