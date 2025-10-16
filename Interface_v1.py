# This script processes groundwater model data to visualize interfaces
# between fresh and saline water, calculates thickness of these interfaces,
# and generates difference maps between model scenarios.

# --- CONFIGURATION SECTION ---
# 1. Model scenarios
modelnames = ["hd_s3"]  # Add more as needed

# 2. Years to process and plot
years = [2021, 2050, 2100]
plotting_years = [2021, 2050, 2100]

# 3. Set to True to run a function to compare two diferent scenarios
RUN_COMPARE_SCENARIOS = True
COMPARE_SCENARIOS = ("hd_s3", "hd_s1")# Tuple of (model_0, model_1) model0 - model 1

VARIABLES_COMPARE = ("thickness_interface", "interface_1_gL", "interface_8_gL")  # Variables to compare

# 4. Paths (change these if you move your data)
external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
results_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"
template_path = f"{external_path}/2-interim/rch_50/template.nc"
aoi_shape_path = f"{external_path}/1-external/aoi/aoi.shp"
top_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
bot_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/BOT/*.IDF"
top_idf_path_S2 = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/TOP_CORRECTED/*.idf"
bot_idf_path_S2 = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/BOT_CORRECTED/*.IDF"
legend_diff_path = f"{external_path}/1-external/legends/residu_detail.leg"

# --- END CONFIGURATION SECTION ---
import pathlib
import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
from shapely import area
import xarray as xr
import os


for modelname in modelnames:
    print(modelname)
    # This is to calculate new z and dz based on the TOP and BOT files
    if modelname == "hd_s2":
        bot = imod.idf.open(bot_idf_path_S2)
        top = imod.idf.open(top_idf_path_S2)
    else :
        bot = imod.idf.open(bot_idf_path)
        top = imod.idf.open(top_idf_path)

    n_layers = top.sizes["layer"]
    dz = np.zeros(n_layers)
    z = np.zeros(n_layers)

    for i in range(n_layers):
        # Top of layer i: max of non-zero values
        top_layer = top.isel(layer=i).values
        top_nonzero = top_layer[top_layer != 0]
        top_val = top_nonzero.max() if len(top_nonzero) > 0 else 0

        # Bottom of layer i
        if i < n_layers - 1:
            # bottom = max of top of next layer (ignore zeros)
            next_layer = top.isel(layer=i+1).values
            next_nonzero = next_layer[next_layer != 0]
            bot_val = next_nonzero.max() if len(next_nonzero) > 0 else 0
        else:
            # last layer: min of non-zero bottom
            bot_layer = bot.isel(layer=i).values
            bot_nonzero = bot_layer[bot_layer != 0]
            bot_val = bot_nonzero.min() if len(bot_nonzero) > 0 else 0

        np.set_printoptions(suppress=True)
        dz[i] = np.round(top_val - bot_val, 2) # Thickness of each layer
        z[i] = np.round((top_val + bot_val) / 2, 2) # Middle of each layer


    # Open templates
    like = xr.open_dataset(template_path)["template"]
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
    zmin = bot.min().compute().item()
    zmax = top.max().compute().item()

    area = gpd.read_file(aoi_shape_path)
    area_raster = imod.prepare.spatial.rasterize(
        area, like=top.isel(layer=0))

    for year in years:
        print(year)
        # Open concentration data
        path = f"{results_path}/{modelname}/conc/conc_{year}0102*.idf"
        conc = imod.idf.open(path)
        print("Loaded concentration data. Shape:", conc.shape)

        # Take average over time
        conc = conc.mean("time")
        conc = conc.assign_coords(z=("layer", z))

        # Turn interactive plotting off
        plt.ioff()

        # Bereken interface conc
        conc_grens = conc.copy()
        conc_grens = conc_grens.swap_dims({"layer": "z"})

        # Resample to higher resolution
        z_new = np.linspace(int(zmin), int(zmax), ((abs(int(zmax - zmin))) * 10 + 1))
        conc_grens = conc_grens.interp(z=z_new, method="linear")

        # Calculating interface
        # For fresh take <= 1.0
        grens_fresh = conc_grens["z"].where(conc_grens <= 1.0).min("z")

        # For middle interface take <= 8.0
        grens_middle = conc_grens["z"].where(conc_grens <= 8.0).min("z")

        grens_model = {"1": grens_fresh, "8": grens_middle}

        # Plot interfaceken
        # Open overlay
        overlays = [{"gdf": gpd.read_file(aoi_shape_path), "edgecolor": "black", "color": "none"}]

        # Get area
        area = gpd.read_file(aoi_shape_path)
        area_raster = imod.prepare.spatial.rasterize(area, like=conc.isel(layer=1))

        # create folder to save to
        pathlib.Path(f"{visualizations_path}/{modelname}/interface").mkdir(
            exist_ok=True, parents=True
        )

        for name, grens in grens_model.items():
            # Plotting interface - 1 and 8 g/L
            plt.axis("scaled")
            fig, ax = imod.visualize.plot_map(
                grens.where(area_raster.notnull()),
                colors="jet",
                levels=np.linspace(-100, 5, 22),
                figsize=[15, 10],
                kwargs_colorbar={"label": "Interface Depth (m-NAP)"},
                overlays=overlays,)

            print(grens.where(area_raster.notnull()).max().values, "max")
            print(grens.where(area_raster.notnull()).min().values, "min")

            maxi = (grens.where(area_raster.notnull()).max().values)
            mini = (grens.where(area_raster.notnull()).min().values)

            # add text box with min and max
            textstr = "\n".join((r"$\mathrm{Maximum:}%d$" % (maxi,),
                                 r"$\mathrm{Minimum:}%d$" % (mini,),))

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.83,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,)


            # Apply zoom (adjust these values to your desired zoom level)
            xmin, xmax = 135275 + 2500, 170275 - 2500
            ymin, ymax = 591725 + 2500, 611725 - 2500
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            plt.tight_layout(pad=0.1)

            # Plot AOI
            ax.set_title(f"Interface {name} g/L, {modelname} {year}")
            if year in plotting_years:
                fig.savefig(
                    f"{visualizations_path}/{modelname}/interface/interface_{name}_gL_{modelname}_{year}.png", dpi=300,
                bbox_inches="tight",
                pad_inches=0,
                )
                plt.close()
            imod.idf.write(
                f"{visualizations_path}/{modelname}/interface/interface_{name}_gL_{modelname}_{year}.idf", grens
            )

            # Diference of reference and interface 1 and 8g/l
            if modelname != "reference":
                reference = imod.idf.open(f"{visualizations_path}/reference/interface/interface_{name}_gL_reference_{year}.idf")

                diff = (grens - reference).where(area_raster.notnull())

                colors, levels = imod.visualize.read_imod_legend(legend_diff_path)
                levels = [-15.0,-10.0,-5.0,-1.0,-0.5,-0.1,0.1,0.5,1.0,5.0,10.0,15.0]

                ##### FIGUUR - VERSCHIL 1 EN 8 G/L
                plt.axis("scaled")
                fig, ax = imod.visualize.plot_map(
                    diff.where(area_raster.notnull()),
                    colors=colors,
                    levels=levels,
                    figsize=[15, 10],
                    kwargs_colorbar={"label": "Difference in Depth of Interface (m)"},
                    overlays=overlays,
                )
                maxi = (diff.where(area_raster.notnull()).max().values)
                mini = (diff.where(area_raster.notnull()).min().values)

                print(maxi, "max")
                print(mini, "min")

                # Apply zoom (adjust these values to your desired zoom level)
                xmin, xmax = 135275 + 2500, 170275 - 2500
                ymin, ymax = 591725 + 2500, 611725 - 2500
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                plt.tight_layout(pad=0.1)

                # add text box
                textstr = "\n".join(
                    (
                        r"$\mathrm{Maximum:}%d$" % (maxi,),
                        r"$\mathrm{Minimum:}%d$" % (mini,),
                    )
                )

                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

                # place a text box in upper left in axes coords
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
                ax.set_title(f"Interface {name} g/L, {modelname} {year}, Difference with Reference")
                if year in plotting_years:
                    fig.savefig(
                        f"{visualizations_path}/{modelname}/interface/interface_{modelname}_difference_reference_{name}_gL_{year}.png",
                        dpi=300,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close()
                imod.idf.write(
                    f"{visualizations_path}/{modelname}/interface/interface_{modelname}_difference_reference_{name}_gL_{year}.idf",
                    diff,
                )


        ## thickness interface
        grens_1 = imod.idf.open(
            f"{visualizations_path}/{modelname}/interface/interface_1_gL_{modelname}_{year}.idf"
        )
        grens_8 = imod.idf.open(
            f"{visualizations_path}/{modelname}/interface/interface_8_gL_{modelname}_{year}.idf"
        )
        thickness_interface = grens_1 - grens_8

        imod.idf.write(
            f"{visualizations_path}/{modelname}/interface/thickness_interface_{modelname}_{year}.idf",
            thickness_interface,)

        plt.axis("scaled")
        xlim = (xmin, xmax)
        ylim = (ymin, ymax)

        fig, ax = imod.visualize.plot_map(
            thickness_interface.where(area_raster.notnull()),
            colors="jet",
            levels=np.linspace(0, 20, 20),
            figsize=[15, 10],
            kwargs_colorbar={"label": "Thickness [m]"},
            overlays=overlays,)

        maxi = (thickness_interface.where(area_raster.notnull()).max().values)
        mini = (thickness_interface.where(area_raster.notnull()).min().values)

        # add text box with min and max
        textstr = "\n".join((r"$\mathrm{Maximum:}%d$" % (maxi,),
                             r"$\mathrm{Minimum:}%d$" % (mini,),))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.83,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,)

        # Apply zoom (adjust these values to your desired zoom level)
        xmin, xmax = 135275 + 2500, 170275 - 2500
        ymin, ymax = 591725 + 2500, 611725 - 2500
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        plt.tight_layout(pad=0.1)
        ax.set_title(f"Thickness Interface, {modelname} {year}")
        if year in plotting_years:
            fig.savefig(
                f"{visualizations_path}/{modelname}/interface/thickness_interface_{modelname}_{year}", dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            )
            plt.close()

        # Figure to compare thickness of model and reference
        if modelname != "reference":
            thickness_ref = imod.idf.open(f"{visualizations_path}/reference/interface/thickness_interface_reference_{year}.idf")
            diff = (thickness_interface - thickness_ref).where(area_raster.notnull())
            colors, levels = imod.visualize.read_imod_legend(legend_diff_path)
            levels = [-15.0,-10.0,-5.0,-1.0,-0.5,-0.1,0.1,0.5,1.0,5.0,10.0,15.0]

            plt.axis("scaled")


            fig, ax = imod.visualize.plot_map(
                diff,
                colors=colors,
                levels=levels,
                figsize=[15, 10],
                kwargs_colorbar={"label": "Difference in thickness of Interface (m)"},
                overlays=overlays,)

            maxi = (diff.max().values)
            mini = (diff.min().values)

            print(maxi, "max")
            print(mini, "min")

            # Apply zoom (adjust these values to your desired zoom level)
            xmin, xmax = 135275 + 2500, 170275 - 2500
            ymin, ymax = 591725 + 2500, 611725 - 2500
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            plt.tight_layout(pad=0.1)

            # add text box
            textstr = "\n".join(
                (
                    r"$\mathrm{Maximum:}%d$" % (maxi,),
                    r"$\mathrm{Minimum:}%d$" % (mini,),
                )
            )

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

            # place a text box in upper left in axes coords
            ax.text(0.83,
                    0.98,
                    textstr,
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=props,)

            # Plot AOI
            ax.set_title(f"Interface Thickness, {modelname} {year}, Difference with Reference")

            if year in plotting_years:
                fig.savefig(f"{visualizations_path}/{modelname}/interface/interface_thickness_{modelname}_difference_reference_{year}.png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,)
                plt.close()
            imod.idf.write(
                f"{visualizations_path}/{modelname}/interface/interface_thickness_{modelname}_difference_reference_{year}.idf",
                diff,)



# Figure to compare thickness of model and reference
def comp_scen_thickness(model_0, model_1):
    for year in years:
        for var in VARIABLES_COMPARE:
            print("Looking for:file 0")
            file_0 = f"{visualizations_path}/{model_0}/interface/{var}_{model_0}_{year}.idf"
            print("Looking for: file_1")
            file_1 = f"{visualizations_path}/{model_1}/interface/{var}_{model_1}_{year}.idf"
            if not (os.path.exists(file_0) and os.path.exists(file_1)):
                print(f"Skipping {var} for {year}: file not found.")
                continue
            data_0 = imod.idf.open(file_0)
            data_1 = imod.idf.open(file_1)

            area = gpd.read_file(aoi_shape_path)
            top = imod.idf.open(top_idf_path).isel(layer=0)
            area_raster = imod.prepare.spatial.rasterize(area, like=top)
            overlays = [{"gdf": gpd.read_file(aoi_shape_path), "edgecolor": "black", "color": "none"}]

            diff = (data_0 - data_1).where(area_raster.notnull())
            colors, levels = imod.visualize.read_imod_legend(legend_diff_path)
            levels = [-15.0,-10.0,-5.0,-1.0,-0.5,-0.1,0.1,0.5,1.0,5.0,10.0,15.0]

            plt.axis("scaled")
            fig, ax = imod.visualize.plot_map(
                diff,
                colors=colors,
                levels=levels,
                figsize=[15, 10],
                kwargs_colorbar={"label": f"Difference in {var.replace('_', ' ').title()} (m)"},
                overlays=overlays,
            )

            maxi = diff.max().values
            mini = diff.min().values

            print(f"{var} {year}: max {maxi}, min {mini}")

            # Apply zoom (adjust these values to your desired zoom level)
            xmin, xmax = 135275 + 2500, 170275 - 2500
            ymin, ymax = 591725 + 2500, 611725 - 2500
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            plt.tight_layout(pad=0.1)

            # add text box
            textstr = "\n".join(
                (
                    r"$\mathrm{Maximum:}%d$" % (maxi,),
                    r"$\mathrm{Minimum:}%d$" % (mini,),
                )
            )
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(0.83, 0.98, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment="top", bbox=props,)

            ax.set_title(f"Difference {var.replace('_', ' ').title()}, {model_0} - {model_1}, {year}")

            if year in plotting_years:
                fig.savefig(
                    f"{visualizations_path}/{model_0}/interface/Diff_{var}_{model_0}_{model_1}_{year}.png",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                plt.close()
            imod.idf.write(
                f"{visualizations_path}/{model_0}/interface/Diff_{var}_{model_0}_{model_1}_{year}.idf",
                diff,
            )


if RUN_COMPARE_SCENARIOS:
    comp_scen_thickness(*COMPARE_SCENARIOS)