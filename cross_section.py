# --- CONFIGURATION SECTION ---

# Models/scenarios to process
modelnames = ["reference"]

# Minimum z value for cross-section plots
zmin = -150

# Base paths
external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
results_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"
ffmpeg_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/EXE/ffmpeg.exe"

# Input data paths
path_cross_section_shape = f"{external_path}/1-external/aoi/cross_section_lines.shp"
path_dem = f"{external_path}/2-interim/rch_50/modeltop.nc"
path_regis = f"{external_path}/1-external/subsurface/regis_v2_2.nc"
path_bot = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/BOT/*.idf"
path_top = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
legend_diff_path = f"{external_path}/1-external/legends/residu_detail.leg"


# Plotting settings
levels_chloride = [0, 0.15, 0.5, 1, 2, 3, 5, 7.5, 10, 16]
cmap_chloride = "jet"

# --- END CONFIGURATION SECTION ---

import pathlib
import subprocess
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap
from datetime import datetime
import imod
import os
import shutil

for modelname in modelnames:
    output_folder = f"{visualizations_path}/{modelname}/conc_crossections"
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Turn interactive plotting off
    plt.ioff()

    # Main output folder for this model
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Bottom
    bot = imod.idf.open(path_bot)
    # Top
    top = imod.idf.open(path_top)

    top = xr.concat(
        [
            top.isel(layer=0).expand_dims("layer"),
            bot.isel(layer=slice(0, 34)),
        ],
        dim="layer",)

    top['layer'] = bot['layer']

    # Dictionary to store the first date of each year
    first_dates = {}

    for filename in os.listdir(f"{results_path}/{modelname}/conc"):
        if "STEADY" not in filename:

            date_str = filename.split('_')[1]
            date = datetime.strptime(date_str, '%Y%m%d')
            year = date.year
            if year not in first_dates or date < first_dates[year]:
                first_dates[year] = date

    # List of earliest dates in 'YYYYMMDD' format
    earliest_dates = sorted([date.strftime('%Y%m%d') for date in first_dates.values()])


    # open raster data
    conc = imod.formats.idf.open(f"{results_path}/{modelname}/conc/conc_{earliest_dates[0]}_L1.idf")
    conc = conc.to_dataset(name="chloride")
    conc = conc.assign_coords(top=top)
    conc = conc.assign_coords(bot=bot)

    _, xmin, xmax, _, ymin, ymax = imod.util.spatial_reference(conc)

    # Define formations
    regis = xr.open_dataset(path_regis).sel(x=slice(xmin, xmax), y=slice(ymax, ymin))
    formations = [fm for fm in regis.formation.values if fm[-2] == "k"]
    formations.append("HLc")
    aquitards = regis.sel(formation=formations)
    aquitards = aquitards.dropna("formation", how="all")
    is_aquitard = aquitards["top"].notnull()
    is_aquitard = is_aquitard.assign_coords(top=aquitards["top"])
    is_aquitard = is_aquitard.assign_coords(bottom=aquitards["bot"])
    is_aquitard = is_aquitard.rename({"formation": "layer"})

    # open shapefile
    gdf = gpd.read_file(path_cross_section_shape)
    linestrings = [ls for ls in gdf.geometry]
    linenames = [ls for ls in gdf.name]

    # Open DEM
    dem = xr.open_dataarray(path_dem)

    # plot map of lines
    plt.axis("scaled")
    fig, ax = imod.visualize.plot_map(
        dem,
        colors="terrain",
        levels=np.linspace(-10, 10, 20),
        figsize=[15, 10],
        kwargs_colorbar={"label": "gw depth (m)"},
    )
    gdf.plot(column="name", legend=True, cmap="Paired", ax=ax)
    fig.delaxes(fig.axes[1])
    ax.set_title("Location cross sections")
    fig.savefig(f"{visualizations_path}/{modelname}/conc_crossections/map.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,)

    aq_sections = [
        imod.select.cross_section_linestring(is_aquitard, ls).compute()
        for ls in linestrings
    ]

    count=0
    count_diff=0
    for year in earliest_dates:
        print(year)

        # open raster data
        conc = imod.formats.idf.open(f"{results_path}/{modelname}/conc/conc_{year}_*.idf")
        conc = conc.to_dataset(name="chloride")
        conc = conc.assign_coords(top=top)
        conc = conc.assign_coords(bottom=bot)

        # Get sections
        sections = [imod.select.cross_section_linestring(conc, ls) for ls in linestrings]

        for species in conc.data_vars:
            for i, (cross_section, aq) in enumerate(zip(sections, aq_sections)):
                print(linenames[i])
                cross_section = cross_section[species]
                cross_section = cross_section.where(cross_section != cross_section.min())

                cross_section = cross_section.where(~(cross_section < 0.0), other=0.0)
                cross_section = cross_section.where(~(cross_section > 15.9), other=15.9)

                cross_section = cross_section.compute()

                output_species_folder = f"{output_folder}/species{species}_{linenames[i]}"
                pathlib.Path(output_species_folder).mkdir(exist_ok=True, parents=True)

                fig, ax = plt.subplots()
                kwargs_aquitards = {
                    "hatch": "/",
                    "edgecolor": "k",
                    "facecolor": "grey",
                    "alpha": 0.3,
                }
                levels = np.array([0, 0.15, 0.5, 1, 2, 3, 5, 7.5, 10, 16])
                cmap = "jet"

                for j, time in enumerate(cross_section.time.values):
                    print(j, time)
                    s = cross_section.sel(time=time)
                    s = s.rename("chloride (g/L)")

                    if j == 0:
                        add_colorbar = True
                    else:
                        add_colorbar = False

                    fig, ax = imod.visualize.cross_section(
                        s,
                        colors=cmap,
                        levels=levels,
                        kwargs_colorbar={
                            "label": "Chloride (g/L)",
                            "whiten_triangles": False,
                        },
                    )
                    # imod.visualize.cross_sections._plot_aquitards(aq, ax, kwargs_aquitards)

                    ax.set_ylim(bottom=zmin, top=10.0)

                    datestring = pd.to_datetime(time).strftime("%Y%m%d")
                    title = f"Section {linenames[i]}, {year[:4]}, {modelname}"
                    ax.set_title(title)
                    plt.tight_layout()
                    fig.savefig(
                        f"{output_species_folder}/{count:03d}.png",
                        dpi=300,
                        bbox_inches="tight",
                        pad_inches=0,)

                    ax.clear()
                    fig.clf()
        count = count+1

        if modelname != "reference":
            # open raster data
            conc_ref = imod.formats.idf.open(f"{results_path}/reference/conc/conc_{year}_*.idf")
            conc_ref = conc_ref.to_dataset(name="chloride")
            conc_ref = conc_ref.assign_coords(top=top)
            conc_ref = conc_ref.assign_coords(bottom=bot)

            diff = conc - conc_ref

            # Get sections
            sections = [imod.select.cross_section_linestring(diff, ls) for ls in linestrings]

            colors, levels = imod.visualize.read_imod_legend(
                f"{external_path}/1-external/legends/residu_detail.leg"
            )
            colors = colors[::-1]

            for species in diff.data_vars:
                for i, (cross_section, aq) in enumerate(zip(sections, aq_sections)):
                    print(linenames[i])
                    cross_section = cross_section[species]
                    cross_section = cross_section.where(cross_section != cross_section.min())

                    cross_section = cross_section.where(~(cross_section < 0.0), other=0.0)
                    cross_section = cross_section.where(~(cross_section > 15.9), other=15.9)

                    cross_section = cross_section.compute()

                    output_species_diff_folder = f"{output_folder}/species{species}_{linenames[i]}_diff_reference"
                    pathlib.Path(output_species_diff_folder).mkdir(exist_ok=True, parents=True)

                    fig, ax = plt.subplots()
                    kwargs_aquitards = {
                        "hatch": "/",
                        "edgecolor": "k",
                        "facecolor": "grey",
                        "alpha": 0.3,
                    }

                    for j, time in enumerate(cross_section.time.values):
                        print(j, time)
                        s = cross_section.sel(time=time)
                        s = s.rename("chloride (g/L)")

                        if j == 0:
                            add_colorbar = True
                        else:
                            add_colorbar = False

                        fig, ax = imod.visualize.cross_section(
                            s,
                            colors=colors,
                            levels=levels,
                            kwargs_colorbar={
                                "label": "Difference in chloride (g/L)",
                                "whiten_triangles": False,
                            },
                        )
                        # imod.visuali e.cross_sections._plot_aquitards(aq, ax, kwargs_aquitards)

                        ax.set_ylim(bottom=zmin, top=10.0)

                        datestring = pd.to_datetime(time).strftime("%Y%m%d")
                        title = f"Section {linenames[i]}, {year[:4]}, {modelname}, difference with reference"
                        ax.set_title(title)
                        plt.tight_layout()
                        fig.savefig(
                            f"{output_species_diff_folder}/{count_diff:03d}.png",
                            dpi=300,
                            bbox_inches="tight",
                            pad_inches=0,)

                        ax.clear()
                        fig.clf()
            count_diff = count_diff+1


        for species in conc.data_vars:
            for i, (cross_section, aq) in enumerate(zip(sections, aq_sections)):
                print(linenames[i])
                area_name = linenames[i].replace(" ", "_")
                output_species_folder = f"{output_folder}/species{species}_{linenames[i]}"
                with imod.util.cd(
                    output_species_folder
                ):
                    output_file = f"{modelname}_{area_name}_species{species}.webm"

                    subprocess.call([
                        ffmpeg_path,
                        "-framerate", "5",
                        "-i", "%03d.png",
                        output_file,
                        "-y"
                    ])
        if modelname != "reference":
            for species in conc.data_vars:
                for i, (cross_section, aq) in enumerate(zip(sections, aq_sections)):
                    print(linenames[i])
                    area_name = linenames[i].replace(" ", "_")
                    output_species_diff_folder = f"{output_folder}/species{species}_{linenames[i]}_diff_reference"
                    with imod.util.cd(
                        output_species_diff_folder
                    ):
                        output_file = f"{modelname}_{area_name}_species{species}.webm"

                        subprocess.call([
                            ffmpeg_path,
                            "-framerate", "5",
                            "-i", "%03d.png",
                            output_file,
                            "-y"
                        ])
