import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt
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
            data = imod.idf.open(
                rf"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{scenario}/{data_type}/{file_path}"
            )  # Replace with actual loading function
            data_list.append(data)
        else:
            for month in range(1, 13):
                for day in [15, 29]:
                    file_path = f"{data_type}_{year}{month:02d}{day:02d}_*.idf"  # Example pattern
                    if month == 2:
                        file_path = f"{data_type}_{year}0301_*.idf"
                    data = imod.idf.open(
                        rf"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{scenario}/{data_type}/{file_path}"
                    )  # Replace with actual loading function
                    data_list.append(data)

    # #open all skip year data and save it
    # print(f"saving {skip_year}...")
    # skips = imod.idf.open(f"p:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{scenario}/{data_type}/{data_type}_{skip_year}*.idf")
    # skips = skips.assign_coords(z=("layer", like["z"].values))
    # skips = skips.swap_dims({"layer": "z"})
    # skips.to_netcdf(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/{data_type}_{scenario}_{skip_year}.nc")


    concated_data = xr.concat(data_list, dim="time")

    # print("saving gxg years...")
    # tmp = concated_data.assign_coords(z=("layer", like["z"].values))
    # tmp = tmp.swap_dims({"layer": "z"})
    # tmp.to_netcdf(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/{data_type}_{scenario}_{start_year}_{end_year}.nc")

    return concated_data


modelnames = ["reference"]#, "Hd", "Hn"]
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
aoi_shape = f"{external_path}/1-external/aoi/aoi.shp"
path_template = f"{external_path}/2-interim/rch_50/template.nc"

# Open templates
like = xr.open_dataset(path_template)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()


# Turn interactive plotting off
# plt.ioff()

top = imod.idf.open(
    r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
).isel(layer=0)


year_sets = [[2016, 2023, 2021], [2047, 2054, 2050], [2097, 2104, 2100]]

subareaen = {
    "All Terschelling" : f"{external_path}/1-external/aoi/aoi.shp",
    "Recreational Areas": f"{external_path}/1-external/aoi/Recreatieterreinen_Terschelling.shp",
}

for subarea, path_shp in subareaen.items():
    print(subarea)
    gdf_area = gpd.read_file(path_shp)
    area_mask = imod.prepare.spatial.rasterize(
        gdf_area, like=top
    )

    data_present = ~area_mask.isnull()

    # Find indices where data is present
    x_indices, y_indices = data_present.where(data_present, drop=True).indexes.values()

    padding=500
    # Calculate the bounds
    xmin, xmax = x_indices.min() - padding, x_indices.max() + padding
    ymin, ymax = y_indices.max() + padding, y_indices.min() - padding

    # Use `.sel()` to select the zoomed area
    zoomed_area = area_mask.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    for modelname in modelnames:
        print(modelname)
            # Create folder to save to
        pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}").mkdir(
            exist_ok=True, parents=True
        )
        for years in year_sets:
            start_year = years[0]
            end_year = years[1]
            skip_year = years[2]
            print(start_year, end_year)

            bdgflf = open_data(
                "BDGFLF",
                modelname,
                start_year=start_year,
                end_year=end_year,
                skip_year=skip_year,
            ).where(zoomed_area.notnull())
            conc = open_data(
                "conc",
                modelname,
                start_year=start_year,
                end_year=end_year,
                skip_year=skip_year,
            ).where(zoomed_area.notnull())

            # conc_min4 = conc.isel(layer=20).mean("time")
            # conc_min4 = conc_min4.rio.write_crs("EPSG:28992")
            # conc_min4.rio.to_raster(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/conc -4mNAP {modelname} {start_year}-{end_year}.tif")

            # head = open_data(
            #     "head",
            #     modelname,
            #     start_year=start_year,
            #     end_year=end_year,
            #     skip_year=skip_year,
            # )

            # head_deep = head.isel(layer=26).mean("time")
            # head_deep = head_deep.rio.write_crs("EPSG:28992")
            # head_deep.rio.to_raster(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/head deep -16mNAP {modelname} {start_year}-{end_year}.tif")


            # m3 -> mm
            seepage = bdgflf / abs(bdgflf["dx"] * bdgflf["dy"]) * 1000.0
            # seepage = head
            # Top layer
            # upper_active_layer = imod.select.upper_active_layer(seepage.isel(time=0), is_ibound=False)
            # seepage = seepage.where(seepage.layer == upper_active_layer)
            # seepage_top = seepage.max("layer")

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
            # Calculate min, mean, and max for the variable
            for name, data in to_stat.items():
                # Calculate min, mean, and max for the variable
                results[name] = {
                    "min": data.min().values,
                    "mean": data.mean().values,
                    "max": data.max().values
                }

                # Convert the dictionary to a Pandas DataFrame
                df = pd.DataFrame(results).T  # Transpose to have variables as rows
                df.reset_index(inplace=True)
                df.rename(columns={"index": "Variable"}, inplace=True)

                # Save to CSV
                df.to_csv(f'P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{name}_stats_{modelname}_{subarea}_{start_year}_{end_year}.csv')



            # mean_total = mean_total.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/head mean_total {start_year}-{end_year}.tif", mean_total)

            # summer = summer.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/head summer {start_year}-{end_year}.tif", summer)

            # winter = winter.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/head winter {start_year}-{end_year}.tif", winter)

            # Open overlay
            overlays = [
                {"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"}
            ]

            # Get area
            area = gpd.read_file(aoi_shape)
            area_raster = imod.prepare.spatial.rasterize(
                area, like=seepage_top.isel(time=0)
            )


            ## Seepage
            # Plot
            for name, data in to_plot.items():
                print(name)
                data.attrs["unit"] = "mm/day"
                # Read legend
                colors, levels = imod.visualize.read_imod_legend(
                    f"{external_path}/1-external/legends/kwel_mmd.leg"
                )
                fig, ax = imod.visualize.plot_map(
                    data.where(area_raster.notnull()),
                    list(reversed(colors)),
                    levels,
                    overlays=overlays,
                    figsize=[12, 6],
                    basemap=cx.providers.Esri.WorldTopoMap
                )
                ax.set_title(f"Upward- and downward-seepage at -0.5 NAP, {name}")
                ax.tick_params(labelsize=7)
                fig.tight_layout()

                # Save

                fig.savefig(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_upward_downward_seepage_{name}.png",
                    dpi=300,
                )
                fig.clf()
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_upward_downward_seepage_{name}.idf",
                    data,
                )
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_upward_downward_seepage_{name}_aoi.idf",
                    data.where(area_raster.notnull()),
                )
                '''imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_upward_seepage_{name}.idf",
                    data.where(data > 0),
                )
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_upward_seepage_{name}_aoi.idf",
                    data.where(area_raster.notnull()).where(data > 0),
                )
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_downward_seepage_{name}.idf",
                    data.where(data < 0),
                )
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_downward_seepage_{name}_aoi.idf",
                    data.where(area_raster.notnull()).where(data < 0),
                )'''
                # Plot
                levels = np.linspace(0, 2, 10)
                plt.axis("scaled")
                fig, ax = imod.visualize.plot_map(
                    data.where(area_raster.notnull()),
                    "Blues",
                    levels,
                    overlays=overlays,
                    figsize=[12, 6],
                    basemap=cx.providers.Esri.WorldTopoMap
                )
                ax.set_title(f"Seepage, -0.5mNAP, {name}")
                ax.tick_params(labelsize=8)
                fig.tight_layout()

                # Save
                fig.savefig(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/seepage_-0.5_{subarea}/{subarea}_seepage_{name}.png", dpi=300
                )
                fig.clf()

            ## salt_load
            # create folder to save to
            pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/salt_load_-0.5").mkdir(
                exist_ok=True, parents=True
            )

            # Calculate salt_load and convert to kg/d/m2
            salt_load = (bdgflf * conc / abs(bdgflf["dx"] * bdgflf["dy"])).sel(layer=16)

            # upper_active_layer = imod.select.upper_active_layer(salt_load.isel(time=0), is_ibound=False)
            # salt_load = salt_load.where(salt_load.layer == upper_active_layer)
            # salt_load = salt_load.max("layer")

            mean_total = salt_load.mean("time")

            # Select summer (april-september) and winter (oktober-maart) and average last ten years
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
                f"Average salt_load {start_year}-{end_year}": mean_total,
                f"Average salt_load summer {start_year}-{end_year}": summer,
                f"Average salt_load winter {start_year}-{end_year}": winter,
            }

            # mean_total = mean_total.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/conc mean_total {start_year}-{end_year}.tif", mean_total)

            # summer = summer.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/conc summer {start_year}-{end_year}.tif", summer)

            # winter = winter.rio.write_crs("EPSG:28992")
            # imod.formats.rasterio.save(f"c:/Users/Terschelling/terschelling-gw-model/data/5-visualization/to_share/conc winter {start_year}-{end_year}.tif", winter)



            for name, data in to_plot.items():
                # Plot
                levels = np.linspace(0, 50, 10)
                data.attrs["unit"] = r"$kg m^{-1} d^{-1}$"
                plt.axis("scaled")
                fig, ax = imod.visualize.plot_map(
                    data.where(area_raster.notnull()),
                    levels=levels,
                    colors="jet",
                    overlays=overlays,
                    figsize=[12, 6],
                    basemap=cx.providers.Esri.WorldTopoMap
                )
                ax.set_title(f"{name}, -0.5m NAP")
                ax.tick_params(labelsize=8)
                fig.tight_layout()

                # Save
                fig.savefig(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/salt_load_-0.5/{subarea}_{name}.png", dpi=300
                )
                fig.clf()
                imod.idf.write(
                    f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/{modelname}/salt_load_-0.5/{name}.idf", data
                )


# import imod
# import os
# import rioxarray 


# for modelname in ["reference", "Hd", "Hn"]:
#     # Define the path to the folder containing the IDF files
#     folder_path = fr"c:\Users\Terschelling\terschelling-gw-model\data\5-visualization\{modelname}\seepage_-0.5"

#     # Loop over all the files in the folder
#     for file_name in os.listdir(folder_path):
#         # Check if the file is an IDF file
#         if file_name.endswith(".idf"):
#             # Full path to the IDF file
#             file_path = os.path.join(folder_path, file_name)
            
#             # Open the IDF file
#             idf_data = imod.idf.open(file_path)
            
#             # Assign CRS (EPSG:28992)
#             idf_data_crs = idf_data.rio.write_crs("EPSG:28992")
            
#             # Define the output file path (same name but with .tiff extension)
#             output_file = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}_{modelname}.tiff")
            
#             # Save the file as a TIFF
#             idf_data.rio.to_raster(output_file)

#             # Print the status for each file processed
#             print(f"Converted {file_name} to {output_file}")

# print("Conversion complete!")