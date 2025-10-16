import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import imod

modelnames = ["reference"]
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
aoi_shape = f"{external_path}/1-external/aoi/aoi.shp"
top = imod.idf.open(
    r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
).isel(layer=0)

def open_data(data_type, scenario, start_year, end_year, skip_year):
    data_list = []
    for year in range(start_year, end_year+1):
        print(year)
        
        if "reference" in scenario :
           base_path = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS"
        else:
           base_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
    
        if year != skip_year:
            file_path = f"{data_type}_{year}*.idf"
            full_path = f"{base_path}/{scenario}/{data_type}/{file_path}"
            data = imod.idf.open(full_path)
            data_list.append(data)
        else:
            for month in range(1, 13):
                for day in [15, 29]:
                    file_path = f"{data_type}_{year}{month:02d}{day:02d}_*.idf"  # Example pattern
                    if month == 2:
                        file_path = f"{data_type}_{year}0301_*.idf"
                        full_path = f"{base_path}/{scenario}/{data_type}/{file_path}"
                        data = imod.idf.open(full_path)
                        data_list.append(data)
                        
        concated_data = xr.concat(data_list, dim="time")
    return concated_data


area = gpd.read_file(aoi_shape)
area_raster = imod.prepare.spatial.rasterize(
    area, like=top
)

overlays = [{"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"}]

# Turn interactive plotting off
plt.ioff()

year_sets = [[2016, 2023, 2021], [2047, 2054, 2050], [2097, 2104, 2100]]


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

        time_series = pd.Series(data['time'].values)
        unique_times_index = ~time_series.duplicated(keep='first')
        data = data.isel(time=unique_times_index)

        pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{modelname}/gxg/").mkdir(
            exist_ok=True, parents=True
        )

        # select upper active layer to plot
        # Select first layer
        upper_active_layer = imod.select.upper_active_layer(data.isel(time=0), is_ibound=False)
        data = data.where(data.layer == upper_active_layer)
        data = data.max("layer")

        #calculate gxg
        gxg_ds = imod.evaluate.calculate_gxg(data)

        gxg = top - gxg_ds

        colors, levels = imod.visualize.read_imod_legend(f"{external_path}/1-external/legends/grondwaterstand_tov_mv.leg")

        for gxg_type in ["ghg", "glg", "gvg"]:#
            print(gxg_type)

            colors, levels = imod.visualize.read_imod_legend(f"{external_path}/1-external/legends/grondwaterstand_tov_mv.leg")
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
                f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{modelname}/gxg/{gxg_type}_{modelname}_{start_year}_{end_year}.png", dpi=300
            )
            plt.close()
            imod.idf.write(
                f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{modelname}/gxg/{gxg_type}_{modelname}_{start_year}_{end_year}.idf", gxg[gxg_type]
            )
            if modelname != "reference":     
                reference = imod.idf.open(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/reference/gxg/{gxg_type}_reference_{start_year}_{end_year}.idf")

                diff = reference - gxg[gxg_type]

                colors, levels = imod.visualize.read_imod_legend(
                        f"{external_path}/1-external/legends/residu_detail.leg"
                )
    
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
                ax.set_title(f"{gxg_type} difference {modelname} and Reference, {start_year} - {end_year}")
                fig.savefig(
                        f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{modelname}/gxg/{gxg_type}_{modelname}_diff_ref_{start_year}_{end_year}.png", dpi=300
                )
                plt.close()
                imod.idf.write(
                        f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{modelname}/gxg/{gxg_type}_{modelname}_diff_ref_{start_year}_{end_year}.idf", gxg[gxg_type]
                )
                
# Figure to compare Gxg of two models
def comp_scen_gxg(model_0, model_1):
    for years in year_sets:
        year =  f"{years[0]}_{years[1]}"
        print (year)  
        
        for gxg_type in ["ghg", "glg", "gvg"]:
               print(gxg_type)
    
               gxg_0 = imod.idf.open(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{model_0}/gxg/{gxg_type}_{model_0}_{year}.idf")
               gxg_1 = imod.idf.open(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{model_1}/gxg/{gxg_type}_{model_1}_{year}.idf")
           
               diff = gxg_1 - gxg_0
   
               colors, levels = imod.visualize.read_imod_legend(
                       f"{external_path}/1-external/legends/residu_detail.leg")
   
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
               
           
               fig.savefig(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{model_1}/gxg/Diff_{gxg_type}_{model_1}_{model_0}_{year}.png",
                   dpi=300,)
               plt.close()
                   
                   
               imod.idf.write(
                   f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{model_1}/gxg/Diff_{gxg_type}_{model_1}_{model_0}_{year}.idf",
                   diff,) 

# Use this line to generate figures comparing the two scenarios (do model_0, model_1)
#comp_scen_gxg("hn_max", "hn_altide")  