import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import imod


def open_data(data_type, modelname, year):
    data_list = []
    
    print(f"Processing year: {year}")
    
    # Process data for the fixed year
    if "MAR" in modelname:
          data = imod.idf.open(f"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS/{modelname}/head/head_{year}0102*.idf")
         
    else:
          data = imod.idf.open(f"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{modelname}/head/head_{year}0102*.idf")

    data_list.append(data)

    # Concatenate all data along the 'time' dimension
    concated_data = xr.concat(data_list, dim="time")
    return concated_data 



modelnames = ["reference", "Hd", "Hn", "reference_MAR", "hd_MAR"] 


# Parameters
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
aoi_shape = f"{external_path}/1-external/aoi/aoi.shp"

top = imod.idf.open(
    r"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/TOP/*.idf"
).isel(layer=0)

area = gpd.read_file(aoi_shape)
area_raster = imod.prepare.spatial.rasterize(
    area, like=top
)

overlays = [{"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"}]

# Turn interactive plotting off
plt.ioff()

gxg = {}

for modelname in modelnames:
    print(modelname)
    head = open_data("head", modelname, year=2100)
    data = head.compute()

    time_series = pd.Series(data['time'].values)
    unique_times_index = ~time_series.duplicated(keep='first')
    data = data.isel(time=unique_times_index)

    pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/").mkdir(
        exist_ok=True, parents=True
    )
  

    # select upper active layer to plot
    # Select first layer
    upper_active_layer = imod.select.upper_active_layer(data.isel(time=0), is_ibound=False)
    data = data.where(data.layer == upper_active_layer)
    data = data.max("layer")
    
    print(data.values[~np.isnan(data.values)])
    plt.figure(figsize=(10, 6))
    data.plot(cmap="viridis")  # or another colormap you prefer
    plt.title(f"Head in Upper Active Layer — {modelname} (2100)")
    plt.savefig(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/head_upper_{modelname}_2100.png")
    plt.close()

    #calculate gxg
    gxg_ds = imod.evaluate.calculate_gxg(data)
    print(f"GXG output for {modelname}:")
    print(gxg_ds)
    print("Non-NaN count (ghg):", np.count_nonzero(~np.isnan(gxg_ds["ghg"].values)))
    gxg[modelname] = top - gxg_ds

    colors, levels = imod.visualize.read_imod_legend(f"{external_path}/1-external/legends/grondwaterstand_tov_mv.leg")
    
scenario_pairs = [("reference_MAR", "reference"), ("hd_MAR", "Hd"),("Hd", "reference"),("Hn","reference")]

for scenario, reference in scenario_pairs:
    for gxg_type in ["ghg", "glg", "gvg"]:
            
        print(scenario,reference)
        print(gxg_type)
    
        a = gxg[scenario][gxg_type]
        b = gxg[reference][gxg_type]
        diff = a - b
        
        # Debug print
        print("Diff array shape:", diff.shape)
        print("Diff min value:", np.nanmin(diff.values))
        print("Diff max value:", np.nanmax(diff.values))
        print("Sample diff values (non-NaN):", diff.values[~np.isnan(diff.values)][:10])  # first 10 values
        
        masked_diff = diff.where(area_raster.notnull())
        print("Non-NaN count after masking:", np.count_nonzero(~np.isnan(masked_diff.values)))
        
        
        colors, levels = imod.visualize.read_imod_legend(
                    f"{external_path}/1-external/legends/residu_detail.leg")

        plt.axis("scaled")
        fig, ax = imod.visualize.plot_map(
               diff.where(area_raster.notnull()),
               colors=colors,
               levels=levels,
               figsize=[15, 10],
               kwargs_colorbar={"label": "Verschil (m)"},
               overlays=overlays,)
        
        # Plot AOI
        ax.set_title(f"{gxg_type} difference {scenario} and {reference}, 2100")
        fig.savefig(
               f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/diff_{gxg_type}_{scenario}_{reference}_2100.png", dpi=300
        )
        plt.close()
        
        imod.idf.write(f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/diff_{gxg_type}_{scenario}_{reference}_2100.idf", diff)
print(a)
