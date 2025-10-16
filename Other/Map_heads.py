import pathlib

import geopandas as gpd
import imod
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr
import os

gxg_results = {}  # Store gxg maps for each model
scenario_pairs = [("reference_MAR", "reference"), ("hd_MAR", "Hd"), ("hn_MAR", "Hn"),("Hd", "reference"),("Hn","reference")]

# Parameters
modelnames = ["reference", "Hd", "Hn", "reference_MAR", "hd_MAR","hn_MAR"] 
year = "2100"

# Paths
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
path_template = f"{external_path}/2-interim/rch_50/template.nc"

# Open templates
like = xr.open_dataset(path_template)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()

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

# Loop over models and years
for modelname in modelnames:
    
       if "MAR" in modelname:
             path = f"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS/{modelname}/head/head_{year}0102*.idf"
             print (path)
       else:
             path = f"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{modelname}/head/head_{year}0102*.idf"
             print (path)
        
       print(f"Processing head in {modelname} for year {year}")
       
       
       # Open head data    
       head = imod.idf.open(path)
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

       #calculate gxg
       gxg_ds = imod.evaluate.calculate_gxg(data)

       gxg = top - gxg_ds
       gxg_results[modelname] = gxg  # Store for later comparison

       colors, levels = imod.visualize.read_imod_legend(f"{external_path}/1-external/legends/grondwaterstand_tov_mv.leg")
    
       
# After processing all modelnames:
for model_a, model_b in scenario_pairs:
    for gxg_type in ["ghg", "glg"]:
        a = gxg_results[model_a][gxg_type]
        b = gxg_results[model_b][gxg_type]
        diff = a - b

        colors, levels = imod.visualize.read_imod_legend(
            f"{external_path}/1-external/legends/residu_detail.leg"
        )

        plt.axis("scaled")
        fig, ax = imod.visualize.plot_map(
            diff.where(area_raster.notnull()),
            colors=colors,
            levels=levels,
            figsize=[15, 10],
            kwargs_colorbar={"label": "Verschil (m)"},
            overlays=overlays,
        )
        
        ax.set_title(f"{gxg_type}, {modelname}, {year}")
        fig.savefig(
        f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/{gxg_type}_{modelname}_{year}.png", dpi=300)
    
        plt.close()

        imod.idf.write(
            f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/gxg/{gxg_type}_{model_a}_diff_{model_b}_{year}.idf",
            diff,
        )