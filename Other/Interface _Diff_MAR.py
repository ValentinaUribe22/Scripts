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


# Parameters
modelnames = ["reference", "Hd", "Hn", "reference_MAR", "hd_MAR","hn_MAR"] 
data_type = ["conc", "head"]
year = "2100"

# Paths
external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
path_template = f"{external_path}/2-interim/rch_50/template.nc"
aoi_shape = f"{external_path}/1-external/aoi/aoi.shp"

# Bottom
bot = imod.idf.open(
    r"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/BOT/*.IDF"
)

# Top
top = imod.idf.open(
    r"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/TOP/*.idf"
)

top = xr.concat(
    [
        top.isel(layer=0).expand_dims("layer"),
        bot.isel(layer=slice(0, 34)),
    ],
    dim="layer",
)

top['layer'] = bot['layer']

# Calculate midpoints (z) for each layer
z = ((top[:-1] + top[1:]) / 2).rename("z")


# Open templates
like = xr.open_dataset(path_template)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()
print (like.z.values)

# Read the area of interest
area = gpd.read_file(aoi_shape)
area_raster = imod.prepare.spatial.rasterize(
    area, like=top.isel(layer=0)
)

data_dict = {}

# Loop over models and years
for modelname in modelnames:
     for dtype in data_type:
         
         
        if "MAR" in modelname:
             path = f"P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS/{modelname}/{dtype}/{dtype}_{year}0102*.idf"
        else:
             path = f"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS/{modelname}/{dtype}/{dtype}_{year}0102*.idf"
         
        if dtype == "conc":
            print(f"Processing {dtype} in {modelname} for year {year}")
            
            # Open concentration data    
            conc = imod.idf.open(path)
       
            # Take average over time
            conc = conc.mean("time")
            conc = conc.assign_coords(z=("layer", like["z"].values))
   
            # Turn interactive plotting off
            plt.ioff()
   
            # Calculate interface concentration
            conc_grens = conc.copy()
            conc_grens = conc_grens.swap_dims({"layer": "z"})
   
            # Resample to higher resolution
            z_new = np.linspace(int(zmin), int(zmax), ((abs(int(zmax - zmin))) * 10 + 1))
            conc_grens = conc_grens.interp(z=z_new, method="linear")
   
            # Interface Calculation
            grens_fresh = conc_grens["z"].where(conc_grens <= 1.0).min("z")
           
            # Save the grens_fresh value in the dictionary
            if dtype not in data_dict:
              data_dict[dtype] = {}
            if modelname not in data_dict[dtype]:
               data_dict[dtype][modelname] = {}
            data_dict[dtype][modelname][year] = grens_fresh
   
            # Plot overlays
            overlays = [{"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"}]
   
            # Get area
            area = gpd.read_file(aoi_shape)
            area_raster = imod.prepare.spatial.rasterize(area, like=conc.isel(layer=1))
   
            
        else:    
            print(f"Processing {dtype} in {modelname} for year {year}")
            
            # Open head data    
            head = imod.idf.open(path).mean("time")
         
            
            # Select the upper active layer (water table)
            upper_active_layer = imod.select.upper_active_layer(head, is_ibound=False)
            head = head.where(head.layer == upper_active_layer)
            
            # Collapse the layer dimension to get 2D (x, y)
            head = head.max("layer").compute()
            
            # Store in the dictionary
            if dtype not in data_dict:
                data_dict[dtype] = {}
            if modelname not in data_dict[dtype]:
                data_dict[dtype][modelname] = {}
            
            
            data_dict[dtype][modelname][year] = head
            
     water_table = data_dict["head"][modelname][year]
     interface = data_dict["conc"][modelname][year]
     
     # Calculate the difference (water table minus interface)
     fw_lens = water_table - interface
     
     # Overwrite the 'conc' entry in data_dict with the new value for 'fw_lens'
     data_dict["conc"][modelname][year] = fw_lens 
           
# At this point, the data_dict contains the data from conc (depth at 1g/L) and head for each model in each year.


# Loop over data type to create figures for both conc and heads

for dtype in data_type:
    if dtype == "conc":
        tit= "Difference Freshwater Lens Thickness"
        lab= "Difference in thickness of Freshwater Lens (m)"
        save= "diff_FW_lens"
        levels = [-15.0,-10.0,-5.0,-1.0,-0.5,-0.1,0.1,0.5,1.0,5.0,10.0,15.0]
    else: 
       tit= "Difference Groundwater Head"
       lab= "Difference in Hydraulic Heads (m)"
       save= "diff_GW_head"
       
            
    comparisons = [
       ("Hd", "reference"),
       ("Hn","reference"),
       ("reference_MAR","hd_MAR"),
       ("reference_MAR","hn_MAR"),
       ("Hd", "hd_MAR",),
       ("Hn", "hn_MAR")]
    
    # Loop over the comparisons and generate figures
    for model_a, model_b in comparisons:
        
        # Compute the difference in the grens_fresh depth at 1g/L
        diff = data_dict[dtype][model_b][year] - data_dict[dtype][model_a][year]
        
         
        # Plot customization
        colors, levels = imod.visualize.read_imod_legend(
        f"{external_path}/1-external/legends/residu_detail.leg"
        )
        if dtype == "conc":
         levels = [-15.0,-10.0,-5.0,-1.0,-0.5,-0.1,0.1,0.5,1.0,5.0,10.0,15.0]
        else:
         levels =  [-5, -4, -3, -2, -1.5,-1,-0,5, 0,0.5, 1,1.5, 2, 3, 4, 5]
         
        # Create figure for the difference map
        plt.axis("scaled")
        fig, ax = imod.visualize.plot_map(
        diff.where(area_raster.notnull()),
        colors=colors,
        levels=levels,
        figsize=[15, 10],
        kwargs_colorbar={"label": lab},
        overlays=overlays,
        )
        
        # Calculate max and min values for the color bar and text box
        maxi = diff.where(area_raster.notnull()).max().values
        mini = diff.where(area_raster.notnull()).min().values
        print(maxi, "max")
        print(mini, "min")
        
        # Add text box with maximum and minimum values
        textstr = "\n".join(
        (
        r"$\mathrm{Maximum:}%d$" % (maxi,),
        r"$\mathrm{Minimum:}%d$" % (mini,),
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
        
        # Set plot title
        ax.set_title(f"{tit} {model_b} - {model_a} {year}")
        
        # Create folder to save to if it doesn't exist
        save_folder = f"P:/11209740-nbracer/Valentina_Uribe/vizualizations/difference_maps/{dtype}"
        pathlib.Path(save_folder).mkdir(exist_ok=True, parents=True)
        
        
        # Save the figure
        fig.savefig(
        f"{save_folder}/{save}_{model_b}_{model_a}_{year}.png",
        dpi=300,
        )
        plt.close()
        
top.isel(layer=0).plot()  # Plot the surface
plt.title('Surface elevation (top layer)')

bot.isel(layer=0).plot()  # Plot the bottom of first layer
plt.title('Bottom of first layer')

water_table = data_dict["head"][modelname][year]
interface = data_dict["conc"][modelname][year]

print("Water table min/max:", water_table.min().values, "/", water_table.max().values)
print("Interface min/max:", interface.min().values, "/", interface.max().values)

print (grens_fresh)
