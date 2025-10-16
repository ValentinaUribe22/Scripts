import imod
import numpy as np
import xarray as xr
import glob
from imod.visualize import plot_map
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

years = ["2005"]#,"2005_6hour"]# "2050", "2100", "2005_6hour", "2050_6hour", "2100_6hour"]
comparisons = [("2005","2005_6hour"),
              ("2050","2050_6hour"),
              ("2100","2100_6hour")]       
level_outputs = {}

for year in years:
    print(year)
    
    # Define file path based on naming convention
    if "6hour" in year:
        if year == "2005_6hour":
            file_path = fr"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/{year}/GHB_T1_GHB_LEVEL*.IDF"
        else:
            clean_year = year.split("_")[0] 
            file_path = fr"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/{clean_year}_6hour/GHB_{clean_year}_GHB_LEVEL*.IDF"
    else:
        file_path = fr"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/{year}/GHB_{year}_GHB_LEVEL*.IDF"
    
    print(file_path)
    idf_files = sorted(glob.glob(file_path))
    layers = []
    for path in idf_files:
        da = imod.idf.open(path)
        layers.append(da)
    
    # Step 3: Combine into one DataArray
    all_files = xr.concat(layers, dim="layer")
    
    # Step 4: Find the index of the layer with the maximum value at each cell
    max_cell = all_files.max(dim='layer')
    max_value_per_cell = max_cell * 100
    
    # Step 5: Define contour levels
    levels_cm = np.arange(-10, 110, 10)  # -10, 0, ..., 100
    levels_cm = np.append(levels_cm, [150])  # Overflow bin for >100 cm
    num_intervals = len(levels_cm) - 1
    
    # Step 6: Define colors
    colors = [(0.8, 0.8, 0.8, 0.3)]  # transparent grey for -10–0 cm
    tab_colors = plt.cm.tab20.colors
    colors += tab_colors[:num_intervals - 2]  # intermediate bins
    colors.append((0.3, 0.1, 0.1, 1.0))  # color for >100 cm
    assert len(colors) == num_intervals
    cmap = ListedColormap(colors)
    
    
    # Set levels and colors based on the data range
    vmin = float(max_value_per_cell.min())
    vmax = float(max_value_per_cell.max())
    
    # Colors list: first color = light grey, rest from tab10
    colors = [(0.8, 0.8, 0.8, 0.3)]  # Transparent grey for 0–10cm
    tab_colors = plt.cm.tab20.colors  # 20 distinct colors
    colors += tab_colors[:num_intervals - 2]  # Get enough distinct colors
    colors.append((0.3, 0.1, 0.1, 1.0))
    assert len(colors) == len(levels_cm) - 1
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    
    # Paths
    external_path= "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
    path_template = r"P:\11207941-005-terschelling-model\terschelling-gw-model\data\2-interim\rch_50/template.nc"
    #path_template = r"P:\11209740-nbracer\Valentina_Uribe\Prepare\templates/template.nc"#_10x10_topfiles
    aoi_shape = f"{external_path}/1-external/aoi/aoi_model_adj.shp"
    aoi = f"{external_path}/1-external/aoi/aoi.shp"
    
    # Open templates
    like = xr.open_dataset(path_template)["template"]
    dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
    zmin = like.zbot.min()
    zmax = like.ztop.max()
    
    # Top
    top = imod.idf.open(r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf")
    
    # Open overlay

    overlays = [
        {"gdf": gpd.read_file(aoi_shape), "edgecolor": "black", "color": "none"},
        {"gdf": gpd.read_file(aoi), "edgecolor": "black", "color": "none"},]
        
    area = gpd.read_file(aoi_shape)
    area_raster = imod.prepare.spatial.rasterize(
        area, like=top.isel(layer=0)
    )
    
    
    # Plot with imod
    plt.axis("scaled")
    xlim = (xmin, xmax)
    ylim = (ymin, ymax)
    fig, ax = imod.visualize.plot_map(
        max_value_per_cell,
        colors=cmap, 
        levels=levels_cm,
        figsize=[15, 10],
        kwargs_colorbar={"label": "Levels (cm NAP)"},
        overlays=overlays,
    )
    
    ax.set_title(f"Levels year {year}")
    plt.show()
    print(vmin)
    print(vmax)
    
    # Save
    fig.savefig(fr"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/input_files/GHB_levels_{year}.png", dpi=300
    )
    
    plt.close()
    
    imod.idf.write(f"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/input_files/GHB_levels_{year}.idf", max_value_per_cell)
    level_outputs[year] = max_value_per_cell

# Make difference maps
for modified, model_6h in comparisons:
    # Compute difference
    diff = level_outputs[model_6h] - level_outputs[modified]
    
    # Compute vmax from valid (non-NaN) values only
    vmax = np.nanmax(np.abs(diff.values))  # Ensures NaNs don't distort scale
    vmax = max(1.0, vmax)  # Set minimum contrast
    
    # Define color levels symmetrically
    levels = np.array([-vmax, -5, -2, -1, 1, 2, 5, vmax])
    levels = np.sort(np.unique(levels))  # Prevent errors if vmax < 5

    # Define color palette with grey center
    cmap_colors = [
        "#67001f",  # dark red
        "#b2182b",  # red
        "#ef8a62",  # light red
        "#d3d3d3",  # light grey (no significant difference)
        "#67a9cf",  # light blue
        "#2166ac",  # blue
        "#053061"   # dark blue
    ]
    custom_cmap = mcolors.ListedColormap(cmap_colors)

    # Fill NaNs with 0 ONLY for visualization
    diff_for_plot = diff.fillna(0)

    # Plot map
    fig, ax = imod.visualize.plot_map(
        diff_for_plot,
        colors=custom_cmap,
        levels=levels,
        figsize=[12, 8],
        kwargs_colorbar={"label": "Water Level Difference (cm NAP)"},
        overlays=overlays,
    )

    ax.set_title(f"Difference Map: {model_6h} - {modified}")
    plt.show()

    # Save figure
    fig.savefig(
        fr"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/2a.tidal_computation/input_files/diff_6hourlevels_{modified}.png",
        dpi=300
    )
    plt.close()
    
    

# # Verify resolution, shape and extent 
model_6h = imod.idf.open(r"P:\11209740-nbracer\Valentina_Uribe\Scenarios_files\2a.tidal_computation\2050_6hour\GHB_2050_GHB_LEVEL_L15.IDF")
modified = imod.idf.open(r"P:\11209740-nbracer\Valentina_Uribe\Scenarios_files\2a.tidal_computation\2050\GHB_2050_GHB_LEVEL_L15.IDF")

dx_6h = model_6h.x[1] - model_6h.x[0]
dy_6h = model_6h.y[0] - model_6h.y[1]  # Note: y usually goes from high to low

dx_mod = modified.x[1] - modified.x[0]
dy_mod = modified.y[0] - modified.y[1]

# Print resolution
print("6-hour model resolution: dx =", dx_6h, ", dy =", dy_6h)
print("Modified model resolution: dx =", dx_mod, ", dy =", dy_mod)

# Print shape
print("6-hour model shape:", model_6h.shape)
print("Modified model shape:", modified.shape)

# Print extent
print("6-hour extent: xmin =", model_6h.x.min(), "xmax =", model_6h.x.max(),
      ", ymin =", model_6h.y.min(), "ymax =", model_6h.y.max())

print("Modified extent: xmin =", modified.x.min(), "xmax =", modified.x.max(),
      ", ymin =", modified.y.min(), "ymax =", modified.y.max())