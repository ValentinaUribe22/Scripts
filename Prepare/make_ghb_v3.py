import imod
import xarray as xr
import numpy as np
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Paths
path_dikes = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/1-external/aoi/dijken.shp"
path_template= "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim/rch_50/template.nc"
path_template_2d = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim/rch_50/template_2d.nc"

# General head boundary (sea) properties
ghb_conductance = 2500
slope_density_conc = 1.25
density_ref = 1000.0
conc_ghb = 16.0

scenarios = [
    {"year": 2005, "eb": -1.0, "vloed": 0.8, "sealevelrise": 0.00},
    {"year": 2050, "eb": -1.0, "vloed": 0.8, "sealevelrise": 0.38},
    {"year": 2100, "eb": -1.0, "vloed": 0.8, "sealevelrise": 1.24}
    ]

# Load the 3D model grid (This is for a resolution of 50x50)
like = xr.open_dataset(path_template)["template"]
like_2d = xr.open_dataset(path_template_2d)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()

#path_modeltop = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/2-interim/rch_50/modeltop.nc"
path_modeltop = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\corrected_files\TOP_CORRECTED\TOP_L1.IDF"
modeltop = imod.idf.open(path_modeltop)
modeltop = modeltop.isel(layer=0)

# model 3d
bnd = imod.idf.open(r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\corrected_files\IBOUND_CORRECTED\*")

# To check the resolution
# dx = modeltop.x[1] - modeltop.x[0]
# dy = modeltop.y[1] - modeltop.y[0]
# print(f"Resolution in x-direction: {dx.values}")
# print(f"Resolution in y-direction: {dy.values}")

# Load and rasterize model domain (AOI)
path_aoi = "P:/11207941-005-terschelling-model/terschelling-gw-model/data/1-external/aoi/aoi_model_adj.shp"
area = gpd.read_file(path_aoi)
aoi_mask = imod.prepare.spatial.rasterize(
    area, like=modeltop)

# Load and rasterize dikes polygons (for masking)
dikes_shp = gpd.read_file(path_dikes)
# Convert the dikes shapefile into a raster mask aligned with the 2D grid
dikes = imod.prepare.rasterize(dikes_shp, like=like_2d, fill=0.0)

for scenario in scenarios:
    print(f"{scenario}_{scenario['year']}")

    eb = scenario["eb"]
    vloed = scenario["vloed"]
    slr = scenario["sealevelrise"]

    # Apply sealevel rise
    eb = eb + slr
    vloed = vloed + slr

    # Create ghb components
    # Find upper active model layer, where the general head boundary will be applied
    upper_active_layer = imod.select.upper_active_layer(bnd, is_ibound=True) #  find the highest (shalowest) layer to apply GHB

    # keep only the locations below flood line, outside of dikes
    upper_active_layer = upper_active_layer.where(modeltop <= vloed)
    upper_active_layer = upper_active_layer.where(dikes == 0)
    upper_active_layer = upper_active_layer.where(aoi_mask == 1)

    # Find the sea
    is_sea = upper_active_layer.notnull().astype(int)

    # Convert to numpy array for connectivity check
    sea_mask = is_sea.values.astype(np.int32)

    # Label connected regions (4-connectivity)
    labeled_array, num_features = label(sea_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])

    # Identify labels connected to domain edges (assumed open sea boundary)
    edge_labels = set()
    rows, cols = sea_mask.shape

    edge_labels.update(labeled_array[0, :])      # top row
    edge_labels.update(labeled_array[rows-1, :]) # bottom row
    edge_labels.update(labeled_array[:, 0])      # left column
    edge_labels.update(labeled_array[:, cols-1]) # right column

    edge_labels.discard(0)  # remove background

    # Keep only connected regions touching edges
    connected_mask = np.isin(labeled_array, list(edge_labels)).astype(np.int32)

    # Update is_sea with filtered mask (back to xarray with same dims and coords)
    is_sea_filtered = xr.DataArray(connected_mask, dims=is_sea.dims, coords=is_sea.coords)
    is_sea_expanded = is_sea_filtered.broadcast_like(bnd)

    # create the area where the GHB will be present
    is_ghb = xr.full_like(bnd, 1.0)
    is_ghb = is_ghb.where(is_sea_expanded == 1)
    is_ghb = is_ghb.where(is_ghb.layer == upper_active_layer) # this selects exactly one layer per cell to keep the value; all other layers become NaN
    is_ghb = is_ghb.swap_dims({"layer": "z"})

    # Setting concentration, density and conductance
    ghb_conc = xr.full_like(is_ghb, conc_ghb)
    ghb_density = ghb_conc * slope_density_conc + density_ref
    ghb_cond = xr.full_like(is_ghb, 1.0) * ghb_conductance

    ghb_cond = ghb_cond.rename({"layer": "z"})
    ghb_conc = ghb_conc.rename({"layer": "z"})
    ghb_density = ghb_density.rename({"layer": "z"})


    # Tidal cycle
    tide_period = 12  # hours

    # Create time array (e.g., simulate for 2 full tidal cycles = 24 hours)
    time_hours = np.linspace(0, 24, 1440)  # 24 points between 0 and 24 hours
    hourly_points = np.arange(0, 25, 1)

    # Calculate mean water level and amplitude
    mean_sea_level = (eb + vloed) / 2
    amplitude = (vloed - eb) / 2

    # Generate tidal signal (sinusoidal)
    tide_level = mean_sea_level + amplitude * np.sin(2 * np.pi * time_hours / tide_period)
    tide_hourly = mean_sea_level + amplitude * np.sin(2 * np.pi * hourly_points / tide_period)

    ## Plot the tidal cycle
    # plt.figure(figsize=(10, 4))
    # plt.plot(time_hours, tide_level, label='Tide Level')
    # plt.plot(hourly_points, tide_hourly, 'o', color='blue')
    # plt.axhline(mean_level, color='gray', linestyle='--', label='Mean Sea Level')
    # plt.xlabel('Time (hours)')
    # plt.ylabel('Water Level (m)')
    # plt.title('Tidal Cycle Over 24 Hours')
    # plt.xticks(np.arange(0, 25, 6))
    # plt.yticks = np.arange(eb + 0.2, vloed + 0.2, 0.2)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    print("Calculating hourly average of water level...")
    results = []
    for hour in range(24):  # all 24 hours in the tidal cycle
        #print (hour)
        tide_level = tide_hourly[hour]
        #print (tide_level)
        # 3D array with tide_level only where GHB applies and NaN elsewhere
        tide_array = xr.where(is_ghb == 1, tide_level, np.nan)
        print(tide_array)
        # Expand modeltop along 'z' dimension to match is_ghb dims
        modeltop_expanded = modeltop.expand_dims(z=is_ghb.z)
        modeltop_expanded = modeltop_expanded.broadcast_like(tide_array)

        # If tide > modeltop → tide
        # Else → modeltop
        water_level = xr.where(tide_array > modeltop_expanded, tide_array, modeltop_expanded)
        results.append(water_level)

    # mean_level = xr.where(is_ghb == 1, mean_sea_level, np.nan) # Activate this line to compute a single constant mean value
    # ghb_level = mean_level.where(is_ghb == 1) # Activate this line to compute a single constant mean value
    # ghb_level = ghb_level.rename({"layer": "z"})# Activate this line to compute a single constant mean value


    # Comment out the hourly varying GHB to use a constant mean value instead
    # Concatenate all 24 hourly 3D arrays into one 4D array and Average over the time dimension (24 hours)
    stacked = xr.concat(results, dim="time")
    ghb_level = stacked.mean(dim="time")


    # making sure GHB is only present in the right layer
    ghb_cond = ghb_cond.where(is_ghb.notnull())
    ghb_level = ghb_level.where(is_ghb.notnull())
    ghb_conc = ghb_conc.where(is_ghb.notnull())
    ghb_density = ghb_density.where(is_ghb.notnull())


    # asserting to ensure ghb is present in the right cells
    assert ghb_cond.count() == ghb_level.count() == ghb_conc.count() == ghb_density.count()
    assert ghb_cond.shape == ghb_level.shape == ghb_conc.shape == ghb_density.shape, "Shape mismatch"
    assert dikes.shape == like_2d.shape
    assert aoi_mask.shape == modeltop.shape
    print ("done with asserting")

    # Create scenario folder
    scenario_folder = f"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Reference_2/{scenario['year']}"
    os.makedirs(scenario_folder, exist_ok=True)

    cond_path = os.path.join(scenario_folder, f"GHB_{scenario['year']}_GHB_COND.idf")
    level_path = os.path.join(scenario_folder, f"GHB_{scenario['year']}_GHB_LEVEL.idf")

    imod.idf.save(cond_path, ghb_cond)
    imod.idf.save(level_path, ghb_level)

    # # To Check some random cells and see that its ok how GHB is calculated in the file, vs manually
    # # Pick first 50 cells with valid GHB level
    # non_nan_indices = np.argwhere(~np.isnan(ghb_level.values))
    # number = 4
    # selected_cells = non_nan_indices[:number]

    # match_count = 0
    # no_match_count = 0

    # print("Testing random cells:")

    # for idx in selected_cells:
    #     z, y, x = idx
    #     ghb_val = float(ghb_level.isel(z=z, y=y, x=x).values)
    #     top_val = float(modeltop.isel(y=y, x=x).values)

    #     # For each hourly tide level, pick max(top_val, wl)
    #     picked_levels = [wl if wl > top_val else top_val for wl in tide_hourly]
    #     avg_level = np.mean(picked_levels)

    #     # Compare by truncating to two decimals (no rounding)
    #     if int(ghb_val * 100) == int(avg_level * 100):
    #         match_count += 1
    #     else:
    #         no_match_count += 1
    #         print(f"Cell (z={z}, y={y}, x={x}): GHB Level = {ghb_val:.5f}, Model Top = {top_val:.5f}")

    # print(f"\nOut of {number} cells tested:")
    # print(f"  Matches: {match_count}")
    # print(f"  No matches: {no_match_count}")