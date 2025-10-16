import xarray as xr
import numpy as np
import geopandas as gpd
import imod
from pathlib import Path

#This script extracts the dz array from the existing top and bot files in the model _ before correcting the TOP

# bot_files = imod.idf.open(r"P:\11207941-005-terschelling-model\TERSCHELLING_IMOD_MODEL_50X50\BOT/*.IDF")
# top_files = imod.idf.open(r"P:\11207941-005-terschelling-model\TERSCHELLING_IMOD_MODEL_50X50\TOP/*.IDF")
bot_files = imod.idf.open(r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/BOT_CORRECTED/*.IDF")
top_files = imod.idf.open(r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/TOP_CORRECTED/*.idf")
print("Top min/max:", float(top_files.min()), float(top_files.max()))
print("Bot min/max:", float(bot_files.min()), float(bot_files.max()))
NO_DATA = 1e20

# Replace NoData with np.nan
top = top_files.where(top_files != NO_DATA, np.nan)
bot = bot_files.where(bot_files != NO_DATA, np.nan)

n_layers = top_files.sizes["layer"]
dz = np.zeros(n_layers)

for i in range(n_layers):
    # Top of layer i: max of non-zero values
    top_layer = top.isel(layer=i).values
    top_layer[top_layer == 0] = np.nan
    top_val = top_layer.max() if len(top_layer) > 0 else 0

    # Bottom of layer i
    if i < n_layers - 1:
        # bottom = max of top of next layer (ignore zeros)
        next_layer = top.isel(layer=i+1).values
        next_layer[next_layer == 0] = np.nan
        bot_val = next_layer.max() if len(next_layer) > 0 else 0
    else:
        # last layer: min of non-zero bottom
        bot_layer = bot.isel(layer=i).values
        bot_layer[bot_layer == 0] = np.nan
        bot_val = bot_layer.min() if len(bot_layer) > 0 else 0


    dz[i] = abs(top_val - bot_val)  # Correct thickness calculation
    print(f"Layer {i+1}: Top={top_val}, Bottom={bot_val}, Thickness={dz[i]}")

print("Final dz array:", dz)

# # Now we modify the existing top files based on a new maximum top elevation (top_max)
# output_path = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP_CORRECTED")
# output_path.mkdir(exist_ok=True)

# # Load all TOP layers into 3D array
# path_top = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP")
# top_files = sorted(path_top.glob("TOP_*.IDF"))
# top_layers = [imod.idf.open(f).squeeze("layer") for f in top_files]
# top_arr = np.stack([t.values for t in top_layers], axis=0)  # shape: (nlay, ny, nx)

# # Load all BOT layers into 3D array
# path_bot = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/BOT")
# bot_files = sorted(path_bot.glob("BOTM_*.IDF")) # Ensure same order as TOP files
# bot_layers = [imod.idf.open(f).squeeze("layer") for f in bot_files]
# bot_arr = np.stack([b.values for b in bot_layers], axis=0)  # shape: (nlay, ny, nx)

# #Load new TOP
# top_new = imod.idf.open("P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/Elevation_correction/TOP_L1_2050_corrected.idf")
# top_new_clipped = np.minimum(top_new, 10.0)
# top_max_arr = top_new.values  # 2D array

# nlay, nrow, ncol = top_arr.shape

# # Create arrays for new TOP and BOT
# top_new_arr = top_arr.copy()
# bot_new_arr = bot_arr.copy()

# # Compute new TOP for each column
# for j in range(nrow):
#     for i in range(ncol):
#         # 1. Find upper-most active layer <= top_max
#         valid_layers = np.where(top_arr[:, j, i] <= top_max_arr[j, i])[0]
#         if len(valid_layers) == 0:
#             continue  # no active layers, skip

#         upper_layer = valid_layers[-1]

#         # 2. Update the upper-most active layer TOP
#         top_new_arr[upper_layer, j, i] = min(bot_arr[upper_layer, j, i] + dz[upper_layer], top_max_arr[j, i])

#         # 3. Layers above upper-most → inactive (NaN)
#         top_new_arr[upper_layer+1:, j, i] = np.nan
#         bot_new_arr[upper_layer+1:, j, i] = np.nan

