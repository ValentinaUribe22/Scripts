import os
import numpy as np
import pandas as pd
from pathlib import Path
import imod
import xarray as xr
import warnings

# Make folders for corrected  files
output_path_riv = Path("P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/RIV_CORRECTED")
output_path_riv.mkdir(exist_ok=True)
warnings.filterwarnings("ignore", message="The `squeeze` kwarg to GroupBy is being removed")

# Load all RIV_T1_COND layers into 3D array
path_riv_t1 = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RIV")
riv_files_t1_cond = sorted(path_riv_t1.glob("RIV_T1_RIV_COND*.IDF"))
riv_layers_t1_cond = imod.idf.open(riv_files_t1_cond)
riv_old_cond_t1 = xr.concat(riv_layers_t1_cond, dim="layer")

# Load all RIV_T1_BOTTOM layers into 3D array
riv_files_t1_bottom = sorted(path_riv_t1.glob("RIV_T1_RIV_BOTTOM*.IDF"))
riv_layers_t1_bottom = imod.idf.open(riv_files_t1_bottom)
riv_old_bottom_t1 = xr.concat(riv_layers_t1_bottom, dim="layer")

# Load all RIV_Hoogpeil layers into 3D array
path_riv_h = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RIV/HOOGPEIL")
riv_files_h = sorted(path_riv_h.glob("RIVLEVEL_HEAD*.IDF"))
riv_layers_h = imod.idf.open(riv_files_h)
riv_old_h = xr.concat(riv_layers_h, dim="layer")

# Load all RIV_Laagpeil layers into 3D array
path_riv_l = Path("P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RIV/LAAGPEIL")
riv_files_l = sorted(path_riv_l.glob("RIVLEVEL_HEAD_*.IDF"))
riv_layers_l = imod.idf.open(riv_files_l)
riv_old_l = xr.concat(riv_layers_l, dim="layer")

# Load new model top elevations (2D array)
path_modeltop = r"P:\11209740-nbracer\Valentina_Uribe\scenarios_files\Island__shape\corrected_files\TOP_CORRECTED\TOP_L1.IDF"
modeltop = imod.idf.open(path_modeltop)
modeltop = modeltop.isel(layer=0)

# Broadcast modeltop (DEM) to 3D shape of river bottom
modeltop_3d = modeltop.expand_dims({'layer': riv_old_bottom_t1.layer})
modeltop_3d = modeltop_3d.broadcast_like(riv_old_bottom_t1)

# Deactivate river cells where river bottom is higher than DEM/top

# Create a spatial mask: any cell that is NaN in any input or where bottom > DEM/top will be masked in all outputs
riv_mask_base = (riv_old_bottom_t1 > modeltop_3d) | riv_old_bottom_t1.isnull()
riv_masked_cond_t1 = riv_old_cond_t1.where(~riv_mask_base)
riv_masked_bottom_t1 = riv_old_bottom_t1.where(~riv_mask_base)
riv_masked_h = riv_old_h.where(~riv_mask_base)
riv_masked_l = riv_old_l.where(~riv_mask_base)
print("Applied mask: only river bottom and DEM/top used for masking all outputs.")


# Save files
cond_path = os.path.join(output_path_riv, f"RIV_T1_RIV_COND.idf")
bottom_path = os.path.join(output_path_riv, f"RIV_T1_RIV_BOTTOM.idf")
head_path_h = os.path.join(output_path_riv, f"HOOGPEIL/RIVLEVEL_HEAD.idf")
head_path_l = os.path.join(output_path_riv, f"LAAGPEIL/RIVLEVEL_HEAD.idf")

imod.idf.save(cond_path, riv_masked_cond_t1)
imod.idf.save(bottom_path, riv_masked_bottom_t1)
imod.idf.save(head_path_h, riv_masked_h)
imod.idf.save(head_path_l, riv_masked_l)


