# This script calculates the total volume of freshwater and salt in the model domain over time
# In model, you can change which models to include in the comparison, lines and colors
# --- CONFIGURATION SECTION ---
models = [
    {"name": "reference", "label": "Reference", "color": "black"},
    {"name": "hd_s2", "label": "Hd_S2", "color": "red"},
    {"name": "hd_s1", "label": "Hd_S3", "color": "blue"},
]
modelname_set = "S2"

# Choose which dz calculation to use
USE_CALCULATED_DZ = True  # Set to False to use template dz instead

external_path = "P:/11207941-005-terschelling-model/terschelling-gw-model/data"
results_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
template_path = f"{external_path}/2-interim/rch_50/template.nc"
bot_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/BOT/*.IDF"
top_idf_path = r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
top_idf_path_S2 = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/TOP_CORRECTED/*.idf"
bot_idf_path_S2 = r"P:/11209740-nbracer/Valentina_Uribe/scenarios_files/Island__shape/corrected_files/BOT_CORRECTED/*.IDF"
output_base = "P:/11209740-nbracer/Valentina_Uribe/visualizations"
reference_conc_folder = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS/reference/conc/"

porosity = 0.3

# --- END CONFIGURATION SECTION ---

import pathlib
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import imod
import os
from datetime import datetime
import matplotlib.lines as mlines

# Read top and bot files and calculate dz per layer
print("Reading TOP and BOT files...")
top_files = imod.idf.open(top_idf_path_S2)
bot_files = imod.idf.open(bot_idf_path_S2)

# Sort by layer to ensure correct order
top_files = top_files.sortby('layer')
bot_files = bot_files.sortby('layer')

print(f"TOP files shape: {top_files.shape}")
print(f"BOT files shape: {bot_files.shape}")

# Calculate dz for each layer (top - bottom of same layer)
# For the first layer, use model top - first bot
# For other layers, use previous bot - current bot
dz_calculated = xr.zeros_like(bot_files)

for layer_idx in range(len(bot_files.layer)):
    layer = bot_files.layer.values[layer_idx]
    print(f"Calculating dz for layer {layer}")

    if layer_idx == 0:
        # First layer: model top - first bottom
        dz_calculated[dict(layer=layer_idx)] = top_files.isel(layer=0) - bot_files.isel(layer=0)
    else:
        # Other layers: previous bottom - current bottom
        dz_calculated[dict(layer=layer_idx)] = bot_files.isel(layer=layer_idx-1) - bot_files.isel(layer=layer_idx)

# Assign the calculated dz values to the template-like structure
dz_values = dz_calculated.values
print(f"Calculated dz shape: {dz_values.shape}")
print(f"Calculated dz values for ALL layers: {np.mean(dz_values, axis=(1,2))}")

def read_conc(scenario):
    all_conc = []
    print(scenario)
    for year in earliest_dates:
        print(year)
        conc = imod.formats.idf.open(f"{results_path}/{scenario}/conc/conc_{year}_*.idf")
        all_conc.append(conc)

    concated_data = xr.concat(all_conc, dim="time")

    # Choose which dz to use based on flag
    if USE_CALCULATED_DZ:
        print("Using calculated dz from TOP/BOT files")
        concated_data = concated_data.assign_coords(dz=("layer", dz_calculated_mean))
    else:
        print("Using template dz")
        concated_data = concated_data.assign_coords(dz=("layer", dz_template))

    return concated_data

# Open template
like = xr.open_dataset(template_path)["template"]
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(like)
zmin = like.zbot.min()
zmax = like.ztop.max()

# Keep both dz options
dz_template = np.abs(like.dz.values)
dz_calculated_mean = np.abs(dz_values.mean(axis=(1,2)))

print(f"Template dz (all layers): {dz_template}")
print(f"Calculated dz (all layers): {dz_calculated_mean}")



# get dataset chloride
folder_path = reference_conc_folder

first_dates = {}
for filename in os.listdir(folder_path):
    if "STEADY" not in filename:
        date_str = filename.split('_')[1]
        date = datetime.strptime(date_str, '%Y%m%d')
        year = date.year
        if year not in first_dates or date < first_dates[year]:
            first_dates[year] = date

#earliest_dates = sorted([date.strftime('%Y%m%d') for date in first_dates.values()])
earliest_dates = sorted([date.strftime('%Y%m%d') for date in first_dates.values()])


# Create folder to save to
output_folder = f"{output_base}/{modelname_set}/water_volume"
pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

# Read modelled data
conc = xr.Dataset()
for model in models:
    conc[model["name"]] = read_conc(model["name"])

 # Get mass loading over time (Freshwater)
conc_fres = conc.where(conc <= 0.15)
conc_fres = conc_fres.where(conc_fres.isnull(), 1)
data = conc_fres * dx * (dy*-1) * (conc_fres.dz*-1) * porosity * 1000

# Sum all mass
som = data.sum(dim=["layer", "x", "y"])
som_tabel = som.to_dataframe()
som_tabel.to_csv(f"{output_folder}/total_freshwater.csv")
print("Saved csv file successfully")

# Plotting Freshwater
fig, ax = plt.subplots(figsize=[10, 8])
legend_handles = []
for model in models:
    som[model["name"]].plot.line(ax=ax, color=model["color"], label=model["label"])
    legend_handles.append(
        mlines.Line2D([], [], color=model["color"], label=model["label"])
    )

ax.set_title("Total amount freshwater (L), (concentration <= 0.15 mg/L)")
ax.set_ylabel("Total amount of freshwater (L)")
ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.02, 0.02))
fig.tight_layout()
print(f"Saving figure to: {output_folder}/total_amount_fresh_water_0.15mgl.png")
fig.savefig(
    f"{output_folder}/total_amount_fresh_water_0.15mgl.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)
print("Figure saved successfully")

 # Get mass loading over time (Salt)
data = conc * (dx) * (dy*-1) * (conc.dz) * porosity

# Sum all mass
som = data.sum(dim=["layer", "x", "y"])
som_tabel = som.to_dataframe()
som_tabel.to_csv(f"{output_folder}/salt_mass.csv")

# Plotting Total amount of salt
fig, ax = plt.subplots(figsize=[10, 8])
legend_handles = []
for model in models:
    som[model["name"]].plot.line(ax=ax, color=model["color"], label=model["label"])
    legend_handles.append(
        mlines.Line2D([], [], color=model["color"], label=model["label"])
    )
ax.set_title("Total amount of salt (kg)")
ax.set_ylabel("kg")
ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.02, 0.02))
fig.tight_layout()
print(f"Saving figure to: {output_folder}/total_amount_salt.png")
fig.savefig(
    f"{output_folder}/total_amount_salt.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)
print("Figure saved successfully")

 # Get mass loading over time (Brackish)
conc_brac = conc.where(conc >= 0.15)
conc_brac = conc_brac.where(conc_brac <= 5.0)
conc_brac = conc_brac.where(conc_brac.isnull(), 1)
data = conc_brac * (dx) * (dy*-1) * (conc_brac.dz*-1) * porosity * 1000

# Sum all mass
som = data.sum(dim=["layer", "x", "y"])
som_tabel = som.to_dataframe()
som_tabel.to_csv(f"{output_folder}/brackishwater.csv")

# Plotting brackish water volume
fig, ax = plt.subplots(figsize=[10, 8])
legend_handles = []
for model in models:
    som[model["name"]].plot.line(ax=ax, color=model["color"], label=model["label"])
    legend_handles.append(
        mlines.Line2D([], [], color=model["color"], label=model["label"])
    )
ax.set_title("Total amount brackish water (L) (concentration 0.15-5.0 mg/L)")
ax.set_ylabel("Total amount brackish water (L)")
ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.02, 0.02))
fig.tight_layout()
print(f"Saving figure to: {output_folder}/total_amount_brackish_water_0.15-5.0mgl.png")
fig.savefig(
    f"{output_folder}/total_amount_brackish_water_0.15-5.0mgl.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
plt.close(fig)
print("Figure saved successfully")