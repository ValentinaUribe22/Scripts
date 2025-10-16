import pathlib
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import imod
import matplotlib as cm

scenario_set = "calculations_all"
pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{scenario_set}").mkdir(exist_ok=True, parents=True)
years = ["2050", "2100"]
scenarios = ["reference","hd_s1", "hn_s1", "hd_s2", "hd_s3"] #"hd_altide", "hn_altide
reference_scenario = "reference"
data_types = {"Interface (m-NAP)":{ "1g/L" : r"interface/interface_1_gL_{scenario}_{year}.idf",
                                    "8g/L" :r"interface/interface_8_gL_{scenario}_{year}.idf",
                                   "Thickness" : r"interface/thickness_interface_{scenario}_{year}.idf"},
            "Seepage (mm/day)": { "wint" : r"seepage_-0.5_All_Terschelling/All_Terschelling_upward_seepage_Average winter seepage {year}.idf",
                                  "summ" : r"seepage_-0.5_All_Terschelling/All_Terschelling_upward_seepage_Average summer seepage {year}.idf",
                                  "avg_year" : r"seepage_-0.5_All_Terschelling/All_Terschelling_upward_seepage_Average seepage {year}.idf"},
            "Salt load (kg/ha/year)":{"wint" : r"salt_load_-0.5/Average salt_load winter {year}.idf",
                                   "summ" : r"salt_load_-0.5/Average salt_load summer {year}.idf",
                                   "avg_year":r"salt_load_-0.5/Average salt_load {year}.idf"},
            "GW depth (m-Gl)": {"glg": r"gxg/glg_{scenario}_{year}.idf",
                                 "ghg": r"gxg/ghg_{scenario}_{year}.idf",
                                 }}

# Function to open all data and store it in a dictionary
def open_data(data_types, scenarios, years):
    print ("Opening data")
    data_dict = {}
    for scenario in scenarios:
        print (scenario)
        data_dict[scenario] = {}  # create sub-dictionary for each scenario

        for data_type, subtypes in data_types.items():
            data_dict[scenario][data_type]= {}  # create sub-dictionary for each scenario


            # This is only due to formating of the file names
            special_years = {2021: "2016-2023",
                             2050: "2047-2054",
                             2100: "2097-2104"}
            special_years_gxg = {2021: "2016_2023",
                                 2050: "2047_2054",
                                 2100: "2097_2104"}

            for year in years:
                data_dict[scenario][data_type][year] = {}  # create sub-dictionary for each year

                for subtype, pattern in subtypes.items():
                    data_dict[scenario][data_type][year][subtype] = {}  # create sub-dictionary for each year


                    if data_type == "Interface (m-NAP)":
                        year_str = year
                    elif data_type == "GW depth (m-Gl)":
                        year_str = special_years_gxg.get(int(year), year)
                    else:
                        year_str = special_years.get(int(year), year)

                    full_pattern = rf"P:/11209740-nbracer/Valentina_Uribe/visualizations/{scenario}/{pattern.format(scenario=scenario, year=year_str)}"

                    data = imod.idf.open(full_pattern)

                    if data_type == "Seepage (mm/day)":
                          data = data.where(data > 0)
                    if data_type == "Salt load (kg/ha/year)":
                        data = data.where(data > 0)


                    #Store directly into the dictionary
                    data_dict[scenario][data_type][year][subtype] = data

    return data_dict

# Run the Open_data function with the specified variables
all_data = open_data(data_types, scenarios, years)

print ("Creating empty database")
# Define subareas and shapefiles for calculations
shp_path = "P:/11209740-nbracer/Valentina_Uribe/Scenarios_files/Shapefiles"
subareas = {
    "All Terschelling" : rf"{shp_path}/aoi.shp",
    "Dunes": rf"{shp_path}/dunes.shp",
    "Eastern Area": rf"{shp_path}/eastern.shp",
    "Polder Area": rf"{shp_path}/polders.shp"}

# Create data frame to save calculations
data_empty = []

# Define all columns in my table
columns=[("GW depth (m-Gl)", "ghg"),
         ("GW depth (m-Gl)", "glg"),
         ("Interface (m-NAP)", "1g/L"),
         ("Interface (m-NAP)", "8g/L"),
         ("Interface (m-NAP)", "Thickness"),
         ("Salt load (kg/ha/year)", "summ"),
         ("Salt load (kg/ha/year)", "wint"),
         ("Salt load (kg/ha/year)", "avg_year"),
         ("Seepage (mm/day)", "summ"),
         ("Seepage (mm/day)", "wint"),
         ("Seepage (mm/day)", "avg_year")]

subarea = list(subareas.keys())
col_index = pd.MultiIndex.from_tuples(
    [(area, indicator, subindicator) for area in subarea for (indicator, subindicator) in columns],
    )

for scenario in scenarios:
    if scenario == "reference":
        continue
    # Define all rows in my table
    row_index = pd.MultiIndex.from_tuples([
    (scenario, year, stat)
    for year in years
    for stat in ["mean", "median"]])


    data = np.full((len(row_index), len(col_index)),np.nan)
    col_index.names = ['subarea', 'data_type', 'subtype']
    row_index.names = ['scenario', 'year', 'stat']
    df = pd.DataFrame(data, index=row_index, columns=col_index)

    data_empty.append(df)


# Concatenate everything into one big table
combined_df = pd.concat(data_empty)

# Sort the index and columns for efficient lookup during processing
combined_df = combined_df.sort_index(axis=0)  # sort rows (index)
combined_df = combined_df.sort_index(axis=1)  # sort columns

top = imod.idf.open(
    r"P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/TOP/*.idf"
).isel(layer=0)


print ("Calculating statistics")
for scenario in scenarios:

    if scenario == "reference":
        continue  # skip reference itself
    print(f"Processing difference: {scenario} - reference")

    for year in years:
       for subarea_name, path_shp in subareas.items():

           # Read polygon and rasterize mask
           gdf_area = gpd.read_file(path_shp)
           area_mask = imod.prepare.spatial.rasterize(gdf_area, like=top)
           polygon_mask = ~area_mask.isnull()

           for data_type, subtypes in data_types.items():
                for subtype in subtypes.keys():

                    # Extract raster data for this scenario/year/type/subtype
                    raster_scenario = all_data[scenario][data_type][year][subtype]
                    raster_reference = all_data["reference"][data_type][year][subtype]

                    if raster_scenario is None or raster_reference is None:
                        print(f"No raster data for {scenario} or {reference_scenario} {year} {data_type} {subtype}")
                        continue

                    # Cell-wise difference
                    if (data_type == "GW depth (m-Gl)"):
                        diff_raster = (raster_scenario - raster_reference) * -1
                    else:
                        diff_raster = raster_scenario - raster_reference

                    # Mask outside polygon
                    diff_masked = diff_raster.where(polygon_mask)

                    # Calculate mean and median ignoring NaNs
                    mean_val = float(diff_masked.mean().values)
                    median_val = float(np.nanmedian(diff_masked.values))

                    # Save mean
                    row_key = (scenario, str(year), "mean")  # e.g. ('hn_max', 2100, 'min')
                    col_key = (subarea_name, data_type, subtype)  # e.g. ('All Terschelling', 'Seepage', 'Avg Year')
                    if row_key in combined_df.index and col_key in combined_df.columns:
                        combined_df.loc[row_key, col_key] = mean_val
                    else:
                        print(f"Index or column not found: row={row_key}, col={col_key}")

                    # Save median
                    row_key = (scenario, str(year), "median")  # e.g. ('hn_max', 2100, 'min')
                    col_key = (subarea_name, data_type, subtype)  # e.g. ('All Terschelling', 'Seepage', 'Avg Year')
                    if row_key in combined_df.index and col_key in combined_df.columns:
                        combined_df.loc[row_key, col_key] = median_val
                    else:
                        print(f"Index or column not found: row={row_key}, col={col_key}")

combined_df = combined_df.round(2)
combined_df.to_excel(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{scenario_set}/data_diff_reference.xlsx")
print("Database saved to excel")


# Make Plots of average values combining all scenarios from 1 set
print ("Generating values plots")

exclude_subtypes = {
    "Seepage (mm/day)": ["avg_year"],
    "Salt load (kg/ha/year)": ["avg_year"]}

for data_type,subtypes in data_types.items():
    subtype = list(subtypes.keys())
    if data_type in exclude_subtypes:
        subtype = [s for s in subtype if s not in exclude_subtypes[data_type]]
    for subarea_name in subareas.keys():

        try:
            # Extract the block of relevant variables
            idx = pd.IndexSlice
            df_block = combined_df.loc[:, idx[subarea_name, data_type, :]]

            # Flatten the columns
            df_block.columns = df_block.columns.get_level_values(2)
            if 'subtype' in df_block.columns:
                df_block = df_block.drop(columns=['subtype'])# Keep only subtype names like 'ghg', 'glg'
            df_block = df_block.reset_index()


            # Melt to long format
            df_long = pd.melt(
                        df_block,
                        id_vars=["scenario", "year", "stat"],
                        value_vars= subtype,
                        var_name="variable",
                        value_name="value"
                    )


            # Pivot to get each scenario as a column
            df_plot = df_long.pivot_table(
                index=["year", "variable"],
                columns="scenario",
                values="value",
                aggfunc="mean"
            ).reset_index()

            # Now plot each variable
            plt.figure(figsize=(10, 6))
            scenarios_in_data = [col for col in df_plot.columns if col not in ['year', 'variable']]

            # Assign a unique color to each scenario
            scenario_colors = plt.cm.get_cmap('tab10', len(scenarios_in_data))
            scenario_color_dict = {scen: scenario_colors(i) for i, scen in enumerate(scenarios_in_data)}

            # Define line styles for each data type (variable)
            # You can expand this dict as needed
            var_linestyles = {
                subtype[0]: '-',
                subtype[1] if len(subtype) > 1 else '': ':',
                subtype[2] if len(subtype) > 2 else '': '--',
            }

            for i, var in enumerate(subtype):
                df_var = df_plot[df_plot["variable"] == var].sort_values("year")

                # Choose a default linestyle if not in dict
                linestyle = var_linestyles.get(var, '-')

                for j, scen in enumerate(scenarios_in_data):
                    plt.plot(
                        df_var["year"],
                        df_var[scen],
                        label=f"{var} ({scen})",
                        color=scenario_color_dict[scen],
                        linestyle=linestyle,
                        linewidth=2.5
                    )


            plt.title(f"Difference {scen} - Reference ({subarea_name})")
            plt.xlabel("Year")
            plt.ylabel(f"Δ {data_type}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            #plt.show()

            # create folder to save to
            _data_type = data_type.split('(')[0].strip().replace(" ", "_")
            _subarea = subarea_name.replace(" ", "_")
            pathlib.Path(f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{scenario_set}/figures_computations").mkdir(
                exist_ok=True, parents=True)

            # Save
            plt.savefig(
                f"P:/11209740-nbracer/Valentina_Uribe/visualizations/{scenario_set}/figures_computations/{_subarea}_{_data_type}.png", dpi=300)
            plt.close()

        except KeyError as e:
            print(f"Missing data for: {data_type} — {e}")
print ("Plots generated")
