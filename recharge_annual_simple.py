# Simple script to plot recharge per year set
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import imod
import pathlib

# --- CONFIGURATION ---
modelnames = ["reference", "hd_s1", "hd_s3"]
year_sets = [[2016, 2023], [2047, 2054], [2097, 2104]]

# Paths
base_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"

def load_recharge_for_period(scenario, start_year, end_year):
    """Load recharge data for one time period"""
    print(f"Loading {scenario}: {start_year}-{end_year}")

    # Handle different paths
    if "reference" in scenario:
        data_path = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS"
    else:
        data_path = base_path

    data_list = []
    for year in range(start_year, end_year + 1):
        try:
            file_pattern = f"{data_path}/{scenario}/rch/rch_{year}*.idf"
            yearly_data = imod.idf.open(file_pattern)
            data_list.append(yearly_data)
        except Exception as e:
            print(f"  Could not load {year}: {e}")

    if data_list:
        full_data = xr.concat(data_list, dim="time")
        # Calculate total recharge (sum all cells)
        total_recharge = full_data.sum(dim=['x', 'y'])
        return total_recharge
    return None

# Create one figure for each year set
for i, years in enumerate(year_sets):
    start_year, end_year = years[0], years[1]
    period_name = f"{start_year}-{end_year}"

    print(f"\n=== Creating figure for {period_name} ===")

    plt.figure(figsize=(12, 6))
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Load and plot data for each scenario
    for j, scenario in enumerate(modelnames):
        recharge_data = load_recharge_for_period(scenario, start_year, end_year)

        if recharge_data is not None:
            # Convert to pandas for easier plotting
            times = pd.to_datetime(recharge_data.time.values)
            values = recharge_data.values

            # Plot the time series
            plt.plot(times, values, color=colors[j], linewidth=1.5,
                    label=scenario, alpha=0.8)

    plt.xlabel('Date')
    plt.ylabel('Total Recharge (mm/day × cells)')
    plt.title(f'Recharge Time Series: {period_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    output_dir = f"{visualizations_path}/recharge_annual"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{output_dir}/recharge_{period_name}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Saved: recharge_{period_name}.png")

print("\nAll figures created!")