# Script to analyze recharge patterns from IDF files
# Creates simple time series graphs for each scenario and year set

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import imod


# --- CONFIGURATION SECTION ---
# 1. Single scenario to analyze
scenario = "reference"

# 2. Single time period to analyze
start_year = 2000
end_year = 2105

# 3. PATHS
data_path_future = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RCH/reference_future/rch"
data_path_historical = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RCH/daily_ref"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"

# --- END CONFIGURATION SECTION ---

def load_recharge_data(scenario, start_year, end_year):
    """Load recharge data for a scenario and time period from multiple paths"""
    print(f"Loading recharge data for {scenario}: {start_year}-{end_year}")

    data_list = []
    paths_to_try = [data_path_historical, data_path_future]

    for year in range(start_year, end_year + 1):
        print(f"  Processing year {year}")

        year_loaded = False
        for data_path in paths_to_try:
            file_pattern = f"{data_path}/rch_{year}*.idf"

            try:
                yearly_data = imod.idf.open(file_pattern)
                data_list.append(yearly_data)
                print(f"    ✓ Loaded from {data_path}")
                year_loaded = True
                break  # Found data, move to next year
            except Exception as e:
                continue  # Try next path

        if not year_loaded:
            print(f"    Warning: No data found for year {year} in any path")

    if not data_list:
        print(f"    No data found for {scenario}")
        return None

    # Concatenate all years
    full_data = xr.concat(data_list, dim="time")

    # Check for and remove duplicates in time dimension
    print(f"  Original data shape: {full_data.shape}")
    time_series = pd.Series(full_data['time'].values)
    duplicates_count = time_series.duplicated().sum()
    print(f"  Found {duplicates_count} duplicate timesteps")

    if duplicates_count > 0:
        print(f"  Sample duplicate times:")
        duplicate_times = time_series[time_series.duplicated(keep=False)]
        for i, dup_time in enumerate(duplicate_times.head(5).values):
            print(f"    {pd.to_datetime(dup_time)}")

    unique_times_index = ~time_series.duplicated(keep='first')
    full_data = full_data.isel(time=unique_times_index)
    print(f"  After removing duplicates: {full_data.shape}")

    return full_data

def create_timeseries_plot(scenario, start_year, end_year, recharge_data):
    """Create a simple time series plot for the recharge data"""

    # Filter for positive recharge only (actual recharge, not discharge)
    positive_recharge = recharge_data.where(recharge_data > 0)

    # Calculate total recharge across all spatial cells for each time step
    total_recharge = positive_recharge.sum(dim=['x', 'y'])

    # Convert time to pandas datetime
    times = pd.to_datetime(total_recharge.time.values)

    # Create DataFrame for export
    df = pd.DataFrame({
        'date': times,
        'total_positive_recharge_mm_day': total_recharge.values
    })

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, total_recharge.values, linewidth=1.5)

    # Add red vertical dotted lines for specific years
    highlight_years = [2016, 2023, 2047, 2054, 2097, 2104]
    for year in highlight_years:
        if year >= start_year and year <= end_year:
            plt.axvline(x=pd.Timestamp(f'{year}-01-01'), color='red', linestyle='--', alpha=0.7, linewidth=1)

    plt.title(f'{scenario} - Total Positive Recharge Time Series ({start_year}-{end_year})')
    plt.xlabel('Time')
    plt.ylabel('Total Positive Recharge (mm/day)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot and data
    output_dir = f"{visualizations_path}/timeseries_plots"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    filename = f"{scenario}_{start_year}_{end_year}_total_positive_recharge_timeseries"

    # Save plot
    plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save Excel file (fixed the path)
    df.to_excel(f"{output_dir}/{filename}.xlsx", index=False)

    print(f"  Saved plot: {output_dir}/{filename}.png")
    print(f"  Saved Excel: {output_dir}/{filename}.xlsx")

# Main execution
if __name__ == "__main__":

    print(f"Processing {scenario} for period {start_year}-{end_year}")

    # Load recharge data
    recharge_data = load_recharge_data(scenario, start_year, end_year)

    if recharge_data is not None:
        # Create time series plot
        create_timeseries_plot(scenario, start_year, end_year, recharge_data)
        print(f"Analysis complete. Plot saved to: {visualizations_path}/timeseries_plots/")
    else:
        print("No data found to process.")