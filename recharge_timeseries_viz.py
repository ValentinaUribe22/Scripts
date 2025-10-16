# Script to visualize recharge time series
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import imod
import numpy as np
import pathlib

# --- CONFIGURATION ---
modelnames = ["reference", "hd_s1", "hd_s3"]  # Add your scenarios here
year_sets = [[2016, 2023], [2047, 2054], [2097, 2104]]  # Time periods to analyze

# Paths
base_path = "P:/11209740-nbracer/Valentina_Uribe/Terschelling_model/RESULTS"
visualizations_path = "P:/11209740-nbracer/Valentina_Uribe/visualizations"

def load_recharge_data(scenario, start_year, end_year):
    """Load recharge data for a scenario and time period"""
    print(f"Loading recharge data for {scenario}: {start_year}-{end_year}")
    
    # Handle different paths for reference vs other scenarios
    if "reference" in scenario:
        data_path = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RESULTS"
    else:
        data_path = base_path
    
    data_list = []
    for year in range(start_year, end_year + 1):
        print(f"  Loading year {year}")
        try:
            file_pattern = f"{data_path}/{scenario}/rch/rch_{year}*.idf"
            yearly_data = imod.idf.open(file_pattern)
            data_list.append(yearly_data)
        except Exception as e:
            print(f"    Could not load {year}: {e}")
    
    if data_list:
        # Concatenate all years
        full_data = xr.concat(data_list, dim="time")
        
        # Remove duplicates if any
        time_series = pd.Series(full_data['time'].values)
        unique_times_index = ~time_series.duplicated(keep='first')
        full_data = full_data.isel(time=unique_times_index)
        
        return full_data
    else:
        return None

def calculate_total_recharge(recharge_data):
    """Calculate total recharge across all cells for each time step"""
    return recharge_data.sum(dim=['x', 'y'])

def plot_recharge_timeseries():
    """Create comprehensive recharge time series plots"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    all_scenario_data = {}
    
    # Load data for all scenarios and periods
    for i, modelname in enumerate(modelnames):
        print(f"\n=== Processing {modelname} ===")
        all_scenario_data[modelname] = {}
        
        for years in year_sets:
            start_year, end_year = years[0], years[1]
            period_name = f"{start_year}-{end_year}"
            
            # Load recharge data
            recharge_data = load_recharge_data(modelname, start_year, end_year)
            if recharge_data is not None:
                # Calculate total recharge time series
                total_recharge = calculate_total_recharge(recharge_data)
                all_scenario_data[modelname][period_name] = total_recharge
    
    # Plot 1: Time series for each scenario and period
    ax1 = axes[0]
    for i, (scenario, periods) in enumerate(all_scenario_data.items()):
        for j, (period, data) in enumerate(periods.items()):
            times = pd.to_datetime(data.time.values)
            values = data.values
            
            linestyle = '-' if '2016' in period else ('--' if '2047' in period else ':')
            ax1.plot(times, values, color=colors[i], linestyle=linestyle, 
                    linewidth=1.5, label=f"{scenario} ({period})", alpha=0.8)
    
    ax1.set_ylabel('Total Recharge (mm/day × cells)')
    ax1.set_title('Recharge Time Series by Scenario and Period')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Annual totals comparison
    ax2 = axes[1]
    bar_width = 0.8 / len(all_scenario_data)
    
    all_annual_data = {}
    for i, (scenario, periods) in enumerate(all_scenario_data.items()):
        annual_totals = {}
        for period, data in periods.items():
            times = pd.to_datetime(data.time.values)
            df = pd.DataFrame({'date': times, 'recharge': data.values})
            df['year'] = df['date'].dt.year
            annual_sums = df.groupby('year')['recharge'].sum()
            
            for year, total in annual_sums.items():
                if year not in annual_totals:
                    annual_totals[year] = total
        
        all_annual_data[scenario] = annual_totals
        
        # Plot bars
        years = list(annual_totals.keys())
        totals = list(annual_totals.values())
        x_pos = np.array(range(len(years))) + i * bar_width
        ax2.bar(x_pos, totals, bar_width, label=scenario, color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Annual Total Recharge')
    ax2.set_title('Annual Total Recharge by Scenario')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Set x-tick labels
    if all_annual_data:
        all_years = sorted(set().union(*[data.keys() for data in all_annual_data.values()]))
        ax2.set_xticks(range(len(all_years)))
        ax2.set_xticklabels(all_years, rotation=45)
    
    # Plot 3: Seasonal patterns
    ax3 = axes[2]
    for i, (scenario, periods) in enumerate(all_scenario_data.items()):
        all_monthly = []
        for period, data in periods.items():
            times = pd.to_datetime(data.time.values)
            df = pd.DataFrame({'date': times, 'recharge': data.values})
            df['month'] = df['date'].dt.month
            monthly_avg = df.groupby('month')['recharge'].mean()
            all_monthly.append(monthly_avg)
        
        if all_monthly:
            # Average across all periods for this scenario
            scenario_monthly = pd.concat(all_monthly, axis=1).mean(axis=1)
            ax3.plot(scenario_monthly.index, scenario_monthly.values, 
                    color=colors[i], marker='o', linewidth=2, label=scenario)
    
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Average Recharge (mm/day × cells)')
    ax3.set_title('Seasonal Recharge Patterns by Scenario')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = f"{visualizations_path}/recharge_timeseries"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{output_dir}/recharge_timeseries_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n=== RECHARGE SUMMARY ===")
    for scenario, periods in all_scenario_data.items():
        print(f"\n{scenario.upper()}:")
        for period, data in periods.items():
            avg_daily = data.mean().values
            max_daily = data.max().values
            min_daily = data.min().values
            print(f"  {period}: Avg={avg_daily:.1f}, Max={max_daily:.1f}, Min={min_daily:.1f}")

# Run the analysis
if __name__ == "__main__":
    plot_recharge_timeseries()
    print("\nRecharge time series analysis complete!")