# Simple script to plot recharge patterns from IDF files
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import imod
import numpy as np


base_path = "P:/11207941-005-terschelling-model/TERSCHELLING_IMOD_MODEL_50X50/RCH/reference_future/rch"
scenario = "reference"  # Change this to your scenario

# Load first 10 years and last 10 years for comparison
first_years = list(range(2024, 2034))  # 2024-2033
last_years = list(range(2096, 2106))   # 2096-2105
years_to_load = first_years + last_years

print(f"Loading first 10 years: {first_years[0]}-{first_years[-1]}")
print(f"Loading last 10 years: {last_years[0]}-{last_years[-1]}")

# Load files year by year to show progress
data_list = []
for year in years_to_load:
    print(f"  Loading year {year}...")
    try:
        year_pattern = f"{base_path}/rch_{year}*.idf"
        yearly_data = imod.idf.open(year_pattern)
        data_list.append(yearly_data)
    except Exception as e:
        print(f"    Could not load {year}: {e}")

if not data_list:
    print("No data found! Check your file path.")
    exit()

print("Concatenating data...")
recharge_data = xr.concat(data_list, dim="time")

# Calculate total recharge over entire model domain (sum all cells) for each time step
# This gives us daily total recharge across all model cells
total_recharge_daily = recharge_data.sum(dim=['x', 'y'])

print(f"Data loaded: {len(total_recharge_daily.time)} daily time steps")
print(f"Each time step represents total recharge across {recharge_data.x.size * recharge_data.y.size} cells")

# Convert to pandas for easier plotting
times = pd.to_datetime(total_recharge_daily.time.values)
values = total_recharge_daily.values

# Create DataFrame
df = pd.DataFrame({
    'date': times,
    'total_recharge': values
})

# --- PLOTTING ---
plt.figure(figsize=(15, 12))

# Plot 1: Annual total recharge (sum per year)
plt.subplot(3, 1, 1)
df['year'] = df['date'].dt.year
annual_sum = df.groupby('year')['total_recharge'].sum()
plt.bar(annual_sum.index, annual_sum.values, color='blue', alpha=0.7, edgecolor='black')
plt.title('Annual Total Recharge (Sum of All Daily Values)')
plt.ylabel('Total Annual Recharge (mm × cells)')
plt.grid(True, alpha=0.3, axis='y')

# Plot 2: Annual averages to see daily average trend
plt.subplot(3, 1, 2)
annual_avg = df.groupby('year')['total_recharge'].mean()
plt.plot(annual_avg.index, annual_avg.values, marker='o', linewidth=2, color='green')
plt.title('Annual Average Daily Recharge')
plt.xlabel('Year')
plt.ylabel('Average Daily Recharge (mm × cells)')
plt.grid(True, alpha=0.3)

# Plot 3: Monthly pattern (average by month across all years)
plt.subplot(3, 1, 3)
df['month'] = df['date'].dt.month
monthly_avg = df.groupby('month')['total_recharge'].mean()

plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, color='red')
plt.title('Seasonal Pattern (Monthly Averages Across All Years)')
plt.xlabel('Month')
plt.ylabel('Average Total Recharge')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some basic statistics
print(f"\n=== RECHARGE PATTERN SUMMARY ===")
print(f"Time period: {times[0].strftime('%Y-%m-%d')} to {times[-1].strftime('%Y-%m-%d')}")
print(f"Total data points: {len(df)}")
print(f"Average recharge: {df['total_recharge'].mean():.2f}")
print(f"Max recharge: {df['total_recharge'].max():.2f}")
print(f"Min recharge: {df['total_recharge'].min():.2f}")

print(f"\nMonthly averages:")
for month, avg in monthly_avg.items():
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    print(f"  {month_name}: {avg:.2f}")

print(f"\nData frequency: Daily ({len(df)} days total)")
print(f"Years covered: {df['year'].min()} to {df['year'].max()}")

# Summary statistics for annual totals
print(f"\nAnnual Total Statistics:")
print(f"  Highest annual total: {annual_sum.max():.0f} (Year {annual_sum.idxmax()})")
print(f"  Lowest annual total: {annual_sum.min():.0f} (Year {annual_sum.idxmin()})")
print(f"  Average annual total: {annual_sum.mean():.0f}")