import numpy as np
import matplotlib.pyplot as plt

# Tidal cycle
tide_period = 12  # hours
eb = -1.0
vloed = 0.8
surface_level = -0.25  # dotted line

# Time array
time_hours = np.linspace(0, 24, 1440)
hourly_points = np.arange(0, 25, 1)

# Mean water level and amplitude
mean_sea_level = (eb + vloed) / 2
amplitude = (vloed - eb) / 2

# Generate tidal signal
tide_level = mean_sea_level + amplitude * np.sin(2 * np.pi * time_hours / tide_period)
tide_hourly = mean_sea_level + amplitude * np.sin(2 * np.pi * hourly_points / tide_period)

# Generate heads in shallow subsurface
heads = np.maximum(tide_level, surface_level)  # stays at surface if tide < surface

# Plot
plt.figure(figsize=(10, 4))
plt.plot(time_hours, tide_level, label='Tidal wave', color='blue')
plt.plot(time_hours, heads, label='Heads in shallow subsurface', color='orange')
plt.axhline(surface_level, color='gray', linestyle='--', label='Surface level')

plt.xlabel('Time (hours)')
plt.ylabel('Elevation (m)')
plt.xticks(np.arange(0, 25, 6))
plt.yticks = np.arange(eb + 0.2, vloed + 0.2, 0.2)
plt.title('Tidal wave and surface level on shallow surface heads')
plt.legend()
plt.tight_layout()
plt.show()
