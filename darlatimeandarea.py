import numpy as np
import matplotlib.pyplot as plt

# Surface area of the solar panels absorbing sunlight as a percentage of the total
A_max = 1
A_min = 0

# Angle array
theta_deg = np.linspace(0, 360, 1000)
time = theta_deg / 15  # convert angle → hours

# === Worst Day Eclipse (GEO) ===
eclipse_duration_h = 1.18        # 70.8 minutes
eclipse_center_h = 12.0          # midnight = GEO eclipse center

eclipse_half_angle = (eclipse_duration_h / 24 * 360) / 2
theta_eclipse_start = 180 - eclipse_half_angle
theta_eclipse_end   = 180 + eclipse_half_angle

# Convert eclipse window to time
t_eclipse_start = theta_eclipse_start / 15
t_eclipse_end   = theta_eclipse_end / 15

# === Create Sunlight Mask (1 in sun, 0 in eclipse) ===
sunlight_mask = np.where(
    (theta_deg >= theta_eclipse_start) & (theta_deg <= theta_eclipse_end),
    0,
    1
)

# === Area function with eclipse applied ===
def area(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    A = (A_max - A_min) * np.abs(np.cos(theta_rad)) + A_min
    return A * sunlight_mask    # <-- zero area during eclipse



energy_heater = 2000
internal_energy_use = 20000 # W
energy_J = 86400 * (energy_heater + internal_energy_use)
I_sun = 1361.0               # W/m²
eta_panel = 0.679             # solar cell efficiency
eta_sys = 0.60               # system efficiency
eta_total = eta_panel * eta_sys   # effective conversion efficiency

power = I_sun * eta_total * area(theta_deg)

power_avg = np.sum(power)/len(power)

energy_day = power_avg * 86400

surface_area = energy_J / energy_day

surface_area_additional = surface_area * 1.1 # to account for lunar eclipses

print(surface_area_additional)


percentage = ((energy_heater + internal_energy_use) / (I_sun * eta_total * surface_area_additional))

print(percentage)


A_values = area(theta_deg)
mask = A_values < percentage * A_max
indices = np.where(mask)[0]

intervals = []
if len(indices) > 0:
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            end = indices[i - 1]
            intervals.append((time[start], time[end]))
            start = indices[i]
    intervals.append((time[start], time[indices[-1]]))

print("Intervals when surface area < 10% (battery needed):")
for (t1, t2) in intervals:
    print(f"From {t1:.2f} h to {t2:.2f} h")

# ================================
# PLOT 1: Area vs Orbital Angle
# ================================
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, area(theta_deg), lw=2, label="Visible Solar Area")

# Shade eclipse
plt.axvspan(theta_eclipse_start, theta_eclipse_end, color='gray', alpha=0.3,
            label="Eclipse (Area=0)")

plt.title('Visible Solar Panel Area vs Orbital Angle')
plt.xlabel('Orbital angle, θ (degrees)')
plt.ylabel('Visible surface area, A (%)')
plt.xlim(0, 360)
plt.ylim(0, A_max * 1.05)
plt.axhline(percentage, color='r', linestyle='--', label="Required Power")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ================================
# PLOT 2: Area vs Time
# ================================
plt.figure(figsize=(10, 5))
plt.plot(time, area(theta_deg), lw=2, label="Visible Solar Area")

# Shade eclipse
plt.axvspan(t_eclipse_start, t_eclipse_end, color='gray', alpha=0.3,
            label="Eclipse (Area=0)")

plt.title('Visible Solar Panel Area vs Time')
plt.xlabel('Time, t (Hours)')
plt.ylabel('Visible surface area, A (%)')
plt.axhline(percentage, color='r', linestyle='--', label="Required Power")
plt.xlim(0, 24)
plt.ylim(0, A_max * 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.show()
