import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Constants 
sigma = 5.67e-8
solar_constant = 1361
albedo = 0.3
Gravitational_constant = 6.67430e-11
mass_of_earth = 5.972e24

threshold_low = 293
threshold_high = 313

# User Input 
height = float(input("Input a height for your satellite (1 - 5 m): "))
width = float(input("Input a width for your satellite (1 - 5 m): "))
depth = float(input("Input a depth for your satellite (3 - 7 m): "))
mass = float(input("Input a mass for your satellite (1000 - 3000 kg): "))

c = 900.0

coatings = {
    'Polymide Nanofiber Films': {'alpha': 0.004, 'epsilon': 0.93},
    'Ta2O5, SiN, SiO2 Film': {'alpha': 0.110, 'epsilon': 0.750},
    'Hughson White Paint A276': {'alpha': 0.26, 'epsilon': 0.88},
    'Bare Polished Aluminum': {'alpha': 0.4, 'epsilon': 0.04}
}

# Orbital Parameters 
altitude = 35786e3
earth_radius = 6371e3
year_seconds = 365.25 * 24 * 3600

orbital_radius = earth_radius + altitude
orbital_period = 2 * np.pi * np.sqrt(orbital_radius**3 / (Gravitational_constant * mass_of_earth))

dt = 10 * 60
day_start = 0
sim_days = 365
simulation_start = day_start * 24 * 3600
simulation_end = (day_start + sim_days) * 24 * 3600

# Time arrays
time_year = np.arange(simulation_start, simulation_end + dt, dt)
time_orbit = np.arange(0, orbital_period + dt, dt)
time_orbit_2 = np.linspace(0, orbital_period, 1000)
time_year_2 = np.linspace(0, year_seconds, int(year_seconds / dt))

time_sweep_start = simulation_start
time_sweep_days = 3
time_sweep = np.arange(time_sweep_start, time_sweep_start + time_sweep_days * 24 * 3600 + dt, dt)

time_long_start = simulation_start
time_long = np.arange(time_long_start, time_long_start + year_seconds + dt, dt)

# Eclipse & Area Functions 
def eclipse_mask(t):
    """Return True if satellite is in Earth's shadow."""
    day = (t / 86400) % 365.25
    season1 = (day > 59) & (day < 99)
    season2 = (day > 239) & (day < 279)
    in_season = season1 | season2

    eclipse_duration = 0.0708 * 86400
    orbit_phase = t % orbital_period
    in_eclipse = orbit_phase < eclipse_duration

    return in_season & in_eclipse


def set_interps_for(time_array):
    """Create interpolators for presented area, sunlight, and Earth view factor."""
    global A_pres_interp, sunlight_interp, A_earth_interp

    sunlight = np.where(eclipse_mask(time_array), 0, 1)

    # Tile one-orbit area profile across full time array
    theta_rad = np.pi/4 * np.sin(2 * np.pi * time_orbit / orbital_period)
    A_presented_orbit = height * (np.abs(width * np.cos(theta_rad)) + np.abs(depth * np.sin(theta_rad)))

    reps = int(np.ceil(len(time_array) / len(A_presented_orbit)))
    A_presented = np.tile(A_presented_orbit, reps)[:len(time_array)]

    half_angle_earth = np.arcsin(earth_radius / orbital_radius)
    A_earth_orbit = np.pi * (earth_radius**2) * (1 - np.cos(half_angle_earth))
    A_earth = A_earth_orbit * sunlight

    A_pres_interp = interp1d(time_array, A_presented, kind='linear', fill_value='extrapolate')
    sunlight_interp = interp1d(time_array, sunlight, kind='nearest', fill_value='extrapolate')
    A_earth_interp = interp1d(time_array, A_earth, kind='nearest', fill_value='extrapolate')


set_interps_for(time_year_2)

# Thermal Model 
def thermal_rhs(t, T, alpha, epsilon, heater_state, heater_power):
    """Compute dT/dt and heater state."""
    T = np.clip(float(T), 0, 750)

    A_pres = float(A_pres_interp(t))
    is_sunlit = float(sunlight_interp(t))
    A_earth_local = float(A_earth_interp(t))

    Q_solar = alpha * solar_constant * is_sunlit * A_pres
    Q_albedo = solar_constant * albedo * (A_earth_local / (4.0 * np.pi * orbital_radius**2))
    Q_ir_loss = epsilon * sigma * (T**4) * A_pres

    if T < threshold_low:
        heater_state = 1
    elif T > threshold_high:
        heater_state = 0

    Q_heater = heater_power * heater_state
    Q_net = Q_solar + Q_albedo + Q_heater - Q_ir_loss
    dTdt = Q_net / (mass * c)

    return dTdt, heater_state, Q_heater


def heat_balance_RK4_with_heater(alpha, epsilon, time_array, heater_power, T_init=300.0):
    """Integrate temperature using RK4 method."""
    n = len(time_array)
    T = np.zeros(n)
    heater_state = np.zeros(n)
    heater_energy = np.zeros(n)

    T[0] = T_init
    heater_state[0] = 0

    for i in range(n - 1):
        t = time_array[i]
        h = time_array[i + 1] - time_array[i]

        # RK4 slopes
        k1, h1_state, Qh1 = thermal_rhs(t, T[i], alpha, epsilon, heater_state[i], heater_power)
        k2, h2_state, Qh2 = thermal_rhs(t + h/2, T[i] + h/2*k1, alpha, epsilon, h1_state, heater_power)
        k3, h3_state, Qh3 = thermal_rhs(t + h/2, T[i] + h/2*k2, alpha, epsilon, h2_state, heater_power)
        k4, h4_state, Qh4 = thermal_rhs(t + h, T[i] + h*k3, alpha, epsilon, h3_state, heater_power)

        slope = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        T[i+1] = T[i] + h * slope
        heater_state[i+1] = h4_state
        heater_energy[i+1] = heater_energy[i] + Qh4 * h

    total_energy_kWh = heater_energy[-1] / 3.6e6
    return T, heater_state, total_energy_kWh

# Heater Power Sweep 
heater_powers = np.arange(3000, 13000, 200)
results = {}

for name, props in coatings.items():
    alpha, epsilon = props['alpha'], props['epsilon']
    energy_vs_power = []

    for power in heater_powers:
        _, _, energy_kWh = heat_balance_RK4_with_heater(alpha, epsilon, time_sweep, power)
        energy_vs_power.append(energy_kWh)

    optimal_power = heater_powers[np.argmin(energy_vs_power)]
    results[name] = {
        "heater_powers": heater_powers,
        "energy_vs_power": np.array(energy_vs_power),
        "optimal_power": optimal_power
    }

    # Plot power sweep
    plt.figure(figsize=(10, 5))
    plt.plot(heater_powers/1000, energy_vs_power, 'o-')
    plt.xlabel("Heater Power (kW)")
    plt.ylabel("Total Energy Used (kWh)")
    plt.title(f"{name} – Heater Energy vs Power")
    plt.grid(True)
    plt.tight_layout()

# Full-Year Simulation 
set_interps_for(time_long)

for name, props in coatings.items():
    alpha, epsilon = props['alpha'], props['epsilon']
    power = results[name]["optimal_power"]

    T_with, heater_state, energy_kWh = heat_balance_RK4_with_heater(alpha, epsilon, time_year_2, power)
    T_no, _, _ = heat_balance_RK4_with_heater(alpha, epsilon, time_year_2, 0)

    results[name].update({
        "T": T_with,
        "T_no_heater": T_no,
        "heater": heater_state,
        "energy_kWh": energy_kWh
    })

    print(f"{name}: {energy_kWh:.2f} kWh/year at {power:.0f} W Heater Power")

# Analysis & Plotting 
in_band_fraction = {name: np.mean((data["T"] >= threshold_low) & (data["T"] <= threshold_high))
                    for name, data in results.items()}

best_name = max(in_band_fraction, key=in_band_fraction.get)
T_best = results[best_name]["T"]

# Deviation analysis
below = T_best[T_best < threshold_low]
above = T_best[T_best > threshold_high]

avg_below = np.mean(threshold_low - below) if len(below) > 0 else 0.0
max_below = np.max(threshold_low - below) if len(below) > 0 else 0.0
avg_above = np.mean(above - threshold_high) if len(above) > 0 else 0.0
max_above = np.max(above - threshold_high) if len(above) > 0 else 0.0

print(f"\n--- Deviation analysis for {best_name} ---")
print(f"Time within band: {in_band_fraction[best_name]*100:.2f}%")
print(f"Below threshold: {len(below)/len(T_best)*100:.2f}%")
print(f"  Avg: {avg_below:.2f} K, Max: {max_below:.2f} K")
print(f"Above threshold: {len(above)/len(T_best)*100:.2f}%")
print(f"  Avg: {avg_above:.2f} K, Max: {max_above:.2f} K")

# Multi-objective scoring
print("\n--- Multi-Objective Optimization ---")
scores = {}

for name, data in results.items():
    if name == "Bare Polished Aluminum":
        continue

    in_band = np.mean((data["T"] >= threshold_low) & (data["T"] <= threshold_high))
    power_used = data["energy_kWh"]

    in_band_score = in_band
    power_score = 1.0 - np.clip(power_used / 100.0, 0, 1)
    combined_score = 0.5 * in_band_score + 0.5 * power_score

    scores[name] = {
        "in_band": in_band * 100,
        "energy_kWh": power_used,
        "combined_score": combined_score
    }

    print(f"{name}: {in_band*100:.2f}% in-band, {power_used:.2f} kWh at heater power {power:.0f}, score={combined_score:.3f}")

best_balanced = max(scores, key=lambda x: scores[x]["combined_score"])
print(f"\n→ Best balanced choice: {best_balanced}")

# Final Plots 
# Eclipse-season window
plt.figure(figsize=(10, 5))
start_day, end_day = 45, 75
start_idx = np.searchsorted(time_year_2, start_day * 86400.0)
end_idx = np.searchsorted(time_year_2, end_day * 86400.0)

for name, data in results.items():
    plt.plot(time_year_2[start_idx:end_idx]/86400.0, data["T"][start_idx:end_idx], label=f"{name}")

plt.fill_between([start_day, end_day], [threshold_low, threshold_low], [threshold_high, threshold_high], 
                 color='orange', alpha=0.15)
plt.fill_betweenx([0,750], 60, 75, color = 'gray', alpha = 0.1)
plt.xlabel("Time (days)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Evolution – Eclipse Season")
plt.legend(fontsize='small')
plt.grid(True)

# Six-month comparison
plt.figure(figsize=(10, 6))
months_6_idx = np.searchsorted(time_year_2, 182.625 * 86400.0)

for name in [best_balanced, "Bare Polished Aluminum"]:
    plt.plot(2 * time_year_2[:months_6_idx] / year_seconds * 6.0, results[name]["T"][:months_6_idx],
             label=f"{name} ({in_band_fraction[name]*100:.1f}%)")

plt.axhline(threshold_low, linestyle='--', color='gray')
plt.axhline(threshold_high, linestyle='--', color='gray')
plt.xlabel("Month")
plt.xticks(ticks=np.arange(0, 6, 1), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
plt.ylabel("Temperature (K)")
plt.title("Six-Month Temperature Profile")
plt.legend()
plt.grid(True)
plt.show()
