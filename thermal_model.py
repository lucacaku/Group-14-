import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# constants and geometry of satellite
sigma = 5.67e-8  # stefan boltzmann constant W/m^2K^4
solar_constant = 1361  # W/m^2
albedo = 0.3
Gravitational_constant = 6.67430e-11  # m^3 kg^-1 s^-2
mass_of_earth = 5.972e24  # kg

threshold_low = 283.0  # K (10 C)
threshold_high = 313.0  # K (40 C)

# satellite geometry (m)
height = 2.7
width = 3.6
depth = 5.7

# mass & thermal capacity
mass = 2630.0  # kg
c = 900.0  # J/kgK

# coatings (alpha = solar absorptivity, epsilon = emissivity)
coatings = {
    'Polymide Nanofiber Films': {'alpha': 0.004, 'epsilon': 0.93},
    'Ta2O5, SiN, SiO2 Film': {'alpha': 0.110, 'epsilon': 0.750},
    'Hughson White Paint A276': {'alpha': 0.26, 'epsilon': 0.88},
    'Bare Polished Aluminum': {'alpha': 0.4, 'epsilon': 0.04}
}

# orbital parameters
altitude = 35786e3
earth_radius = 6371e3
year_seconds = 365.25 * 24 * 3600.0

orbital_radius = earth_radius + altitude
orbital_period = 2 * np.pi * np.sqrt(orbital_radius**3 / (Gravitational_constant * mass_of_earth))

# time arrays
time_orbit = np.linspace(0, orbital_period, 1000)    # one orbit sampling
time_year = np.linspace(0, year_seconds, 25000)     # full-year sampling

# eclipse and presented area
def eclipse_mask(t):
    # convert to seconds
    day = (t / 86400.0) % 365.25

    # eclipse seasons 
    season1 = (day > 59) & (day < 99)   # centered around equinoxes (mar 20, sep 22)
    season2 = (day > 239) & (day < 279)  
    in_season = season1 | season2

    # orbital-phase eclipse duration 
    eclipse_duration = 0.0708 * 86400.0  # seconds
    orbit_phase = t % orbital_period
    in_eclipse = orbit_phase < eclipse_duration

    return in_season & in_eclipse

sunlight_year = np.where(eclipse_mask(time_year), 0.0, 1.0) # zero if eclipse

# orbital presented area model
theta_rad = np.pi/4 * np.sin(2 * np.pi * time_orbit / orbital_period)
A_presented_orbit = height * (np.abs(width * np.cos(theta_rad)) + np.abs(depth * np.sin(theta_rad)))

# tile presented area for full year
reps = int(np.ceil(len(time_year) / len(A_presented_orbit)))
A_presented = np.tile(A_presented_orbit, reps)[:len(time_year)]

# earth-view area
half_angle_earth = np.arcsin(earth_radius / orbital_radius)
A_earth_orbit = np.pi * (earth_radius**2) * (1 - np.cos(half_angle_earth))
A_earth = A_earth_orbit * sunlight_year  

# create interpolants for continuous time use
A_pres_interp = interp1d(time_year, A_presented, kind='linear', fill_value='extrapolate')
sunlight_interp = interp1d(time_year, sunlight_year, kind='nearest', fill_value='extrapolate')
A_earth_interp = interp1d(time_year, A_earth, kind='linear', fill_value='extrapolate')

# heater prerequisites
heater_power = 5000.0  # W 

def thermal_derivative(t, T, alpha, epsilon, heater_state):
    A_pres = float(A_pres_interp(t))
    is_sunlit = float(sunlight_interp(t))
    A_earth_local = float(A_earth_interp(t))

    # direct solar 
    Q_solar = alpha * solar_constant * is_sunlit * A_pres

    # albedo/earth-scattered component approximation 
    Q_albedo = solar_constant * albedo * (A_earth_local / (4.0 * np.pi * orbital_radius**2))

    # radiative loss to space 
    Q_ir_loss = epsilon * sigma * (T**4) * A_pres

    Q_heater = heater_power * heater_state

    Q_net = Q_solar + Q_albedo + Q_heater - Q_ir_loss

    dTdt = Q_net / (mass * c)
    return dTdt

# integrator with persistent heater
def heat_balance_RK4_with_heater(alpha, epsilon, time_array, T_init=300.0):
    n = len(time_array)
    T = np.zeros(n)
    heater_state = np.zeros(n, dtype=int)
    heater_energy_J = np.zeros(n) 

    T[0] = T_init
    heater_state[0] = 0

    for i in range(n-1):
        t = time_array[i]
        dt = time_array[i+1] - time_array[i]

        # current heater state 
        h_state = int(heater_state[i])

        # RK4 slopes
        k1 = thermal_derivative(t, T[i], alpha, epsilon, h_state)
        k2 = thermal_derivative(t + dt/2.0, T[i] + dt/2.0 * k1, alpha, epsilon, h_state)
        k3 = thermal_derivative(t + dt/2.0, T[i] + dt/2.0 * k2, alpha, epsilon, h_state)
        k4 = thermal_derivative(t + dt, T[i] + dt * k3, alpha, epsilon, h_state)

        T_new = T[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        T[i+1] = T_new

        # update heater state based on new temperature
        if T_new < threshold_low:
            heater_state[i+1] = 1
        elif T_new > threshold_high:
            heater_state[i+1] = 0
        else:
            heater_state[i+1] = heater_state[i]  # hold previous state in the band

        # accumulate heater energy 
        heater_energy_J[i+1] = heater_energy_J[i] + (heater_power * h_state * dt)

    total_energy_kWh = heater_energy_J[-1] / 3.6e6
    return T, heater_state, total_energy_kWh

# no heater simulation
def heat_balance_no_heater(alpha, epsilon, time_array, T_init=300.0):
    dt = time_array[1] - time_array[0]
    T = np.zeros_like(time_array)
    T[0] = T_init
    for i in range(1, len(time_array)):
        # simple explicit Euler for baseline
        is_sunlit = sunlight_year[i]
        Q_solar = alpha * solar_constant * is_sunlit * A_presented[i]
        Q_albedo = solar_constant * albedo * (A_earth[i] / (4.0 * np.pi * orbital_radius**2))
        Q_ir = epsilon * sigma * (T[i-1]**4) * A_presented[i]
        net_Q = Q_solar + Q_albedo - Q_ir
        T[i] = T[i-1] + (net_Q * dt) / (mass * c)
    return T

# run sims for each coating
results = {}

# initial temp
T_init = 300.0

for name, props in coatings.items():
    alpha = props['alpha']
    epsilon = props['epsilon']

    T_with, heater_state, energy_kWh = heat_balance_RK4_with_heater(alpha, epsilon, time_year, T_init=T_init)
    T_no = heat_balance_no_heater(alpha, epsilon, time_year, T_init=T_init)

    results[name] = {
        "T": T_with,
        "T_no_heater": T_no,
        "heater": heater_state,
        "energy_kWh": energy_kWh
    }

    print(f"{name}: Total heater energy = {energy_kWh:.2f} kWh/year")

# first 30 days plot
plt.figure(figsize=(10, 5))
days_30_index = np.searchsorted(time_year, 30*86400.0)  # index for 30 days
for name, data in results.items():
    plt.plot(time_year[:days_30_index]/86400.0, data["T"][:days_30_index], label=f"{name} (with heater)")
    plt.plot(time_year[:days_30_index]/86400.0, data["T_no_heater"][:days_30_index], linestyle='--', alpha=0.7, label=f"{name} (no heater)")
plt.fill_between([0,30],[threshold_low, threshold_low],[threshold_high, threshold_high], color='orange', alpha=0.15)
plt.xlabel("Time (days)")
plt.xlim(0, 30)
plt.ylabel("Temperature (K)")
plt.title("Temperature evolution (first 30 days) — with & without heater")
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.show()

# find best coating
in_band_fraction = {}
for name, data in results.items():
    T = data["T"]
    fraction_in_band = np.mean((T >= threshold_low) & (T <= threshold_high))
    in_band_fraction[name] = fraction_in_band

best_name = max(in_band_fraction, key=in_band_fraction.get)
worst_name = "Bare Polished Aluminum"  # known to be worst as heater energy used = 0 kWh and steady state = 700K

# print percentage of year in thresholds
print("Fraction of year within thresholds:")
for name, frac in in_band_fraction.items():
    print(f"  {name}: {frac*100:.1f}%")

print(f"\nBest coating: {best_name}")
print(f"Worst coating: {worst_name}")

# name of best coating
T_best = results[best_name]["T"]

# calculate deviations
below = T_best[T_best < threshold_low]
above = T_best[T_best > threshold_high]

if len(below) > 0:
    avg_below = np.mean(threshold_low - below)
    max_below = np.max(threshold_low - below)
else:
    avg_below = 0.0
    max_below = 0.0

if len(above) > 0:
    avg_above = np.mean(above - threshold_high)
    max_above = np.max(above - threshold_high)
else:
    avg_above = 0.0
    max_above = 0.0

# print results
print(f"\n--- Deviation analysis for ({best_name}) ---")
print(f"Time within band: {in_band_fraction[best_name]*100:.2f}%")
print(f"Below threshold: {len(below)/len(T_best)*100:.2f}% of year")
print(f"  Avg deviation: {avg_below:.2f} K, Max deviation: {max_below:.2f} K")
print(f"Above threshold: {len(above)/len(T_best)*100:.2f}% of year")
print(f"  Avg deviation: {avg_above:.2f} K, Max deviation: {max_above:.2f} K")

# plot worst and best on 6 month plot
plt.figure(figsize=(10, 6))
months_6_index = np.searchsorted(time_year, 182.625*86400.0)

for name in [best_name, worst_name]:
    plt.plot(2*time_year[:months_6_index]/year_seconds * 6.0,
             results[name]["T"][:months_6_index],
             label=f"{name} ({in_band_fraction[name]*100:.1f}% in range)")

# threshold lines
plt.axhline(threshold_low, linestyle='--', color='gray', label='Thresholds')
plt.axhline(threshold_high, linestyle='--', color='gray')

# labels and styling
plt.xlabel("Month of Year")
plt.xticks(ticks=np.arange(0,6,1), labels=['Jan','Feb','Mar','Apr','May','Jun'])
plt.ylabel("Temperature (K)")
plt.title("Six-Month Temperature Profile — Best vs Worst Coating (with heater)")
plt.grid(True)
plt.legend()
plt.show()

