import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MultipleLocator

data_point = 4000
sigma = 5.670374419e-8  # Stefan-Boltzmann 

method = 'RK45'

m = 3000.0          # mass (kg)
cp = 900.0          # specific heat capacity (J/kg·K) 
SA_main_body = 55.4  # radiating surface area (m²)
epsilon = 0.85      # emissivity
Q_int = 50        # (50) internal heat generation (W)
Q_earthIR = 20    # (20)Earth IR absorbed during shadow (W)

Area_list = np.full(data_point, SA_main_body, dtype=float)


T_on = 273.15 + 20      # heater turns ON
T_off = 273.15 + 25    # heater turns OFF

heater_power = 23000   # heater power (W)

# Surface area of the solar panels absorbing sunlight as a percentage of the total
# Parameters 
A_max = 100   # surface area at 0 degrees as a percentage
A_min = 0    # surface area at 90 degrees as a percentage
occlusion_start = 135.05  # angle at which satellite first cannot be seen
occlusion_end   = 224.95  # angle at which satellite re-emerges

# Define orbital angle (0° → 360° over 24 hours)
theta_deg = np.linspace(0, 360, data_point)
theta_rad = np.deg2rad(theta_deg)

# Convert to time (seconds)
t_eval = (theta_deg / 360) * 24 * 3600   # one full orbit = 24 hours
t_span = (0, 24 * 3600)

# Projected-area function: A(theta) = (A_max - A_min)*|cos(theta)| + A_min
# Use shifted angle so min area is at t=0 (theta=0)
A_solar_panel = (A_max - A_min) * np.abs(np.cos(theta_rad)) + A_min
# Make the main body area an array (one value per angle) so masking/occlusion
# assignments like A_main_body[in_occlusion] = 0.0 work without error.
A_main_body = np.full(theta_deg.shape, SA_main_body, dtype=float)

# Apply occlusion: set area to zero inside the occlusion interval
in_occlusion = (theta_deg >= occlusion_start) & (theta_deg <= occlusion_end)
A_solar_panel[in_occlusion] = 0.0
A_main_body[in_occlusion] = 0.0
Area_list[in_occlusion] = 0.0


# Adding a second graph to show variation with time
time = theta_deg / 15
occlusion_start_time = occlusion_start / 15  # angle at which satellite first cannot be seen
occlusion_end_time   = occlusion_end / 15


print(f"The satellite is in the Earths shadow between t = {occlusion_start_time:.2f} hours and t = {occlusion_end_time:.2f} hours.")

def solar_input(t):
    I_sun = 1361.0       # Solar constant (W/m²)
    alpha = 0.25          # Absorptivity
    orbit_time = 24 * 3600  # full orbit = 24 hours
    index = int((t / orbit_time) * (len(Area_list) - 1))
    index = np.clip(index, 0, len(Area_list) - 1)  # avoid overflow
    return I_sun * alpha * Area_list[index]

heater_state = []

def satellite_temp_ODE(t, T):
    T = float(T)
    Q_out = epsilon * sigma * SA_main_body * T**4

    # Proportional heater control
    # if T < T_on:
    #     heater_fraction = max(0.0, min(1.0, (T_on - T) / (T_on - T_off)))
    # else:
    #     heater_fraction = 0.0

    # Q_heater = heater_power * heater_fraction
    # Q_net = Q_int + Q_earthIR + solar_input(t) + Q_heater - Q_out
    # dTdt = Q_net / (m * cp)
    # return dTdt

    if T < T_on:
        heater_state[0] = 1.0
    elif T > T_off:
        heater_state[0] = 0.0
    
    Q_heater = heater_power * heater_state


def satellite_cooling_with_heater(t, T):
    T_arr = np.asarray(T)
    if T_arr.size == 1:
        T = float(T_arr.item())
    else:
        T = float(T_arr.ravel()[0])
    Q_out = epsilon * sigma * SA_main_body * T**4
    
    #Heater control logic 
    # if T < T_on:
    #     heater_fraction = max(0.0, min(1.0, (T_on - T) / (T_on - T_off)))
    # else:
    #     heater_fraction = 0.0

    # Q_heater = heater_power * heater_fraction

    if T < T_on:
        heater_state[0] = 1.0
    elif T > T_off:
        heater_state[0] = 0.0
    
    Q_heater = heater_power * heater_state

    Q_net = Q_int + Q_earthIR + solar_input(t) + Q_heater - Q_out
    dTdt = Q_net / (m * cp)
    return dTdt


    # Proportional heater control
    # if T < T_on:
    #     heater_fraction = max(0.0, min(1.0, (T_on - T) / (T_on - T_off)))
    # else:
    #     heater_fraction = 0.0

    # Q_heater = heater_power * heater_fraction
    

# --- Integrate using solve_ivp ---
T0 = [273.15 + 20]   # initial temperature
#heater_state = [0.0] # mutable list to track heater on/off state

def rhs(t, y):
    return satellite_cooling_with_heater(t, y)

sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method=method)

#heater_states = (sol.y[0] < T_on).astype(float)

heater_powers = []
for T, t in zip(sol.y[0], sol.t):
    if T < T_on:
        heater_fraction = max(0.0, min(1.0, (T_on - T) / (T_on - T_off)))
    else:
        heater_fraction = 0.0
    heater_powers.append(heater_power * heater_fraction)

#heater_states = np.array(heater_states)
heater_powers = np.array(heater_powers)

energy_J = np.trapezoid(heater_powers, sol.t)  # J = ∫ P dt
energy_Wh = energy_J / 3600.0 

print(f"Total heater energy used during shade: {energy_J:,.1f} J ({energy_Wh:.2f} Wh)")
print(f"Equivalent average heater power: {energy_J / t_span[1]:.2f} W")

print(f"\nTotal heater energy used: {energy_J:,.1f} J ({energy_Wh:.2f} Wh)")
print(f"Average heater power: {energy_J / (24*3600):.2f} W")



I_sun = 1361.0          # solar flux at GEO (W/m²)
eta_panel = 0.25        # solar cell efficiency (30%)
eta_sys = 0.40          # overall power system efficiency (DC/DC, battery, etc.)

A_solar_panel_norm = A_solar_panel / np.max(A_solar_panel)

# Solve for required maximum panel area so total energy matches heater energy
k = energy_J / np.trapezoid(I_sun * eta_panel * eta_sys * A_solar_panel_norm, t_eval)
A_panel_required = k
print(f"Required maximum panel area: {A_panel_required:.2f} m²")



# # Plot against angle of rotation
# plt.figure(figsize=(10, 5))
# plt.plot(theta_deg, A, lw=2) 

# # Annotations and labels
# plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Orbital Angle')
# plt.axvspan(occlusion_start, occlusion_end, color='grey', alpha=0.3, label='Occluded Region')
# plt.xlabel('Orbital angle, θ (degrees)')
# plt.ylabel('Visible surface area, A (m²)')
# plt.xlim(0, 360)
# plt.ylim(0, A_max * 1.05)
# x_interval = 90 
# plt.gca().xaxis.set_major_locator(MultipleLocator(x_interval))
# plt.grid(alpha=0.3)
# plt.show()

# # Plot against time
# plt.figure(figsize=(10, 5))
# plt.plot(time, A, lw=2)

# # Annotations and labels
# plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Time')
# plt.axvspan(occlusion_start_time, occlusion_end_time, color='grey', alpha=0.3, label='Occluded Region')
# plt.xlabel('Time, t (Hours)')
# plt.ylabel('Visible surface area, A(θ) (m²)')
# plt.xlim(0, 24)
# plt.ylim(0, A_max * 1.05)
# plt.grid(alpha=0.3)
# plt.show()

# --- Plot temperature over 24 hours ---

heater_on_mask = np.array(heater_powers) > 0.5

plt.figure(figsize=(10, 5))
plt.plot(sol.t / 3600, sol.y[0] - 273.15, lw=2, label="Satellite Temperature")
plt.axhline(T_on - 273.15, color="blue", linestyle="--", label="Heater ON threshold")
plt.axhline(T_off - 273.15, color="red", linestyle="--", label="Heater OFF threshold")
plt.fill_between(sol.t / 3600, -100, 100, where=heater_on_mask,
                 color="orange", alpha=0.3, label="Heater ON")

plt.axvspan(occlusion_start/15, occlusion_end/15, color='grey', alpha=0.3, label="Earth Shadow")

plt.xlabel("Time (hours)")
plt.ylabel("Temperature (°C)")
plt.title("Satellite Temperature Over 24-Hour Orbit")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# # --- Optional: Plot solar panel visible percentage ---
# plt.figure(figsize=(10, 4))
# plt.plot(theta_deg, A_solar_panel, lw=2)
# plt.axvspan(occlusion_start, occlusion_end, color='grey', alpha=0.3, label='Shadow')
# plt.xlabel("Orbital Angle (°)")
# plt.ylabel("Visible Solar Panel Area (%)")
# plt.title("Solar Panel Visibility Over Orbit")
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show() 


powers = np.arange(1000, 30000 + 1, 250)  # test heater powers from 1kW to 30kW
energy_usages = []

for power in powers:
    heater_power = power
    heater_state = [0.0]

    def rhs(t, y):
        return satellite_temp_ODE(t, y)

    sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method=method)

    heater_powers = []
    for T, t in zip(sol.y[0], sol.t):
        if T < T_on:
            heater_fraction = max(0.0, min(1.0, (T_on - T) / (T_on - T_off)))
        else:
            heater_fraction = 0.0
        heater_powers.append(heater_power * heater_fraction)


    energy_J = np.trapezoid(heater_powers, sol.t)
    energy_usages.append(energy_J / 3600.0)  # convert to Wh

from scipy.interpolate import make_interp_spline

powers_smooth = np.linspace(min(powers), max(powers), 300)
spline = make_interp_spline(powers, energy_usages)
energy_smooth = spline(powers_smooth)

plt.figure(figsize=(10, 5))
plt.plot(powers_smooth, energy_smooth, lw=2, label="Smoothed Curve")
plt.scatter(powers, energy_usages, color='red', label="Original Points", alpha=0.5)
plt.xlabel("Heater Power Output (W)")
plt.ylabel("Total Energy Usage (Wh)")
plt.title("Smoothed Energy Usage vs Heater Power Output")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
