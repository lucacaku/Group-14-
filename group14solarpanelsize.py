import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

print("starting code")

sigma = 5.670374419e-8  # Stefan-Boltzmann 

m = 3000.0          # mass (kg)
cp = 900.0          # specific heat capacity (J/kg·K)
A = 55.4             # radiating surface area (m²)
epsilon = 0.85      # emissivity
Q_int = 50        # (50) internal heat generation (W)
Q_earthIR = 20    # (20)Earth IR absorbed during shadow (W)

time_in_shadow = 10 # hours to seconds

heater_power = 25000   # heater power (W)
T_on = 273.15 + 10      # heater turns ON below 5 °C
T_off = 273.15 + 20    # heater turns OFF above 15 °C

# --- Simulation time (shadow period) ---
t_span = (0, 3600 * time_in_shadow)   
t_eval = np.linspace(*t_span, 2000)

# --- ODE definition ---
def satellite_cooling_with_heater(t, T, heater_state):
    # Ensure T is a Python float scalar. solve_ivp may pass T as a 1-element
    # array; converting directly with float(array) is deprecated in NumPy 1.25+.
    T_arr = np.asarray(T)
    if T_arr.size == 1:
        T = float(T_arr.item())
    else:
        # If for some reason a vector is passed, use the first element.
        T = float(T_arr.ravel()[0])
    Q_out = epsilon * sigma * A * T**4
    
    # Heater control logic (simple hysteresis)
    if T < T_on:
        heater_state[0] = 1.0
    elif T > T_off:
        heater_state[0] = 0.0
    
    Q_heater = heater_power * heater_state[0]
    
    # Energy balance ODE
    dTdt = (Q_int + Q_earthIR + Q_heater - Q_out) / (m * cp)
    return dTdt

# --- Integrate using solve_ivp ---
T0 = [273.15 + 5]   # initial temperature (20°C)
heater_state = [0.0] # mutable list to track heater on/off state

def rhs(t, y):
    return satellite_cooling_with_heater(t, y, heater_state)

sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method='RK45')

# --- Track heater state over time ---
heater_states = []
heater_powers = []
for T in sol.y[0]:
    if T < T_on:
        heater_state[0] = 1.0
    elif T > T_off:
        heater_state[0] = 0.0
    heater_states.append(heater_state[0])
    heater_powers.append(heater_power * heater_state[0])
heater_states = np.array(heater_states)
heater_powers = np.array(heater_powers)

# --- Calculate total heater energy used ---
# Integrate power over time (Joules)
# Use numpy.trapezoid (replaces deprecated np.trapz) to compute the integral
# sol.t is in seconds, heater_powers is in watts (J/s), so integral is in J
energy_J = np.trapezoid(heater_powers, sol.t)  # J = ∫ P dt
energy_Wh = energy_J / 3600.0                  # convert J → Wh

print(f"Total heater energy used during shade: {energy_J:,.1f} J ({energy_Wh:.2f} Wh)")
print(f"Equivalent average heater power: {energy_J / t_span[1]:.2f} W")

# --- Plot temperature and heater ON periods ---
plt.figure(figsize=(9, 5))
plt.plot(sol.t / 60, sol.y[0] - 273.15, label='Satellite Temperature (°C)')
plt.axhline(T_on - 273.15, color='blue', linestyle='--', label='Heater ON threshold (5°C)')
plt.axhline(T_off - 273.15, color='red', linestyle='--', label='Heater OFF threshold (15°C)')
plt.fill_between(sol.t / 60, -50, 100, where=heater_states > 0.5,
                 color='orange', alpha=0.25, label='Heater ON')

plt.title('Satellite Temperature in Earth Shadow with Active Heater')
plt.xlabel('Time in Shadow (minutes)')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot heater power profile (optional) ---
plt.figure(figsize=(9, 3))
plt.plot(sol.t / 60, heater_powers, color='orange', label='Heater Power (W)')
plt.xlabel('Time in Shadow (minutes)')
plt.ylabel('Power (W)')
plt.title('Heater Power Usage During Shadow')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

Q_out = epsilon * sigma * A * (273)**4
print(Q_out)

# --- Solar panel sizing to supply heater energy ---

# Orbital sunlight/shadow durations (for GEO near equinox)
t_sun = 24*3600 - (time_in_shadow*3600) # seconds in sunlight (~22.5 h)

# Solar array and system parameters
I_sun = 1361.0          # solar flux at GEO (W/m²)
eta_panel = 0.30        # solar cell efficiency (30%)
eta_sys = 0.80          # overall power system efficiency (DC/DC, battery, etc.)

# Compute required solar panel area (m²)
A_panel = energy_J / (I_sun * eta_panel * eta_sys * t_sun)

# Also compute corresponding electrical power during sun
P_panel = A_panel * I_sun * eta_panel * eta_sys

print("\n--- Solar Power Sizing ---")
print(f"Sunlight duration per orbit: {t_sun/3600:.2f} hours")
print(f"Total heater energy to recharge: {energy_J:,.1f} J ({energy_Wh:.2f} Wh)")
print(f"Required solar panel area: {A_panel:.3f} m²")
print(f"Equivalent average electrical output during sunlight: {P_panel:.1f} W")

# --- Check for catastrophic temperature limits ---
T_min_limit = 273.15 + 0   # 0 °C lower limit
T_max_limit = 273.15 + 30  # 30 °C upper limit

min_temp = sol.y[0].min()
max_temp = sol.y[0].max()

if min_temp < T_min_limit:
    print(f"SATELLITE HAS EXPLODED ")
elif max_temp > T_max_limit:
    print(f"SATELLITE HAS EXPLODED ")
else:
    print(f"Safe ")