import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Surface area of the solar panels absorbing sunlight as a percentage of the total
# Parameters 
A_max = 100   # surface area at 0 degrees as a percentage
A_min = 0    # surface area at 90 degrees as a percentage
occlusion_start = 135.05  # angle at which satellite first cannot be seen
occlusion_end   = 224.95  # angle at which satellite re-emerges

# Angle array
theta_deg = np.linspace(0, 360, 2000)
theta_rad = np.deg2rad(theta_deg)

# Projected-area function: A(theta) = (A_max - A_min)*|cos(theta)| + A_min
# Use shifted angle so min area is at t=0 (theta=0)
A = (A_max - A_min) * np.abs(np.sin(theta_rad)) + A_min

# Apply occlusion: set area to zero inside the occlusion interval
in_occlusion = (theta_deg >= occlusion_start) & (theta_deg <= occlusion_end)
A[in_occlusion] = 0.0

# Plot against angle of rotation
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, A, lw=2) 

# Annotations and labels
plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Orbital Angle')
plt.axvspan(occlusion_start, occlusion_end, color='grey', alpha=0.3, label='Occluded Region')
plt.xlabel('Orbital angle, θ (degrees)')
plt.ylabel('Visible surface area, A (m²)')
plt.xlim(0, 360)
plt.ylim(0, A_max * 1.05)
x_interval = 90 
plt.gca().xaxis.set_major_locator(MultipleLocator(x_interval))
plt.grid(alpha=0.3)
plt.show()

# Adding a second graph to show variation with time
time = theta_deg / 15
occlusion_start_time = occlusion_start / 15  # angle at which satellite first cannot be seen
occlusion_end_time   = occlusion_end / 15

# Plot against time
plt.figure(figsize=(10, 5))
plt.plot(time, A, lw=2)

# Annotations and labels
plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Time')
plt.axvspan(occlusion_start_time, occlusion_end_time, color='grey', alpha=0.3, label='Occluded Region')
plt.xlabel('Time, t (Hours)')
plt.ylabel('Visible surface area, A(θ) (m²)')
plt.xlim(0, 24)
plt.ylim(0, A_max * 1.05)
plt.grid(alpha=0.3)

plt.show()
