import numpy as np
import matplotlib.pyplot as plt

# Cuboid dimensions
width = 3.6
height = 2.7
depth = 5.7
tilt_deg = 23.44

# Rotation angles
theta_deg = np.linspace(0, 360, 1000)
theta_rad = np.radians(theta_deg)
tilt_rad = np.radians(tilt_deg)

# Front and side faces
front_side_visible = height * (np.abs(width * np.cos(theta_rad)) + np.abs(depth * np.sin(theta_rad)))

# Top and bottom faces
top_face_visible = width * depth * np.sin(tilt_rad) * (1 + np.cos(theta_rad)) / 2
bottom_face_visible = width * depth * np.sin(tilt_rad) * (1 - np.cos(theta_rad)) / 2

# Total visible surface area
visible_area = front_side_visible + top_face_visible + bottom_face_visible

# Plot
plt.figure(figsize=(10, 6))
plt.plot(theta_deg, visible_area, label='Visible Surface Area')
plt.title('Visible Surface Area of Tilted Cuboid')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Visible Surface Area (m²)')
plt.grid(True)
plt.legend()
plt.xticks(np.arange(0, 361, 45))
plt.tight_layout()
plt.show()
