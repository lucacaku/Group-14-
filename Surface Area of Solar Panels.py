import numpy as np
import matplotlib.pyplot as plt

# Surface area of the solar panels absorbing sunlight as a percentage of the total
A_max = 100   # surface area at 0 degrees as a percentage
A_min = 0     # surface area at 90 degrees as a percentage
    
# Angle array
theta_deg = np.linspace(0, 360, 1000)

# Area function
def area(theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    return (A_max - A_min) * np.abs(np.cos(theta_rad)) + A_min


# Plot vs orbital angle
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, area(theta_deg), lw=2)
plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Orbital Angle')
plt.xlabel('Orbital angle, θ (degrees)')
plt.ylabel('Visible surface area, A (m²)')
plt.xlim(0, 360)
plt.ylim(0, A_max * 1.05)
plt.grid(alpha=0.3)
plt.show()

# Plot vs time
time = theta_deg / 15
plt.figure(figsize=(10, 5))
plt.plot(time, area(theta_deg), lw=2)
plt.title('Variation of the Percentage of Visible Surface Area of Solar Panels with Time')
plt.xlabel('Time, t (Hours)')
plt.ylabel('Visible surface area, A(θ) (m²)')
plt.xlim(0, 24)
plt.ylim(0, A_max * 1.05)
plt.grid(alpha=0.3)
plt.show()


# Root finding - Ralston-Rabinowitz
dx = 1e-6

# First derivative (central difference)
def area_prime(theta_deg):
    return (area(theta_deg + dx) - area(theta_deg - dx)) / (2 * dx)

# Second derivative (central difference)
def area_double_prime(theta_deg):
    return (area(theta_deg + dx) - 2*area(theta_deg) + area(theta_deg - dx)) / (dx ** 2)

def ralston_rabinowitz(f, f1, f2, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        fpx = f1(x)
        fppx = f2(x)

        denom = (fpx)**2 - fx * fppx
        if abs(denom) < 1e-12:
            raise ZeroDivisionError("Denominator too small — method failed to converge.")

        x_new = x - (fx * fpx) / denom
        if abs(x_new - x) < tol:
            return x_new
        x = x_new

    raise ValueError("Method did not converge.")


root1 = ralston_rabinowitz(area, area_prime, area_double_prime, 85)
root2 = ralston_rabinowitz(area, area_prime, area_double_prime, 275)

print(f"The solar panel surface area will be zero at the angles: {root1:.2f}, {root2:.2f}")

# Convert to time
roots = np.array([root1, root2])
roots_time = roots / 15
roots_time_str = " ".join(f"{r:.2f}" for r in roots_time)

print(f"The solar panel surface area will be zero at the times: {roots_time_str}")


# when area is less than 10% - battery power needed
mask = area(theta_deg) < 10
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

print("Intervals when surface area < 10%:")
for (t1, t2) in intervals:
    print(f"From {t1:.2f} h to {t2:.2f} h")

