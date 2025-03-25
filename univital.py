import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants for cosmology
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 3.0e8        # Speed of light (m/s)
H0 = 70.0        # Hubble constant (km/s/Mpc), approximately 70 km/s per megaparsec
rho = 1.0e-26     # Example energy density (kg/m^3), adjust as needed
p = 1.0e-25       # Example pressure (Pa), adjust as needed

# Adjustments for biological/feedback factors
alpha = 0.05     # Feedback scaling factor for self-regulation
beta = 0.02      # Autopoiesis scaling factor
gamma = 0.1      # Biodiversity scaling factor
delta = 0.05     # Consciousness field scaling factor

# Function to compute the Hubble parameter at time t, using the Friedmann equation
def Hubble(t):
    # Use a simple model where the Hubble parameter decreases over time due to cosmic expansion
    return H0 / np.sqrt(1 + t / 13.8e9)  # Scale the Hubble parameter over the age of the universe

# Placeholder functions for temperature, atmospheric conditions, etc.
def f(T, A, B):
    return 0.1 * T * A * np.log(1 + B)  # Modified feedback with logarithmic interaction

def g(T, A, B):
    return 0.05 * T + 0.2 * A - 0.1 * B  # Linear relationship

def h(T, A, B):
    return 0.03 * B * np.exp(-0.05 * T) + 0.02 * A  # Biodiversity decreases exponentially with temperature

def L(phi):
    return 0.1 * phi * (1 - phi) * np.tanh(phi)  # Consciousness field with damping

def field_dynamics(phi):
    return phi * (1 - phi)  # Logistic growth for consciousness field

# Define the differential equation system
def model(y, t):
    a, v, T, A, B, x, y_b, phi = y
    
    # Hubble parameter based on cosmic time t (scaled to 13.8 billion years)
    H = Hubble(t)
    
    # Cosmic expansion equation (Friedmann equation term)
    cosmic_term = - (4 * np.pi * G / 3) * (rho + 3 * p / c**2) * a
    
    # Self-regulation term (example feedback)
    self_regulation = alpha * f(T, A, B)
    
    # Autopoiesis term (self-organization)
    autopoiesis = beta * (alpha * x - beta * x * y_b)
    
    # Biodiversity term
    biodiversity = gamma * h(T, A, B)
    
    # Consciousness field term
    consciousness_field = delta * L(phi)
    
    # Differential equations for the system
    da_dt = v  # Rate of change of the scale factor
    dv_dt = cosmic_term + self_regulation + autopoiesis + biodiversity + consciousness_field
    dT_dt = f(T, A, B)  # Temperature dynamics
    dA_dt = g(T, A, B)  # Atmospheric dynamics
    dB_dt = h(T, A, B)  # Biodiversity dynamics
    dx_dt = alpha * x - beta * x * y_b  # Species dynamics
    dy_b_dt = gamma * y_b - delta * x * y_b  # Bird species dynamics
    dphi_dt = field_dynamics(phi)  # Consciousness field evolution
    
    return [da_dt, dv_dt, dT_dt, dA_dt, dB_dt, dx_dt, dy_b_dt, dphi_dt]

# Initial conditions (example values)
a0 = 1.0  # Initial scale factor at Big Bang
v0 = 0.0  # Initial velocity (no initial expansion)
T0 = 3000  # Initial temperature after recombination (in Kelvin)
A0 = 1.0  # Initial atmospheric conditions
B0 = 1.0  # Initial biodiversity conditions
x0 = 1.0  # Initial population of species x
y0 = 1.0  # Initial population of bird species
phi0 = 0.1  # Initial consciousness field strength

# Time array from Big Bang (0) to present (13.8 billion years)
t = np.linspace(0, 13.8e9, 1000)  # Time from 0 to 13.8 billion years, with 1000 time steps

# Initial state vector
y0 = [a0, v0, T0, A0, B0, x0, y0, phi0]

# Solve the system of differential equations
solution = odeint(model, y0, t)

# Extract the solution components
a_sol = solution[:, 0]
v_sol = solution[:, 1]
T_sol = solution[:, 2]
A_sol = solution[:, 3]
B_sol = solution[:, 4]
x_sol = solution[:, 5]
y_b_sol = solution[:, 6]
phi_sol = solution[:, 7]

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(t, a_sol)
plt.title('Scale Factor (a) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Scale Factor')

plt.subplot(3, 2, 2)
plt.plot(t, v_sol)
plt.title('Velocity (v) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Velocity')

plt.subplot(3, 2, 3)
plt.plot(t, T_sol)
plt.title('Temperature (T) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Temperature (K)')

plt.subplot(3, 2, 4)
plt.plot(t, A_sol)
plt.title('Atmospheric Conditions (A) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Atmospheric Conditions')

plt.subplot(3, 2, 5)
plt.plot(t, B_sol)
plt.title('Biodiversity (B) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Biodiversity')

plt.subplot(3, 2, 6)
plt.plot(t, phi_sol)
plt.title('Consciousness Field (phi) Over Time')
plt.xlabel('Time (years)')
plt.ylabel('Consciousness Field')

plt.tight_layout()
plt.show()
