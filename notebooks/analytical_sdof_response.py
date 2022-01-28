"""
Analytical differentiation of SDOF time reponse.

Intention is that this response should be used in the bearing model.
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy as sym
from sympy.solvers import solve

# Define symbols
zeta, omega_n, t, d_0, v_0 = sym.symbols("zeta,omega_n,t,d_0,v_0", real=True)


# Define terms in expression
omega_d = omega_n * sym.sqrt(abs(zeta ** 2 - 1))  # Damped natural frequency
a = sym.exp(-zeta * omega_n * t)
b = d_0 * sym.cos(omega_d * t)
c = ((v_0 + zeta * omega_n * d_0) / omega_d) * sym.sin(omega_d * t)

# Expression for time domain solution for underdamped SDOF system with initial conditions
x = a * (b + c)  # For 0<= zeta <1

# x = x.subs(d_0,0)
# x = x.subs(v_0,0)

# Differentiate displacement twice to get acceleration response
xdot = sym.diff(x, t)
xdotdot = sym.diff(xdot, t)

# Find the acceleration response at t=0
xdotdot0 = xdotdot.subs(t, 0).subs(d_0, 0)


# Solve for v_0 if t=0 and d=0 and the initial acceleration x_dotdot0 = 1
v_0_for_t0_d0 = solve(xdotdot0 - 1, v_0)[0]

# Substitute the initial conditions d_0 = 0, and the solved for v_0
xdotdot_implement = xdotdot.subs(d_0, 0).subs(v_0, v_0_for_t0_d0)
print("xdotdot to implement: ", xdotdot_implement.simplify())

# Turn the function into an expression to visualize it
ldf = sympy.lambdify([t, zeta, omega_n], xdotdot_implement)

# Show the acceleration response
r = [ldf(t, 0.01, 1000) for t in np.linspace(0, 1, 1000)]
plt.figure()
plt.plot(r)
