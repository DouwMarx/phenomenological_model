import matplotlib.pyplot as plt
import numpy as np
import sympy
import sympy as sym

zeta,omega_n,t,d_0,v_0 = sym.symbols("zeta,omega_n,t,d_0,v_0",real=True)

omega_d = omega_n*sym.sqrt(abs(zeta**2-1)) # Damped natural frequency

a = sym.exp(-zeta*omega_n*t)
b = d_0*sym.cos(omega_d*t)
c = ((v_0+zeta*omega_n*d_0)/omega_d)*sym.sin(omega_d*t)

x = a*(b+c)   # For 0<= zeta <1

# x = x.subs(d_0,0)
# x = x.subs(v_0,0)



xdot = sym.diff(x,t)
xdotdot = sym.diff(xdot,t)

xdotdot0 = xdotdot.subs(t,0).subs(d_0,0)

from sympy.solvers import solve
from sympy import Symbol


# v_0_for_t0_d0 = solve(xdotdot0 - 1, v_0)

# If we let initial acceleration be equal to 1
# v_0_for_t0_d0 = -1/(2*omega_n*zeta)

v_0_for_t0_d0 = -1/(2*omega_n*zeta)


print(xdotdot)
print(xdotdot0)
print(v_0_for_t0_d0)

xdotdot_implement = xdotdot.subs(d_0,0).subs(v_0,v_0_for_t0_d0)
print(xdotdot_implement.simplify(
))

numpy_str = str(xdotdot_implement)
numpy_str = numpy_str.replace("sin","np.sin")
numpy_str = numpy_str.replace("exp","np.exp")
numpy_str = numpy_str.replace("sqrt","np.sqrt")
numpy_str = numpy_str.replace("cos","np.cos")
numpy_str = numpy_str.replace("zeta","self.zeta")
numpy_str = numpy_str.replace("omega_n","self.omegan")
numpy_str = numpy_str.replace("Abs","np.abs")
numpy_str = numpy_str.replace("*t","*self.t_range")


ldf = sympy.lambdify([t,zeta,omega_n],xdotdot_implement)

r = [ldf(t,0.01,1000) for t in np.linspace(0,1,1000)]

plt.figure()
plt.plot(r)