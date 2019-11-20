import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

L,F_1,phi_1,A_1,F_2,phi_2,A_2 = np.genfromtxt('scripts/Zylinder.txt',unpack=True)

y=F_2-F_1
x=L
def f(L,c):
    return c/(2*L)

plt.plot(x,y, 'kx', label ='Messwerte')
plt.xlabel(r'$L/\si{\milli\meter}$')
plt.ylabel(r'$\Delta F/\si{\kilo\hertz}$')
params, covariance_matrix = curve_fit(f, x, y, p0 = (1))
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(x), np.max(x), 1000)
plt.plot(x_plot, f(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot.pdf')
plt.clf()

print("Lichtgeschwindigkeit")
print(params)
print(errors)