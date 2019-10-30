import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as cst

theta,I = np.genfromtxt('Polarisation.txt',unpack=True)
I=I*10**(-6)
theta=theta*2*cst.pi/360

def Polarisation(theta, t0, I0):
    return I0*np.cos(theta+t0)**2

plt.plot(theta,I, 'k.', label ='Messwerte')
plt.xlabel(r'$\Theta/\si{\radian}$')
plt.ylabel(r'$I/\si{\ampere}$')
params, covariance_matrix = curve_fit(Polarisation, theta, I)
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(0, 2*cst.pi, 1000)
plt.plot(x_plot, Polarisation(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/plot.pdf')
