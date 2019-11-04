import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as cst


#Polarisation
theta,I = np.genfromtxt('Polarisation.txt',unpack=True)
I=I*10**(-6)
theta=theta*2*cst.pi/360

def Polarisation(theta, t0, I0):
    return I0*np.cos(theta+t0)**2

plt.plot(theta,I, 'kx', label ='Messwerte')
plt.xlabel(r'$\Theta/\si{\radian}$')
plt.ylabel(r'$I/\si{\ampere}$')
params, covariance_matrix = curve_fit(Polarisation, theta, I)
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(0, 2*cst.pi, 1000)
plt.plot(x_plot, Polarisation(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Polarisation.pdf')
plt.clf()

#Justage konkav:flach (140:inf)
L, I = np.genfromtxt('Justage_kon_flach.txt', unpack=True)

L*=10**(-2)
I*=10**(-6)

plt.plot(L,I, 'kx', label='Messwerte')
plt.xlabel(r'L/\si{\meter}')
plt.ylabel(r'I/\si{\ampere}')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Justage_kon_flach.pdf')
plt.clf()

#Justage konkav:konkav (140:140)
L, I = np.genfromtxt('Justage_kon_kon.txt', unpack=True)

L*=10**(-2)
I*=10**(-6)

plt.plot(L,I, 'kx', label='Messwerte')
plt.xlabel(r'L/\si{\meter}')
plt.ylabel(r'I/\si{\ampere}')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Justage_kon_kon.pdf')
plt.clf()


#Modenanalyse
l, I00, I01 = np.genfromtxt('Moden.txt',unpack=True)
#l*=10**(-3)
#I00*=10**(-6)
#I01*=10**(-9)

#00
def Mode_00(x, x0, I0, w):
    return I0*np.exp(-2*(x-x0)**2/w**2)

plt.plot(l,I00, 'kx', label ='Messwerte')
plt.xlabel(r'$\Delta X/\si{\milli\meter}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
params, covariance_matrix = curve_fit(Mode_00, l, I00)
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(l), np.max(l), 1000)
plt.plot(x_plot, Mode_00(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Moden_00.pdf')
plt.clf()

#01

def Mode_01(x, x0, I0, w,a):  
    #return (I0-a*x)*(x-x0)**2*np.exp(-2*(x-x0)**2/w**2)    #Achtung! Modifiziert unter der Annahme, dass unsere Laserleistung linear abnimmt
    return I0*(x-x0)**2*np.exp(-2*(x-x0)**2/w**2)

plt.plot(l*2,I01, 'kx', label ='Messwerte')
plt.xlabel(r'$\Delta X/\si{\milli\meter}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
params, covariance_matrix = curve_fit(Mode_01, l*2, I01, p0=(-0.46568761, 3.29542082, 12.97749986, 5))
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(l*2), np.max(l*2), 1000)
plt.plot(x_plot, Mode_01(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

print(params)

plt.savefig('build/Moden_01.pdf')
plt.clf()