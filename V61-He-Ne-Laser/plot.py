import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import scipy.constants as cst


#Polarisation
theta,I = np.genfromtxt('Polarisation.txt',unpack=True)
#I=I*10**(-6)
theta=theta*2*cst.pi/360

def Polarisation(theta, t0, I0):
    return I0*np.cos(theta+t0)**2

plt.plot(theta,I, 'kx', label ='Messwerte')
plt.xlabel(r'$\phi/\si{\radian}$')
plt.ylabel(r'$I/\si{\micro\ampere}$')
params, covariance_matrix = curve_fit(Polarisation, theta, I, p0=(0.5,100))
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(0, 2*cst.pi, 1000)
plt.plot(x_plot, Polarisation(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

print("Polarisation: \Theta_0 I_0")
print(params)
print(errors)

plt.savefig('build/Polarisation.pdf')
plt.clf()

#Justage konkav:flach (140:inf)
L, I = np.genfromtxt('Justage_kon_flach.txt', unpack=True)

#L*=10**(-2)
#I*=10**(-6)

plt.plot(L,I, 'kx', label='Messwerte')
plt.xlabel(r'L/\si{\centi\meter}')
plt.ylabel(r'I/\si{\micro\ampere}')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Justage_kon_flach.pdf')
plt.clf()

#Justage konkav:konkav (140:140)
L, I = np.genfromtxt('Justage_kon_kon.txt', unpack=True)

#L*=10**(-2)
#I*=10**(-6)

plt.plot(L,I, 'kx', label='Messwerte')
plt.xlabel(r'L/\si{\centi\meter}')
plt.ylabel(r'I/\si{\micro\ampere}')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Justage_kon_kon.pdf')
plt.clf()

#Justage Theorie
def konkav(x, r):
    return (r-x)**2/r**2
def flach(x, r):
    return (r-x)/r

x_plot=np.linspace(0,280,1000)
plt.plot(x_plot,konkav(x_plot,140), 'b-', label="Konkav:Konkav")
plt.plot(x_plot/2,flach(x_plot/2,140), 'k--', label="Konkav:Flach")
plt.xlabel(r'L/\si{\centi\meter}')
plt.ylabel(r'$g_1g_2$')

plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

plt.savefig('build/Justage_Theorie.pdf')
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

print("Mode_00: x_0 I_0 \omega")
print(params)
print(errors)

plt.savefig('build/Moden_00.pdf')
plt.clf()

#01

def Mode_01(x, x0, I0, w,a):  
    return (I0-a*x)*(x-x0)**2*np.exp(-2*(x-x0)**2/w**2)    #Achtung! Modifiziert unter der Annahme, dass unsere Laserleistung linear abnimmt
    #return I0*(x-x0)**2*np.exp(-2*(x-x0)**2/w**2)

plt.plot(l*2,I01, 'kx', label ='Messwerte')
plt.xlabel(r'$\Delta X/\si{\milli\meter}$')
plt.ylabel(r'$I/\si{\nano\ampere}$')
params, covariance_matrix = curve_fit(Mode_01, l*2, I01, p0=(-0.46568761, 3.29542082, 12.97749986, 5))
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(np.min(l*2), np.max(l*2), 1000)
plt.plot(x_plot, Mode_01(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

print("Mode_01: x_0 I_0 \omega")
print(params)
print(errors)

plt.savefig('build/Moden_01.pdf')
plt.clf()


print("Wellenlänge")

d=0.7
g=100*10**3
x=unp.uarray([0.045,0.045],[0.001,0.001])
l=1/g*unp.sin(unp.arctan(x/d))
print(unp.nominal_values(l))
print(unp.std_devs(l))