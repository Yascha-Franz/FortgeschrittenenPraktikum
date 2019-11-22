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

#\alpha/° A_{3.666kHz}/V A_{6.174kHz}/V A_{7.379kHz}
alpha, A_3, A_6, A_7 = np.genfromtxt('scripts/Kugel_Winkel.txt', unpack=True)
alpha *= (2*np.pi)/360
Theta = np.arccos(0.5*(np.cos(alpha)-1))
A_3=A_3/max(A_3)
A_6=A_6/max(A_6)
A_7=A_7/max(A_7)
degree=360/(2*np.pi)

def Y00(x):
    return 1+0*x

def Y10(x):
    return abs((np.cos(x)))
    
def Y20(x):
    return abs(3*np.cos(x)**2-1)/2

def Y30(x):
    return abs(5*np.cos(x)**3-3*np.cos(x))/2

def Y40(x):
   return abs(35*np.cos(x)**4-30*np.cos(x)**2+3)/8

def Y50(x):
   return abs(np.cos(x)*(63*np.cos(x)**4-70*np.cos(x)**2+15))/8

def Y70(x):
   return abs(np.cos(x)*(429*np.cos(x)**6-693*np.cos(x)**4+315*np.cos(x)**2-35))/16

def Y11(x,y):
    return abs((np.cos(x)*np.cos(y)))
   
def Y21(x,y):
    return np.sqrt(((3*np.cos(x)**2-1)*(np.cos(y)))**2+((3*np.cos(x)**2-1)*(np.sin(y)))**2)/2


def polarplot(x,y,f,savepath):
    ax=plt.figure().gca(polar=True)
    ax.plot(x, y, 'kx', label ='Messwerte')
    x_plot = np.linspace(np.min(x), np.max(x), 1000)
    ax.plot(x_plot, f(x_plot), 'b-', label=f.__name__)
    ax.legend(loc='best')
    ax.set_thetamin(np.min(x)*degree)
    ax.set_thetamax(np.max(x)*degree)
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)


    plt.savefig(savepath)
    plt.clf()

polarplot(Theta,A_3,Y20,'build/Kugel_Winkel_3.pdf')
polarplot(Theta,A_6,Y40,'build/Kugel_Winkel_6.pdf')
polarplot(Theta,A_7,Y50,'build/Kugel_Winkel_7.pdf')

alpha, A_21, A_22=np.genfromtxt('scripts/Kugel_9mm_Ring.txt', unpack=True)
alpha=alpha/degree
A_21=A_21/max(A_21)
A_22=A_22/max(A_22)
polarplot(alpha,A_21,Y00,'build/Kugel_Ring_21.pdf')
polarplot(alpha,A_22,Y10,'build/Kugel_Ring_22.pdf')