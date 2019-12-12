import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import scipy.constants as cst

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None, save=True, color='k'):
    colors = ['k', 'b', 'g', 'r', 'y']
    if (np.size(x[0])>1):
        param, error = plotfit(x[0],y[0],f,savepath,slice_=slice_,yerr=yerr[0], p0=p0, save = False, color=colors[0])
        params = [param]
        errors = [error]
        for i in range(1,np.shape(x)[0]):
            param, error = plotfit(x[i],y[i],f,savepath,slice_=slice_,yerr=yerr[i], p0=p0, save = False, color=colors[i])
            params = np.append(params, [param], axis = 0)
            errors = np.append(errors, [error], axis = 0)
    else:
        if yerr is None:
            plt.plot(x,y, color=color, linestyle='', marker='.', label ='Messwerte')
        else:
            plt.errorbar(x,y,yerr=yerr, color=color, linestyle='', marker='x', label ='Messwerte')
        params, covariance_matrix = curve_fit(f, x[slice_], y[slice_],p0=p0)
        errors = np.sqrt(np.diag(covariance_matrix))
        x_plot = np.linspace(np.min(x[slice_]), np.max(x[slice_]), 1000)
        plt.plot(x_plot, f(x_plot, *params), color=color, linestyle='-', label=f.__name__)
        plt.legend(loc='best')
        plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    if save:
        plt.savefig(savepath)
        plt.clf()
    return params, errors

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

def Fit(x,a,b):
    return x*a +b
x,y = np.genfromtxt('scripts/data.txt',unpack=True)
plt.xlabel('X')
plt.ylabel('Y')
plotfit(x,y,Fit,'build/plot.pdf', slice_=(x<=7))

p, Amax, Amin = np.genfromtxt('scripts/dichteprofil.txt', unpack = True)
A = (Amax + Amin)/2
Aerr = (Amax - Amin)/2

p2, Amax2, Amin2 = np.genfromtxt('scripts/dichteprofil_Au.txt', unpack = True)
A2 = (Amax2 + Amin2)/2
Aerr2 = (Amax2 - Amin2)/2
plt.xlabel(r'$p/\si{\milli\bar}$')
plt.ylabel(r'$\overline{A}/\si{\volt}$')

params, errors = plotfit([p,p2],[A,A2], Fit,'build/dichteprofil.pdf', yerr=[Aerr,Aerr2])
param = unp.uarray(params[0], errors[0])
print("Dichteprofil ohne Folie:")
print("b, A_0:")
print(param)

param_Au = unp.uarray(params[1], errors[1])
print("Dichteprofil mit Folie:")
print("b, A_0:")
print(param_Au)

#Energieverlust in MeV
Ediff = 5.48*(1-param_Au[1]/param[1])
print("Energieverlust = ")
print(Ediff)

#Geschwindigkeit alpha Teilchen
E_mid = 1/2*5.48*1e6*cst.e*(1 + param_Au[1]/param[1])
v_alpha = unp.sqrt(2*E_mid/cst.value("alpha particle mass"))
print("E_mid:")
print(E_mid)
print("v_alpha = ")
print(v_alpha)

#Dicke der Goldfolie
n = (79*19.32*1e3)/(196.967*cst.value("atomic mass constant"))
log_arg = 2*cst.m_e*v_alpha**2/(10*79*cst.e)
zaehler = cst.m_e*v_alpha**2*(4*np.pi*cst.epsilon_0)**2
nenner = 4*np.pi*cst.e**4*2**2*n

dicke = -Ediff*1e6*cst.e*(zaehler/nenner)/unp.log(log_arg)
print("Dicke der Goldfolie:")
print(dicke)

def Rutherford(theta, c, theta0):
    return c/(np.sin((theta-theta0)*2*cst.pi/360/2)**4)

def streu(path, name, savepath,s):
    N, theta, t = np.genfromtxt(path, unpack = True)
    Nerr = np.sqrt(N)
    uN = unp.uarray(N,Nerr)
    n = uN/t
    plt.xlabel(r'$\Theta/°$')
    plt.ylabel(r'$N/\si{\becquerel}$')
    if s<=0:
        params, errors = plotfit(theta, unp.nominal_values(n), Rutherford, savepath, yerr=unp.std_devs(n), slice_=(theta<=s), p0=(0.001,2))
    else:
        params, errors = plotfit(theta, unp.nominal_values(n), Rutherford, savepath, yerr=unp.std_devs(n), slice_=(theta>=s), p0=(0.001,2))
    param = unp.uarray(params, errors)
    print(name)
    print("C, \Theta_0")
    print(param)
    return param[0]

c1 = streu('scripts/streu_bismut.txt', 'Bismut', 'build/streu_bismut.pdf',0)
c2 = streu('scripts/streu_gold.txt', 'Gold', 'build/streu_gold.pdf',-4)
c3 = streu('scripts/streu_platin.txt', 'Platin', 'build/streu_platin.pdf',0)

def C(Z, c_0):
    return c_0 * Z**2

c = np.array([unp.nominal_values(c1), unp.nominal_values(c2), unp.nominal_values(c3)])
cerr = np.array([unp.std_devs(c1), unp.std_devs(c2), unp.std_devs(c3)])
Z = np.array([83, 79, 78])
#plotfit(Z, unp.nominal_values(c), C, 'build/ordnung.pdf', yerr=unp.std_devs(c))
plt.xlabel('C')
plt.ylabel('Ordnungszahl Z')
params, errors = plotfit(Z, c, C, 'build/ordnung.pdf', yerr=cerr)