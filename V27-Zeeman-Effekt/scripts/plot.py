import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as cst
from scipy.optimize import curve_fit

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None, save=True, color='k', label='Messwerte'):
    colors = ['k', 'b', 'g', 'r', 'y']
    if (np.size(x[0])>1):
        param, error = plotfit(x[0],y[0],f,savepath,slice_=slice_,yerr=yerr[0], p0=p0, save = False, color=colors[0], label = label[0])
        params = [param]
        errors = [error]
        for i in range(1,np.shape(x)[0]):
            param, error = plotfit(x[i],y[i],f,savepath,slice_=slice_,yerr=yerr[i], p0=p0, save = False, color=colors[i], label = label[i])
            params = np.append(params, [param], axis = 0)
            errors = np.append(errors, [error], axis = 0)
    else:
        if yerr is None:
            plt.plot(x,y, color=color, linestyle='', marker='.', label =label)
        else:
            plt.errorbar(x,y,yerr=yerr, color=color, linestyle='', marker='x', label =label)
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
plotfit(x,y,Fit,'build/plot.pdf')

#Eichung
B, I = np.genfromtxt('scripts/feldeichung.txt', unpack = True)
plt.xlabel(r'$I/\si{\ampere}$')
plt.ylabel(r'$B/\si{\milli\tesla}$')
params, errors = plotfit(I, B, Fit, 'build/feldeichung.pdf')
print('Eichung')
print('A/\si{\milli\\tesla\per\\ampere}, B_0/\si{\milli\\tesla}')
print(*params)
print(*errors)

B_ = unp.uarray(params, errors)

def B(I):
    return (B_[0]*I + B_[1])*10**(-3)

#Lande-Faktoren
print()
print('Lande-Faktoren')

def dellambda(ds,Ds,Dl):
    dl = (ds/Ds*Dl)/2
    return unp.uarray(np.mean(dl), np.std(dl))

def g_j(dl, l, B):
    return cst.h * cst.c/(l**2 * cst.value('Bohr magneton') * B) * dl

I = [4, 2.2, 6]   #rot, blau_sigma, blau_pi
linien = ['rot', 'blau_sigma', 'blau_pi']

rot_D, rot_d = np.genfromtxt('scripts/rot.txt', unpack = True)
blau_sigma_D, blau_sigma_d = np.genfromtxt('scripts/blau_sigma.txt', unpack = True)
blau_pi_D, blau_pi_d = np.genfromtxt('scripts/blau_pi.txt', unpack = True)
Ds = [rot_D, blau_sigma_D, blau_pi_D]
ds = [rot_d, blau_sigma_d, blau_pi_d]
Dl = np.array([48.913, 26.952, 26.952])
l = np.array([643.8, 480, 480])
Dl *= 10**(-12)
l *= 10**(-9)

for i in range(0,3):
    print(linien[i])
    print('B = ', B(I[i]))
    print('g_j = ', g_j(dellambda(ds[i], Ds[i], Dl[i]), l[i], B(I[i])))