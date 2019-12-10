import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def plotfit(x,y,f,savepath,slice_=slice(0,None),yerr=None, p0=None):
    if yerr==None:
        plt.plot(x,y, 'k.', label ='Messwerte')
    else:
        plt.errorbar(x,y,yerr=yerr, fmt='kx', label ='Messwerte')
    params, covariance_matrix = curve_fit(f, x[slice_], y[slice_], p0=p0)
    errors = np.sqrt(np.diag(covariance_matrix))
    x_plot = np.linspace(np.min(x[slice_]), np.max(x[slice_]), 1000)
    plt.plot(x_plot, f(x_plot, *params), 'b-', label=f.__name__)
    plt.legend(loc='best')
    plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)

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