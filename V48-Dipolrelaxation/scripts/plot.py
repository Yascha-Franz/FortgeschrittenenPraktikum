import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate
import scipy.constants as cst
from matplotlib.legend_handler import HandlerTuple

def plotfit(x,y,f,savepath,slice_=slice(None,None),yerr=None, p0=None, save=True, color='k', label='Messwerte',fullfit=False):
    colors = ['k', 'b', 'g', 'r', 'y']
    if (np.size(x[0])>1):
        param, error, p1 = plotfit(x[0],y[0],f,savepath,slice_=slice_,yerr=yerr[0], p0=p0, save = False, color=colors[0], label = label[0], fullfit = fullfit[0])
        params = [param]
        errors = [error]
        p = [p1]
        for i in range(1,np.shape(x)[0]):
            param, error, p1= plotfit(x[i],y[i],f,savepath,slice_=slice_,yerr=yerr[i], p0=p0, save = False, color=colors[i], label = label[i], fullfit = fullfit[i])
            params = np.append(params, [param], axis = 0)
            errors = np.append(errors, [error], axis = 0)
            p = np.append(p, [p1], axis = 0)
    else:
        if yerr is None:
            p1, = plt.plot(x,y, color=color, linestyle='', marker='.', label =label)
        else:
            p1, = plt.errorbar(x,y,yerr=yerr, color=color, linestyle='', marker='x', label =label)
        params, covariance_matrix = curve_fit(f, x[slice_], y[slice_],p0=p0)
        errors = np.sqrt(np.diag(covariance_matrix))
        if fullfit:
            x_plot = np.linspace(np.min(x), np.max(x), 1000)
        else :
            x_plot = np.linspace(np.min(x[slice_]), np.max(x[slice_]), 1000)
        p2, = plt.plot(x_plot, f(x_plot, *params), color=color, linestyle='-', label=f.__name__.replace("_", " "))
        p = [p1, p2]
        plt.legend(loc='best')
        plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
    if save:
        plt.savefig(savepath)
        plt.clf()
    return params, errors, p

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

def Lineare_Regression(x,a,b):
    return x*a +b
x,y = np.genfromtxt('scripts/data.txt',unpack=True)
plt.xlabel('X')
plt.ylabel('Y')
plotfit(x,y,Lineare_Regression,'build/plot.pdf')

U_E = 950#V
#KBr 0.005%Mol Sr

def Exponentieller_Untergrund(x,a,b,c):
    return a*np.exp(b*x)+c

T_2, t_2, I_2 = np.genfromtxt('scripts/2Grad.txt', unpack=True)
T_15, t_15, I_15 = np.genfromtxt('scripts/1,5Grad.txt', unpack=True)
T_2+=273.15
T_15+=273.15
null_2 = t_2>60
null_2[t_2==59] = True
null_2[t_2<34] = True
null_2[t_2<32] = False
null_2[t_2<8] = True


plt.xlabel(r'T/\si{\kelvin}')
plt.ylabel(r'I/\si{\pico\ampere}')
params_2, erros_2, p_2 = plotfit(T_2, I_2, Exponentieller_Untergrund, None, slice_=null_2, fullfit=True, save = False, p0=(0.42, 0.08, -0.42))
p2, = plt.plot(T_2[~null_2], I_2[~null_2], 'b.')
plt.legend([(p_2[0], p2),(p_2[1])], ['Messwerte', 'Exponentieller Untergrund'], handler_map = {tuple: HandlerTuple(ndivide=None)})
plt.savefig('build/I_mit_untergrund_2.pdf')
plt.clf()
print("Exponentieller Untergrund 2Grad/min")
print("a, b, c")
print(params_2)
print(erros_2)
print()

null_15 = t_15>78
null_15[t_15<42] = True
null_15[t_15<40] = False
null_15[t_15<7] = True
plt.xlabel(r'T/\si{\kelvin}')
plt.ylabel(r'I/\si{\pico\ampere}')
params_15, erros_15, p_15 = plotfit(T_15, I_15, Exponentieller_Untergrund, None, slice_=null_15, fullfit=True, save = False, p0=(1.2, 0.03,-1.1))
p15, = plt.plot(T_15[~null_15], I_15[~null_15], 'b.')
plt.legend([(p_15[0], p15),(p_15[1])], ['Messwerte', 'Exponentieller Untergrund'], handler_map = {tuple: HandlerTuple(ndivide=None)})
plt.savefig('build/I_mit_untergrund_15.pdf')
plt.clf()

print("Exponentieller Untergrund 1.5Grad/min")
print("a, b, c")
print(params_15)
print(erros_15)
print()

I_2_korrigiert = I_2 - Exponentieller_Untergrund(T_2, *params_2)
I_15_korrigiert = I_15 - Exponentieller_Untergrund(T_15, *params_15)

plt.xlabel(r'T/\si{\kelvin}')
plt.ylabel(r'I/\si{\pico\ampere}')
plt.plot(T_2, I_2_korrigiert, 'b.', label='Bereinigte Messwerte')
plt.legend()
plt.savefig('build/I_2.pdf')
plt.clf()

plt.xlabel(r'T/\si{\kelvin}')
plt.ylabel(r'I/\si{\pico\ampere}')
plt.plot(T_15, I_15_korrigiert, 'b.', label='Bereinigte Messwerte')
plt.legend()
plt.savefig('build/I_15.pdf')
plt.clf()

mask=T_2>232
mask[T_2>263]=False

plt.xlabel(r'$\frac{1}{k_bT}/\si{\per\joule}$')
plt.ylabel(r'$\text{ln}\left(I/\si{\pico\ampere}\right)$')
params_Anlauf_2, errors_Anlauf_2, p = plotfit(1/(cst.k*T_2[mask]), np.log(I_2_korrigiert[mask]), Lineare_Regression, 'build/Anlauf_2.pdf')

mask=T_15>245
mask[T_15>260]=False

plt.xlabel(r'$\frac{1}{k_bT}/\si{\per\joule}$')
plt.ylabel(r'$\text{ln}\left(I/\si{\pico\ampere}\right)$')
params_Anlauf_15, errors_Anlauf_15, p = plotfit(1/(cst.k*T_15[mask]), np.log(I_15_korrigiert[mask]), Lineare_Regression, 'build/Anlauf_15.pdf')

print("Anlauf 2Grad/min")
print("W, b")
print(params_Anlauf_2)
print(errors_Anlauf_2)
print()

print("Anlauf 1.5Grad/min")
print("W, b")
print(params_Anlauf_15)
print(errors_Anlauf_15)
print()

def f(I, T):
    return_ = integrate.simps(I, T)/I[0]
    for i in range(1,np.size(T)-1):
        return_ = np.append(return_, integrate.simps(I[i:], T[i:]/I[i]))
    return np.log(return_)

mask=T_2>232
mask[T_2>263]=False
mask2=T_2>232
mask2[T_2>266]=False

plt.xlabel(r'$\frac{1}{k_bT}/\si{\per\joule}$')
plt.ylabel('f(T)')
params_Integriert_2, errors_Integriert_2 , p = plotfit(1/(cst.k*T_2[mask]), f(I_2_korrigiert[mask2], T_2[mask2]), Lineare_Regression, 'build/Integriert_2.pdf')

mask=T_15>245
mask[T_15>260]=False
mask2=T_15>245
mask2[T_15>262]=False

plt.xlabel(r'$\frac{1}{k_bT}/\si{\per\joule}$')
plt.ylabel('f(T)')
params_Integriert_15, errors_Integriert_15 , p = plotfit(1/(cst.k*T_15[mask]), f(I_15_korrigiert[mask2], T_15[mask2]), Lineare_Regression, 'build/Integriert_15.pdf')

print("Integriert 2Grad/min")
print("W, b")
print(params_Integriert_2)
print(errors_Integriert_2)
print()

print("Integriert 1.5Grad/min")
print("W, b")
print(params_Integriert_15)
print(errors_Integriert_15)
print()
