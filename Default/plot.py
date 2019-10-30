import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x,y = np.genfromtxt('data.txt',unpack=True)

def f(x,a,b):
    return x*a +b

plt.plot(x,y, 'k.', label ='Messwerte')
plt.xlabel('X')
plt.ylabel('Y')
params, covariance_matrix = curve_fit(f, x, y, p0 = (1,1))
errors = np.sqrt(np.diag(covariance_matrix))
x_plot = np.linspace(0, 10, 2)
plt.plot(x_plot, f(x_plot, *params), 'b-', label='Fit')
plt.legend(loc='best')

plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
