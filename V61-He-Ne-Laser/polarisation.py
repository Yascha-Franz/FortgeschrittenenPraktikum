import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
#from pylab import *

phi = np.array([0, 0.262, 0.524, 0.785, 1.047, 1.309, 1.571, 1.833, 2.094, 2.356, 2.617, 2.878, 3.142, 3.403, 3.665, 3.927, 4.189, 4.451, 4.712, 4.974, 5.236, 5.498, 5.760, 6.021])
I = np.array([])
#genfromtxt funktioniert hier irgendwie nicht

plt.plot(phi, I, 'x', label=r'Messwerte')
plt.legend()
plt.grid()
plt.xlabel('p')
plt.ylabel('A')


#plt.savefig('pol_plot.pdf')
plt.show()
