import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt

#X = np.loadtxt('adp_ann_example')
X = np.loadtxt('adp_abf_example')
Y = X[:,1]
Z = X[:,3]
X = X[:,0]

dX = np.loadtxt('dx.txt')
dY = np.loadtxt('dy.txt')

dXbp = np.loadtxt('dxy_bp.txt')
dYbp = dXbp[:,1]
dXbp = dXbp[:,0]

xg = np.reshape(X, (31, 31))
yg = np.reshape(Y, (31, 31))
zg = np.reshape(Z, (31, 31))

Znn = np.loadtxt('output.txt')
znng = np.reshape(Znn, (31, 31))

plt.figure()
plt.contour(xg, yg, zg,colors='k')
plt.contourf(xg, yg, zg)
plt.show()

plt.figure()
plt.contour(xg, yg, znng,colors='k')
plt.contourf(xg, yg, znng)
plt.show()

plt.figure()
plt.quiver(X,Y,dX,dY)
plt.show()

plt.figure()
plt.quiver(X,Y,dXbp,dYbp)
plt.show()


