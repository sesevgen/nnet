import numpy as np
import matplotlib.pyplot as plt


npoints = 500
noise = 0.5

X = np.linspace(-3.5,3.5,1000)
Y = np.exp(X/2) + np.sin(3*X) + np.power(X,2) -np.power(X,4)/10

X2 = np.linspace(-3.5,3.5,npoints)
Y2 = np.exp(X2/2) + np.sin(3*X2) + np.power(X2,2) -np.power(X2,4)/10
Z = Y2 + (np.random.normal(0,noise,npoints))


np.savetxt('X.txt',X)
np.savetxt('Y.txt',Y)
np.savetxt('X2.txt',X2)
np.savetxt('Z.txt',Z)


fig = plt.figure()
plt.plot(X,Y)
plt.plot(X2,Z,'.')
plt.savefig('curveanddata.png')

fig = plt.figure()
plt.plot(X2,Z,'.')
plt.savefig('data.png')
