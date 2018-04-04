import numpy as np
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

X = np.loadtxt('X.txt')
fit = np.loadtxt('output.txt')
org = np.loadtxt('Y.txt')

#X2 = np.loadtxt('X2.txt')
#movavg = np.loadtxt('Z.txt')

#movavg = running_mean(movavg,5)
#print len(movavg)

fig = plt.figure()
plt.plot(X,fit)
plt.plot(X,org)
#plt.plot(X2[2:-2],movavg)
plt.savefig('fit.png')
