#Credit for original:
#iamtrask.github.io

import numpy as np
import matplotlib.pyplot as plt

#Define layers, always 2 hidden layers but can change size.
input_dim =1
hidden_layer1 = 4
hidden_layer2 = 2
output_dim = 1

#Training data size
dataset = 50

# sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):	
		return x*(1-x)

	return 1/(1+np.exp(-x))
    
# input dataset
X = np.linspace(0.0,1,dataset)
    
# output dataset            
y = np.power(X,2) + 0.0*np.random.random(dataset)

# switch to column format
X = np.reshape(X,(dataset,1))
y = np.reshape(y,(dataset,1))

# derivative dataset
z = 2*X

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((input_dim,hidden_layer1)) - 1
syn1 = 2*np.random.random((hidden_layer1,hidden_layer2)) - 1
syn2 = 2*np.random.random((hidden_layer2,output_dim)) - 1


for iter in xrange(20000):

	# forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l3 = nonlin(np.dot(l2,syn2))

	# forward propogation of derivative
	l0d = np.ones((dataset,input_dim))
	l1d = np.dot(l0d,syn0)*nonlin(l1,True)
	l2d = np.dot(l1d,syn1)*nonlin(l2,True)
	l3d = np.dot(l2d,syn2)*nonlin(l3,True)

	# define error	
	l3_error = y - l3

	# backward propogation	
	l3_delta = l3_error * nonlin(l3,deriv=True)

	l2_error = l3_delta.dot(syn2.T)
	l2_delta = l2_error * nonlin(l2,deriv=True)

	l1_error = l2_delta.dot(syn1.T)	
	l1_delta = l1_error * nonlin(l1,deriv=True)

	# update weights
	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)
	syn2 += np.dot(l2.T,l3_delta)

	# print error
	if (iter % 1000 == 0):
		print "Error: " +str(np.mean(np.abs(l3_error)))

plt.figure()
plt.plot(X,y,X,l3)
plt.show()

plt.figure()
plt.plot(X,np.gradient(l3[:,0],X[1]-X[0]),X,l3d)
plt.show()

# now try to teach from derivative

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((input_dim,hidden_layer1)) - 1
syn1 = 2*np.random.random((hidden_layer1,hidden_layer2)) - 1
syn2 = 2*np.random.random((hidden_layer2,output_dim)) - 1


for iter in xrange(10000):

	# forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l3 = nonlin(np.dot(l2,syn2))

	# forward propogation of derivative
	l0d = np.ones((dataset,input_dim))
	l1d = np.dot(l0d,syn0)*nonlin(l1,True)
	l2d = np.dot(l1d,syn1)*nonlin(l2,True)
	l3d = np.dot(l2d,syn2)*nonlin(l3,True)

	# define error
	l3_error = (y-l3)	

	# define error on the derivative
	l3_der_error = (z - l3d)	

	# backward propogation
	# second derivative of nonlinearity is simply f*f'
	l3_delta = l3_der_error*(nonlin(l3,True)*l3d+nonlin(l3,True)*l3*l3d

	l2_error = l3_delta.dot(syn2.T)
	l2_delta = l2_error*(nonlin(l2,True)*l2)*l2d

	l1_error = l2_delta.dot(syn1.T)	
	l1_delta = l1_error*(nonlin(l1,True)*l1)*l1d

	# update weights
	syn0 += np.dot(l0.T,l1_delta)
	syn1 += np.dot(l1.T,l2_delta)
	syn2 += np.dot(l2.T,l3_delta)

	# print error
	if (iter % 1000 == 0):
		print "Error: " +str(np.mean(np.abs(l3_der_error)))

plt.figure()
plt.plot(X,y,X,l3)
plt.show()

plt.figure()
plt.plot(X,np.gradient(l3[:,0],X[1]-X[0]),X,l3d,X,z)
plt.show()

