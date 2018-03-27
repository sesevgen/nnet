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
dataset = 1

# sigmoid function
def nonlin(x,deriv=False):
	if(deriv==True):	
		return x*(1-x)

	return 1/(1+np.exp(-x))
    
# input dataset
X = np.linspace(0.2,1,dataset)
    
# output dataset            
y = np.power(X,2) + 0.0*np.random.random(dataset)

# derivative dataset
z = 2*X

# switch to column format
X = np.reshape(X,(dataset,1))
y = np.reshape(y,(dataset,1))

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

	print l0
	print l1
	print l2
	print l3

	print " "
	print " "

	# forward propogation of derivative
	l0d = np.ones((dataset,input_dim))
	l1d = np.dot(l0d,syn0)
	print l1d
	print nonlin(l1,True)
	print syn0	
	print " " 
	print l1d*nonlin(l1,True)	

	#l1d = nonlin(l1,True)np.dot(l0d,syn0)
	#print l1d
	#l2d = nonlin(l2,True)*l1d
	#l3d = nonlin(l3,True)*l2d

	exit()

	# backward propogation
	l3_error = y - l3
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

print "Output After Training:"
plt.figure()
plt.plot(X,y,X,l3)
plt.show()

plt.figure()
plt.plot(X,np.gradient(y[:,0],X[1]-X[0]),X,l3d)
plt.show()

print np.gradient(y[:,0],X[1]-X[0])
print l3d
exit()

# now try to teach from derivative

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

	# backward propogation
	l3_der_error = z - l3

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

print "Output After Training:"
plt.figure()
plt.plot(X,y,X,l3)
plt.show()




