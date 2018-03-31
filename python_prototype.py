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
def nonlin(x,y=0):
	if(y==1):	
		return x*(1-x)

	if(y==2):	
		return x*(1-x)*(1-2*x)

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
	l1d = np.dot(l0d,syn0)*nonlin(l1,1)
	l2d = np.dot(l1d,syn1)*nonlin(l2,1)
	l3d = np.dot(l2d,syn2)*nonlin(l3,1)

	# define error	
	l3_error = y - l3

	# backward propogation	
	l3_delta = l3_error * nonlin(l3,1)

	l2_error = l3_delta.dot(syn2.T)
	l2_delta = l2_error * nonlin(l2,1)

	l1_error = l2_delta.dot(syn1.T)	
	l1_delta = l1_error * nonlin(l1,1)

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

for iter in xrange(20000):

	# forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l3 = nonlin(np.dot(l2,syn2))

	# forward propogation of derivative
	#l0d = np.ones((dataset,input_dim))
	#l1d = np.dot(l0d,syn0)*nonlin(l1,1)
	#l2d = np.dot(l1d,syn1)*nonlin(l2,1)
	#l3d = np.dot(l2d,syn2)*nonlin(l3,1)	

	# define error
	dEdl3 = (y - l3) * 1

	# define error on the derivative
	dEdl3d = (z - l3d)*0

	
	#Regular backprop
	#l3_delta = dEdl3 * nonlin(l3,1)

	#dEdl2 = l3_delta.dot(syn2.T)
	#l2_delta = dEdl2 * nonlin(l2,1)


	#dEdl1 = l2_delta.dot(syn1.T)
	#l1_delta = dEdl1 * nonlin(l1,1)
	

	#syn0 += np.dot(l0.T,l1_delta)
	#syn1 += np.dot(l1.T,l2_delta)
	#syn2 += np.dot(l2.T,l3_delta)

	#Full backprop
	l3_delta1 = dEdl3 * nonlin(l3,1)
	l3_delta2 = dEdl3d  * nonlin(l3,2) * np.dot(l2d,syn2)
	l3_delta4 = dEdl3d * nonlin(l3,1)

	dEdl2 = (l3_delta1 + l3_delta2).dot(syn2.T)
	dEdl2d = l3_delta4.dot(syn2.T)

	l2_delta1 = dEdl2 * nonlin(l2,1)
	l2_delta2 = dEdl2d * nonlin(l2,2) * np.dot(l1d,syn1)
	l2_delta4 = dEdl2d * nonlin(l2,1) 
	
	dEdl1 = (l2_delta1 + l2_delta2).dot(syn1.T)
	dEdl1d = l2_delta4.dot(syn1.T)

	l1_delta1 = dEdl1 * nonlin(l1,1)
	l1_delta2 = dEdl1d * nonlin(l1,2) * np.dot(l0d,syn0)
	l1_delta4 = dEdl1d * nonlin(l1,1) 	

	syn0 += np.dot(l0.T,l1_delta1) + np.dot(l0.T,l1_delta2) + np.dot(l0d.T,l1_delta4)
	syn1 += np.dot(l1.T,l2_delta1) + np.dot(l1.T,l2_delta2) + np.dot(l1d.T,l2_delta4)
	syn2 += np.dot(l2.T,l3_delta1) + np.dot(l2.T,l3_delta2) + np.dot(l2d.T,l3_delta4)
	

	#dEdl3*nonlin(l3,2)*np.dot(l2d,syn2)
	#combined backprop
	#l3_delta = dEdl3 * nonlin(l3,1) + dEdl3d*nonlin(l3,1) + dEdl3d*nonlin(l3,2)*np.dot(l2d,syn2)
	#l3_delta = dEdl3 * nonlin(l3,1) + dEdl3d*nonlin(l3,1) 
	#l3_deltad = dEdl3d*nonlin(l3,2)*np.dot(l2d,syn2)
	
	#dEdl2 = (dEdl3*nonlin(l3,1) + dEdl3d*nonlin(l3,2)*np.dot(l2d,syn2)).dot(syn2.T)
	#dEdl2d = (dEdl3d*nonlin(l3,1)).dot(syn2.T)

	#l2_delta = dEdl2 * nonlin(l2,1) + dEdl2d*nonlin(l2,1) + dEdl2d*nonlin(l2,2)*np.dot(l1d,syn1)
	#l2_delta = dEdl2 * nonlin(l2,1) + dEdl2d*nonlin(l2,1)
	#l2_deltad = dEdl2d*nonlin(l2,2)*np.dot(l1d,syn1)

	#dEdl1 = (dEdl2*nonlin(l2,1) + dEdl2d*nonlin(l2,2)*np.dot(l1d,syn1)).dot(syn1.T)
	#dEdl1d = (dEdl2d*nonlin(l2,1)).dot(syn1.T)
	
	#l1_delta = dEdl1 * nonlin(l1,1) + dEdl1d*nonlin(l1,1) + dEdl1d*nonlin(l1,2)*np.dot(l0d,syn0)
	#l1_delta = dEdl1 * nonlin(l1,1) + dEdl1d*nonlin(l1,1) 
	#l1_deltad = dEdl1d*nonlin(l1,2)*np.dot(l0d,syn0)

	# backward propogation	
	#l3_delta = l3_error * nonlin(l3,1)

	#l2_error = l3_delta.dot(syn2.T)
	#l2_delta = l2_error * nonlin(l2,1)

	#l1_error = l2_delta.dot(syn1.T)	


	# backward propogation of 1st derivative error
	#l3_der_delta1 = np.dot(l2d,syn2)*nonlin(l3,2)*l3_der_error
	#l3_der_delta2 = nonlin(l3,1)*l3_der_error

	#l2_der_error = l3_der_delta1.dot(syn2.T) + l3_der_delta2.dot(syn2.T)
	#l2_der_delta1 = np.dot(l1d,syn1)*nonlin(l2,2)*l2_der_error
	#l2_der_delta2 = nonlin(l2,1)*l2_der_error

	#l1_der_error = l2_der_delta1.dot(syn1.T) + l2_der_delta2.dot(syn1.T)
	#l1_der_delta1 = np.dot(l0d,syn0)*nonlin(l1,2)*l1_der_error
	#l1_der_delta2 = nonlin(l1,1)*l1_der_error


	#redoing
	# update weights
	#syn2 += np.dot(l2.T,dEdz_l3+dEdzdot_l3_1) + np.dot(l2d.T,dEdzdot_l3_2)
	#syn1 += np.dot(l1.T,dEdz_l2+dEdzdot_l2_1) + np.dot(l1d.T,dEdzdot_l2_2)
	#syn0 += np.dot(l0.T,dEdz_l1+dEdzdot_l1_1) + np.dot(l0d.T,dEdzdot_l1_2)

	#syn0 += np.dot(l0.T,l1_delta) + np.dot(l0d.T,l1_deltad)
	#syn1 += np.dot(l1.T,l2_delta) + np.dot(l1d.T,l2_deltad)
	#syn2 += np.dot(l2.T,l3_delta) + np.dot(l2d.T,l3_deltad) 

	#syn2 += np.dot(l2.T,l3_der_delta1) + np.dot(l2d.T,l3_der_delta2)
	#syn1 += np.dot(l1.T,l2_der_delta1) + np.dot(l1d.T,l2_der_delta2)
	#syn0 += np.dot(l0.T,l1_der_delta1) + np.dot(l0d.T,l1_der_delta2)

	#print np.dot(l2.T,l3_delta)
	#print np.dot(l2.T,l3_der_delta1) + np.dot(l2d.T,l3_der_delta2)
	#print " "
	
	#exit()

	# print error
	if (iter % 1000 == 0):
		print "Error: " +str(np.mean(np.abs(dEdl3)))
		#print "Derivative Error: " +str(np.mean(np.abs(dEdl3d)))
		print " "

		
print(l3)
plt.figure()
plt.plot(X,y,X,l3)
plt.show()

plt.figure()
plt.plot(X,np.gradient(l3[:,0],X[1]-X[0]),X,l3d,X,z)
plt.show()

