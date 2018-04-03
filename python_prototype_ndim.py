#Credit for original:
#iamtrask.github.io

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt


#Define layers, always 2 hidden layers but can change size.
input_dim = 2
hidden_layer1 = 12
hidden_layer2 = 8
output_dim = 1

importing = 1
loadw = 1

# sigmoid function
def nonlin(x,y=0):
	if(y==1):	
		return x*(1-x)

	if(y==2):	
		return x*(1-x)*(1-2*x)

	return 1/(1+np.exp(-x))

if (importing == 0):

	#Training data size
	dataset = 100

	# input dataset
	X = np.linspace(0.0,1,dataset)
		
	# output dataset            
	y = np.power(X,2) + 0.0*np.random.random(dataset)

	# switch to column format
	X = np.reshape(X,(dataset,1))
	y = np.reshape(y,(dataset,1))

	# derivative dataset
	z = 2*X

else:
	#X = np.loadtxt('adp_ann_example')
	X = np.loadtxt('adp_abf_example')
	
	#For testing only
	y = X[:,2]
	
	z = X[:,input_dim:]
	X = X[:,0:input_dim]
	dataset = X.shape[0]
	
	X = np.reshape(X,(dataset,input_dim))
	y = np.reshape(y,(dataset,output_dim))

	#scale data
	xshift = np.ones(2)
	xshift[0] = np.amin(X[:,0])
	xshift[1] = np.amin(X[:,1])

	X[:,0] = X[:,0] - xshift[0]
	X[:,1] = X[:,1] - xshift[1]

	yshift = np.amin(y)
	y = y - yshift

	xscale = np.ones(2)
	xscale[0] = np.amax(X[:,0])
	xscale[1] = np.amax(X[:,1])

	X[:,0] = X[:,0] / xscale[0]
	X[:,1] = X[:,1] / xscale[1]

	yscale = np.amax(y)
	y = y / yscale

	zshift = np.ones(2)
	zshift[0] = np.amin(z[:,0])
	zshift[1] = np.amin(z[:,1])

	z[:,0] = z[:,0] - zshift[0]
	z[:,1] = z[:,1] - zshift[1]

	zscale = np.ones(2)
	zscale[0] = np.amax(z[:,0])
	zscale[1] = np.amax(z[:,1])

	z[:,0] = z[:,0] / zscale[0]
	z[:,1] = z[:,1] / zscale[1]

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(2)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((input_dim,hidden_layer1)) - 1
syn1 = 2*np.random.random((hidden_layer1,hidden_layer2)) - 1
syn2 = 2*np.random.random((hidden_layer2,output_dim)) - 1

if(0):
	for iter in xrange(20000):

		# forward propagation
		l0 = X
		l1 = nonlin(np.dot(l0,syn0))
		l2 = nonlin(np.dot(l1,syn1))
		l3 = nonlin(np.dot(l2,syn2))

		# forward propogation of derivative
		l0d = np.ones((dataset,input_dim))
		l0da = np.zeros((dataset,input_dim))
		l0db = np.zeros((dataset,input_dim))
		l0da[:,0] = 1
		l0db[:,1] = 1
		
		l1da = np.dot(l0da,syn0)*nonlin(l1,1)
		l2da = np.dot(l1da,syn1)*nonlin(l2,1)
		l3da = np.dot(l2da,syn2)*nonlin(l3,1)

		l1db = np.dot(l0db,syn0)*nonlin(l1,1)
		l2db = np.dot(l1db,syn1)*nonlin(l2,1)
		l3db = np.dot(l2db,syn2)*nonlin(l3,1)

		# define error	
		l3_error = (y - l3)*0.01

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
			print "Error: " +str(np.mean(np.abs(l3_error))) + " ; Iteration: " + str(iter)


	xg = np.reshape(X[:,0], (31, 31))
	yg = np.reshape(X[:,1], (31, 31))
	zg = np.reshape(-y, (31, 31))
	l3g = np.reshape(-l3, (31,31))

	plt.figure()
	plt.contour(xg, yg, zg,colors='k')
	plt.contourf(xg, yg, zg)
	plt.savefig('training_data.png')

	plt.figure()
	plt.contour(xg, yg, l3g,colors='k')
	plt.contourf(xg, yg, l3g)
	plt.savefig('nn_output.png')

	plt.figure()
	plt.quiver(X[:,0],X[:,1],l3da,l3db)
	plt.savefig('nn_gradient.png')

	num_gradientx, num_gradienty = np.gradient(-l3g)
	plt.figure()
	plt.quiver(X[:,0],X[:,1],num_gradienty,num_gradientx)
	plt.savefig('num_gradient_of_nn_output.png')


	#plt.figure()
	#plt.plot(X,y,X,l3)
	#plt.contourf(X[:,0],X[:,1],y[:,0])
	#plt.savefig('fig1.png')
	#plt.show()

	#plt.figure()
	#plt.plot(X,np.gradient(l3[:,0],X[1]-X[0]),X,l3d)
	#plt.savefig('fig2.png')

	# now try to teach from derivative
	# initialize weights randomly with mean 0

if(1):

	print syn0.shape
	print syn1.shape
	print syn2.shape

	if(loadw == 0):
		syn0 = 2*np.random.random((input_dim,hidden_layer1)) - 1
		syn1 = 2*np.random.random((hidden_layer1,hidden_layer2)) - 1
		syn2 = 2*np.random.random((hidden_layer2,output_dim)) - 1

	else:
		syn0 = np.loadtxt('syn0')
		syn1 = np.loadtxt('syn1')
		syn2 = np.loadtxt('syn2')
		syn2 = np.reshape(syn2,(syn2.shape[0],1))
		print "loaded"

	print syn0.shape
	print syn1.shape
	print syn2.shape

	grad0 = syn0
	grad1 = syn1
	grad2 = syn2

	alpha = 0.01
	beta = 0.9

	for iter in xrange(20000):

		# forward propagation
		l0 = X
		l1 = nonlin(np.dot(l0,syn0))
		l2 = nonlin(np.dot(l1,syn1))
		l3 = nonlin(np.dot(l2,syn2))

		# forward propogation of derivative
		l0d = np.ones((dataset,input_dim))
		l0da = np.zeros((dataset,input_dim))
		l0db = np.zeros((dataset,input_dim))
		l0da[:,0] = 1
		l0db[:,1] = 1
		
		l1da = np.dot(l0da,syn0)*nonlin(l1,1)
		l2da = np.dot(l1da,syn1)*nonlin(l2,1)
		l3da = np.dot(l2da,syn2)*nonlin(l3,1)

		l1db = np.dot(l0db,syn0)*nonlin(l1,1)
		l2db = np.dot(l1db,syn1)*nonlin(l2,1)
		l3db = np.dot(l2db,syn2)*nonlin(l3,1)


		# define error
		dEdl3 = (y - l3) * 0

		# define error on the derivative
		dEdl3da = (z[:,0] - l3da[:,0]) * 1
		dEdl3db = (z[:,1] - l3db[:,0]) * 1

		dEdl3da = np.reshape(dEdl3da,(dataset,output_dim))
		dEdl3db = np.reshape(dEdl3db,(dataset,output_dim))

		#Full backprop
		l3_delta1 = dEdl3	* nonlin(l3,1)
		l3_delta2a = dEdl3da	* nonlin(l3,2)	* (l2da.dot(syn2))
		l3_delta2b = dEdl3db	* nonlin(l3,2)	* (l2db.dot(syn2))
		l3_delta3a = dEdl3da	* nonlin(l3,1)
		l3_delta3b = dEdl3db	* nonlin(l3,1)

		dEdl2 	= (l3_delta1 + l3_delta2a + l3_delta2b).dot(syn2.T)
		dEdl2da	= l3_delta3a.dot(syn2.T)
		dEdl2db	= l3_delta3b.dot(syn2.T)

		l2_delta1 = dEdl2	* nonlin(l2,1)
		l2_delta2a = dEdl2da	* nonlin(l2,2)	* (l1da.dot(syn1))
		l2_delta2b = dEdl2db	* nonlin(l2,2)	* (l1db.dot(syn1))
		l2_delta3a = dEdl2da	* nonlin(l2,1)
		l2_delta3b = dEdl2db	* nonlin(l2,1)

		dEdl1 	= (l2_delta1 + l2_delta2a + l2_delta2b).dot(syn1.T)
		dEdl1da	= l2_delta3a.dot(syn1.T)
		dEdl1db	= l2_delta3b.dot(syn1.T)

		l1_delta1 = dEdl1	* nonlin(l1,1)
		l1_delta2a = dEdl1da	* nonlin(l1,2)	* (l0da.dot(syn0))
		l1_delta2b = dEdl1db	* nonlin(l1,2)	* (l0db.dot(syn0))
		l1_delta3a = dEdl1da	* nonlin(l1,1)
		l1_delta3b = dEdl1db	* nonlin(l1,1)

		grad0prev = grad0
		grad1prev = grad1
		grad2prev = grad2

		grad0 = beta*grad0prev + (np.dot(l0.T,dEdl1*nonlin(l1,1)) + np.dot(l0da.T,dEdl1da*nonlin(l1,1)) + np.dot(l0.T,dEdl1da*nonlin(l1,2)*np.dot(l0da,syn0)) + np.dot(l0db.T,dEdl1db*nonlin(l1,1)) + np.dot(l0.T,dEdl1db*nonlin(l1,2)*np.dot(l0db,syn0)))/dataset

		grad1 = beta*grad1prev + (np.dot(l1.T,dEdl2*nonlin(l2,1)) + np.dot(l1da.T,dEdl2da*nonlin(l2,1)) + np.dot(l1.T,dEdl2da*nonlin(l2,2)*np.dot(l1da,syn1)) + np.dot(l1db.T,dEdl2db*nonlin(l2,1)) + np.dot(l1.T,dEdl2db*nonlin(l2,2)*np.dot(l1db,syn1)))/dataset

		grad2 = beta*grad2prev + (np.dot(l2.T,dEdl3*nonlin(l3,1)) + np.dot(l2da.T,dEdl3da*nonlin(l3,1)) + np.dot(l2.T,dEdl3da*nonlin(l3,2)*np.dot(l2da,syn2)) + np.dot(l2db.T,dEdl3db*nonlin(l3,1)) + np.dot(l2.T,dEdl3db*nonlin(l3,2)*np.dot(l2db,syn2)))/dataset

		syn0 += alpha*grad0
		syn1 += alpha*grad1
		syn2 += alpha*grad2

		#exit()

		# print error
		if (iter % 1000 == 0):
			print "Error: " +str(np.mean(np.abs(y - l3)))
			print "Derivative Error: " +str(np.mean(np.abs(z[:,0] - l3da[:,0])+np.abs(z[:,1] - l3db[:,0]))) + " ; Iteration: " + str(iter)
			print " "

	
	xg = np.reshape(X[:,0], (31, 31))
	yg = np.reshape(X[:,1], (31, 31))
	l3g = np.reshape(-l3, (31,31))

	plt.figure()
	plt.contour(xg, yg, l3g,colors='k')
	plt.contourf(xg, yg, l3g)
	plt.savefig('nn_output.png')

	plt.figure()
	plt.quiver(X[:,0],X[:,1],l3da,l3db)
	plt.savefig('nn_gradient.png')

	np.savetxt('syn0',syn0)
	np.savetxt('syn1',syn1)
	np.savetxt('syn2',syn2)
		
	#plt.figure()
	#plt.plot(X,y,X,l3)
	#plt.show()

	#plt.figure()
	#plt.plot(X,np.gradient(l3[:,0],X[1]-X[0]),X,l3d)
	#plt.show()

