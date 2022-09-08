import numpy as np


#_____________________________________________________________________
#--this function evaluates basis functions at a particular reference node
# Input: xi: list of parametric coordinates
# output: evaluation of shape functions at xi

def Shape_Function(p, xi):
	x,y,z = xi[0],xi[1],xi[2]
	if p == 1:
		return np.array([1.-x-y-z,x,y,z])
	if p == 2:
		return np.array([(1-x-y-z)*(1-2*x-2*y-2*z), 
							x*(2*x-1), 
							y*(2*y-1),
							z*(2*z-1), 
							4*x*(1-x-y-z), 
							4*x*y, 
							4*y*(1-x-y-z),
							4*z*(1-x-y-z), 
							4*x*z, 
							4*y*z ])
#_____________________________________________________________________



#_____________________________________________________________________
#--this function evaluates basis function derivatives at a particular reference node
# Input: local_xi: local coordinates, 
# output: matrix of basis function derivatives 

def Shape_Deri(p, xi):
	x,y,z = xi[0],xi[1],xi[2]
	if p == 1:
		return np.array([[-1.0,-1.0,-1.0], [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]) #parametric derivative 4x3
	if p == 2:
		return np.array([   [-3 + 4*x + 4*y + 4*z,-3 + 4*x + 4*y + 4*z,-3 + 4*x + 4*y + 4*z],
							[4*x - 1, 0,0],
							[0,4*y-1,0],
							[0,0,4*z-1],
							[4 - 8*x - 4*y - 4*z, -4*x, -4*x],
							[4*y, 4*x, 0],
							[-4*y, 4 - 4*x - 8*y - 4*z, -4*y],
							[-4*z, -4*z, 4 - 4*x - 4*y - 8*z],
							[4*z,0,4*x],
							[0,4*z,4*y]])




#_____________________________________________________________________
#_____________________________________________________________________
#--this function computes the Jacobian matrix used for isoparametric mapping, at a particular reference node
# Input: p: polynomial deg
#        P: the physical tetra 4 node. 4x3 numpy array
#		 local_xi: local coordinates, 4x1, usually gaussian nodes, the point to evaluate in the parametric space
# output: the Jacobian matrix

def Jacobian(p,P, local_xi):
	# step 1: compute parametric derivative 4x3
	D_xi = Shape_Deri(p,local_xi)
	J = np.zeros((3,3))
	for i in [0,1,2]:  # rows
		for j in [0,1,2]: # cols
			J[i,j] = np.dot(D_xi[:,j], P[:,i]) 
	return J
#_____________________________________________________________________


#_____________________________________________________________________
#--this function represent a physical point in terms of parametric coordinates
# Input: P: the physical tetra 4 node. 4x3 numpy array
#		 local_xi: local coordinates, 4x1, usually gaussian nodes, the point to evaluate in the parametric space
# output: X: point in the physical space

def IsoparametricMap(p,P, local_xi):
	X = np.zeros((len(local_xi),1)) # initialize 
	shape_function = Shape_Function(p,local_xi) # evaluate shape function at local_xi
	for i in [0,1,2]:
		X[i] = np.dot(P[:,i],shape_function)
	return X


