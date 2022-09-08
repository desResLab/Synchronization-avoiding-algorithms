# this code contains some useful common functions

import numpy as np


# linear ramp function, ends at t=1s
def linear_ramp(t):
	if t<= 1:
		return t
	else:
		return 1.0


# Build class of problem-dependent parameters
class elasticity:
	# gather lame parameters and density 
	def __init__(self, lmd, mu, rho, fz, R):
		self.lmd = lmd
		self.mu  = mu
		self.rho = rho
		self.fz  = fz
		self.R   = R

	# compute elasticity coefficient matrix
	def D(self): 
		return np.array([[self.lmd + 2.0*self.mu , self.lmd , self.lmd, 0.0, 0.0, 0.0] \
						,[self.lmd , self.lmd + 2.0*self.mu, self.lmd, 0.0, 0.0, 0.0] \
						,[self.lmd , self.lmd , self.lmd + 2.0*self.mu, 0.0, 0.0, 0.0] \
						,[0.0, 0.0, 0.0, self.mu, 0.0, 0.0] \
						,[0.0, 0.0, 0.0, 0.0, self.mu, 0.0] \
						,[0.0, 0.0, 0.0, 0.0, 0.0, self.mu]])
	

	# volumetric loading density, now the loading is in y-z plane
	def f(self,X,t):  
		if self.R == False:
			return np.array([[0.0],
							 [-self.fz],
							 [-self.fz]])
		else:
			return np.array([[0.0],[-self.fz*linear_ramp(t)],[-self.fz*linear_ramp(t)]])




# build time integration class for displacement based solver
class Time_integration_displacement:
	def __init__(self, tn, dt, d0, dn):
		self.tn   = tn # current time
		self.dt   = dt # timestep size
		self.d0   = d0    # dn
		self.dn   = dn  # dn-1

	def tn_plus_1(self):
		return self.tn + self.dt



# this function find the global dof number of a global node
# input:
#      d: dimension
#      ls: list of constrained direction 
#      P: list of global node id
# output:
#      list of constrained global dof
def node_to_dof(d, ls, P):
	global_dof = []
	for g in P:
		for i in ls:
			global_dof.append( int( d*g ) + i )
	return global_dof

# this function computes the minimal tetrahedron element size by inscribed sphere
# input: 
#        Element: all elements of the mesh
#        Points: all points of the mesh
# output: he: minimal mesh size
# Note: by vtk's format, the first four nodes are the vertices, which can be seen from the basis function labels
def Meshsize(Element, Points):
	edge_list = []
	for ele in Element:
		P   = np.array([Points[ele[0]],Points[ele[1]],Points[ele[2]],Points[ele[3]] ]) # locate 4 node
		min_edge_length = min(  np.linalg.norm(P[0,:]-P[1,:]), \
							np.linalg.norm(P[1,:]-P[2,:]), \
							np.linalg.norm(P[2,:]-P[3,:]), \
							np.linalg.norm(P[1,:]-P[3,:]), \
							np.linalg.norm(P[0,:]-P[3,:]),\
							np.linalg.norm(P[0,:]-P[2,:]) )
		edge_list.append(min_edge_length)
	return 2.0*min(edge_list)/np.sqrt(24)




# Naive lumping method by row summing to get a diagonal matrix
def lumping(M):
	M_lumped = np.zeros((len(M), len(M)))
	for i in range(len(M)):
		M_lumped[i,i] = np.sum(M[i,:])
	return M_lumped

# Naive lumping method by row summing to get a vector
def lumping_to_vec(M):
	M_lumped_vec = np.zeros((len(M), 1))
	for i in range(len(M)):
		M_lumped_vec[i] = np.sum(M[i,:])
	return M_lumped_vec


# The dirac delta function
def dirac(i,j):
	if i == j:
		return 1.0
	else:
		return 0.0

# cartesian basis vector
def basis(i):
	if i == 0:
		return np.array([1,0,0])
	if i == 1:
		return np.array([0,1,0])
	if i == 2:
		return np.array([0,0,1])
	else:
		print('Not a basis!')
