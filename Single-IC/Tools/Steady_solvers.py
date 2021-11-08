from Tools.Mat_construction import *
from Tools.commons import *
import numpy as np
from scipy.linalg import eigh

pi = np.pi

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


# Solve Kd = F, the steady state
def Steady_Elasticity_solver(p, Cells, Points, Dirichlet, elas, t=None, Facets=None, Neumann=None):
	M,K,F = Global_Assembly(p,Cells, Points, Dirichlet, elas, t, steady=True)
	# Taking care of dirichlet BC
	for i in range(len(Points)*3):
		for A in [0,1,2]:
			dirich = (node_to_dof(3, [A] , [i]))[0] # global equation number, global dof
			if dirich in Dirichlet:
				K[dirich,dirich] = 1
				F[dirich]      = 0
	return np.linalg.solve(K,F)


def Eigen_mode(deg, Cells, Points, Dirichlet, elas, t=None, Facets=None, Neumann=None):
	
	M,K,_ = Global_Assembly(deg, Cells, Points, Dirichlet, elas, t, steady=False)
	for i in range(len(Points)*3):
		for A in [0,1,2]:
			dirich = (node_to_dof(3, [A] , [i]))[0] # global equation number, global dof
			if dirich in Dirichlet:
				M[dirich,dirich] = 1

	# print(is_pos_def(M))
	# print(is_pos_def(K))

	omega_sq, _ = eigh(K, M)
	print((omega_sq**(0.5)/2/pi)[0:50])

	return 0



